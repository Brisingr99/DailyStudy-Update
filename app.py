import streamlit as st
import requests
import datetime
import xml.etree.ElementTree as ET
import os
import google.generativeai as genai

# --- Konfiguration der Seite ---
st.set_page_config(
    page_title="Tägliches Med-Update",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Tägliches Medizinisches Studien-Update")
st.caption("Aktuelle Erkenntnisse aus Kardiologie, Pneumologie, Gastroenterologie, Endokrinologie & Innerer Medizin")

def get_clean_xml_text(node):
    """Extrahiert den gesamten Text eines XML-Knotens inkl. Unterelementen (z. B. <i>, <b>)."""
    if node is not None:
        return "".join(node.itertext()).strip()
    return ""

# --- PubMed API Abfrage ---
@st.cache_data(ttl=86400)
def fetch_daily_pubmed_study():
    """Holt eine aktuelle Studie mit Volltext-Abstract aus den Ziel-Fachbereichen der letzten 10 Jahre."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": "(Cardiology OR Pneumology OR Gastroenterology OR Endocrinology OR Internal Medicine) AND HASABSTRACT",
        "reldate": 3650,
        "retmode": "json",
        "retmax": 500,
        "sort": "relevance"
    }
    
    headers = {
        "User-Agent": "MedUpdateApp/1.0"
    }
    
    try:
        res = requests.get(search_url, params=params, headers=headers, timeout=10)
        data = res.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return None, None, None, None

        day_seed = int(datetime.date.today().strftime("%Y%m%d"))
        start_idx = day_seed % len(id_list)
        
        # Wir testen bis zu 15 Studien, bis wir eine mit vollständigem Abstract finden
        for i in range(15):
            selected_pmid = id_list[(start_idx + i) % len(id_list)]
            
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": selected_pmid,
                "retmode": "xml"
            }
            
            fetch_res = requests.get(fetch_url, params=fetch_params, headers=headers, timeout=10)
            root = ET.fromstring(fetch_res.content)
            
            title_node = root.find(".//ArticleTitle")
            title = get_clean_xml_text(title_node)
            
            abstract_nodes = root.findall(".//AbstractText")
            abstract_parts = []
            for node in abstract_nodes:
                label = node.get("Label", "")
                text = get_clean_xml_text(node)
                if text:
                    if label:
                        abstract_parts.append(f"**{label}:** {text}")
                    else:
                        abstract_parts.append(text)
            
            abstract = "\n\n".join(abstract_parts)
            
            journal_node = root.find(".//Title")
            journal = get_clean_xml_text(journal_node) or "Unbekanntes Journal"
            
            # Nur zurückgeben, wenn ein valider Titel und ein aussagekräftiger Abstract vorliegen (>100 Zeichen)
            if title and len(abstract) > 100:
                return selected_pmid, title, abstract, journal
        
        return None, None, None, None
        
    except Exception:
        return None, None, None, None

# --- Abbildung / Schema Abfrage (Wikimedia Commons API) ---
@st.cache_data(ttl=86400)
def fetch_schema_image(keyword):
    """Sucht nach einem passenden medizinischen Schema/Diagramm bei Wikimedia Commons."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{keyword} diagram OR schema OR anatomy",
        "gsrlimit": 1,
        "prop": "imageinfo",
        "iiprop": "url"
    }
    try:
        r = requests.get(url, params=params, timeout=10).json()
        pages = r.get("query", {}).get("pages", {})
        for _, page in pages.items():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                return imageinfo[0]["url"]
    except Exception:
        pass
    return None

# --- KI-Zusammenfassung generieren mit dynamischer Modellerkennung ---
def summarize_with_ai(title, abstract, api_key):
    """Generiert eine deutsche Zusammenfassung und wählt dynamisch ein funktionierendes Modell."""
    genai.configure(api_key=api_key)
    
    prompt = f"""
    Du bist ein Experte für medizinische Fachliteratur. Fasse die folgende medizinische Studie präzise, fachlich korrekt und auf Deutsch zusammen.
    
    Titel: {title}
    Abstract: {abstract}
    
    Strukturiere die Antwort in folgende Abschnitte:
    1. **Fachbereich & Hauptthema** (Kardiologie, Pneumologie, Gastro, Endokrinologie oder Innere Medizin)
    2. **Hintergrund & Fragestellung**
    3. **Methodik & Studiendesign**
    4. **Zentrale Ergebnisse**
    5. **Klinische Relevanz / Fazit für die Praxis**
    6. **Schlagwort für Schema-Suche** (Ein einziges englisches Haupt-Suchwort zur Erkrankung/Anatomie, z.B. "Heart failure", "Asthma", "Cirrhosis")
    """
    
    available_models = []
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available_models.append(m.name.replace("models/", ""))
    except Exception:
        pass
    
    candidates = available_models + ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    seen = set()
    unique_candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    
    last_error = None
    for model_name in unique_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            continue
            
    if last_error:
        raise last_error
    else:
        raise RuntimeError("Kein passendes Gemini-Modell für diesen API-Schlüssel gefunden.")

# --- Hauptanwendungslogik ---
pmid, title, abstract, journal = fetch_daily_pubmed_study()

if pmid:
    st.subheader(f"📅 Studie des Tages ({datetime.date.today().strftime('%d.%m.%Y')})")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(f"### {title}")
        st.caption(f"**Journal:** {journal} | **PMID:** [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
        
        api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            api_key = st.text_input("Bitte Gemini API Key eingeben:", type="password")
            
        if api_key:
            with st.spinner("Zusammenfassung wird erstellt..."):
                try:
                    summary = summarize_with_ai(title, abstract, api_key)
                    
                    st.markdown("---")
                    st.markdown(summary)
                    
                    search_kw = title.split()[0]
                    if "Schlagwort für Schema-Suche:" in summary:
                        search_kw = summary.split("Schlagwort für Schema-Suche:")[-1].strip().split("\n")[0]
                    
                    with col_right:
                        st.markdown("### 📊 Thema / Schemata")
                        img_url = fetch_schema_image(search_kw)
                        if img_url:
                            st.image(img_url, caption=f"Schematische Übersicht zum Thema: {search_kw}", use_container_width=True)
                        else:
                            st.info("Kein direktes Schema in Open-Access-Datenbanken gefunden.")
                        
                        with st.expander("Original Abstract (Englisch) anzeigen"):
                            st.write(abstract)
                except Exception as e:
                    st.error(f"Fehler bei der KI-Generierung: {e}")
        else:
            st.warning("Bitte einen API-Schlüssel angeben, um die tägliche deutsche Zusammenfassung zu generieren.")

else:
    st.error("Heute konnten keine vollständigen Studien mit Abstract abgerufen werden. Bitte versuche es später erneut.")
