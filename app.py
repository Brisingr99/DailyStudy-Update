import streamlit as st
import requests
import datetime
import xml.etree.ElementTree as ET
import os
import re
import google.generativeai as genai

# --- Konfiguration der Seite ---
st.set_page_config(
    page_title="Tägliches Med-Update",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medizinisches Studien-Update")
st.caption("Aktuelle Erkenntnisse aus Kardiologie, Pneumologie, Gastroenterologie, Endokrinologie & Innerer Medizin")

def get_clean_xml_text(node):
    """Extrahiert den gesamten Text eines XML-Knotens inkl. Unterelementen."""
    if node is not None:
        return "".join(node.itertext()).strip()
    return ""

# --- PubMed API Abfragen ---
@st.cache_data(ttl=86400)
def fetch_pubmed_ids():
    """Holt die Liste der PubMed-IDs für die Zielbereiche der letzten 10 Jahre."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": "(Cardiology OR Pneumology OR Gastroenterology OR Endocrinology OR Internal Medicine) AND HASABSTRACT",
        "reldate": 3650,
        "retmode": "json",
        "retmax": 500,
        "sort": "relevance"
    }
    headers = {"User-Agent": "MedUpdateApp/1.0"}
    
    try:
        res = requests.get(search_url, params=params, headers=headers, timeout=10)
        data = res.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

@st.cache_data(ttl=86400)
def fetch_single_study_xml(pmid):
    """Holt Titel, Abstract und Journal für eine spezifische PMID."""
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    headers = {"User-Agent": "MedUpdateApp/1.0"}
    
    try:
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
        
        if title and len(abstract) > 100:
            return title, abstract, journal
    except Exception:
        pass
    return None, None, None

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

# --- Gemini KI API Helper ---
def call_gemini_api(prompt, api_key):
    """Hilfsfunktion zum Ausführen von Gemini Prompts mit automatischer Modellauswahl."""
    genai.configure(api_key=api_key)
    
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

def summarize_with_ai(title, abstract, api_key):
    """Generiert eine strukturierte deutsche Zusammenfassung der Studie."""
    prompt = f"""
    Du bist ein Experte für medizinische Fachliteratur. Fasse die folgende medizinische Studie präzise, fachlich korrekt und auf Deutsch zusammen.
    
    Titel: {title}
    Abstract: {abstract}
    
    Strukturiere die Antwort EXAKT in folgende Abschnitte:
    1. **Fachbereich & Hauptthema** (z. B. Kardiologie: Koronare Herzkrankheit, Pneumologie: Lungenkarzinom)
    2. **Hintergrund & Fragestellung**
    3. **Methodik & Studiendesign**
    4. **Zentrale Ergebnisse**
    5. **Klinische Relevanz / Fazit für die Praxis**
    6. **Schlagwort**: [Ein prägnantes medizinisches Haupt-Schlagwort/Erkrankung, z. B. "Lungenkarzinom", "KHK", "Asthma", "Heart Failure"]
    """
    return call_gemini_api(prompt, api_key)

def summarize_keyword(keyword, api_key):
    """Generiert eine klinische Kurzzusammenfassung zum medizinischen Hauptthema."""
    prompt = f"""
    Du bist ein erfahrener Facharzt der Inneren Medizin. Erstelle eine prägnante, klinisch orientierte Kurzzusammenfassung zum Krankheitsbild: "{keyword}".
    
    Gliedere die Antwort präzise in:
    - **Definition & Leitsymptome**
    - **Diagnostik & Hauptparameter**
    - **Therapieprinzipien / Standardtherapie**
    
    Halte dich kurz und übersichtlich (max. 150 Wörter).
    """
    return call_gemini_api(prompt, api_key)

def extract_main_topic(summary_text):
    """Extrahiert das Haupt-Schlagwort sauber per Regex aus dem KI-Text."""
    # 1. Versuch: Punkt 6 (Schlagwort)
    match_6 = re.search(r"6\.\s*\*\*Schlagwort[^*]*\*\*:?\s*(.+)", summary_text, re.IGNORECASE)
    if match_6:
        raw_kw = match_6.group(1).strip()
        clean_kw = re.sub(r"[\*\_]", "", raw_kw).strip().split("\n")[0].split(",")[0].strip()
        if clean_kw and len(clean_kw) < 40:
            return clean_kw

    # 2. Versuch: Punkt 1 (Fachbereich & Hauptthema)
    match_1 = re.search(r"1\.\s*\*\*Fachbereich[^*]*\*\*:?\s*(.+)", summary_text, re.IGNORECASE)
    if match_1:
        raw_topic = match_1.group(1).strip()
        clean_topic = re.sub(r"[\*\_]", "", raw_topic).strip().split("\n")[0].split("(")[0].strip()
        if clean_topic and len(clean_topic) < 40:
            return clean_topic

    return "Hauptthema"

# --- Session State Management ---
id_list = fetch_pubmed_ids()

if "study_index" not in st.session_state:
    day_seed = int(datetime.date.today().strftime("%Y%m%d"))
    st.session_state.study_index = (day_seed % len(id_list)) if id_list else 0

# --- Header mit Aktualisierungs-Button ---
col_head1, col_head2 = st.columns([3, 1])

with col_head1:
    st.subheader(f"📅 Studie des Tages ({datetime.date.today().strftime('%d.%m.%Y')})")

with col_head2:
    if st.button("🎲 Weitere Studie laden", use_container_width=True):
        st.session_state.study_index += 1
        if "kw_summary_text" in st.session_state:
            del st.session_state["kw_summary_text"]
        st.rerun()

# --- Studie ermitteln ---
current_pmid = None
title, abstract, journal = None, None, None

if id_list:
    attempts = 0
    while attempts < 30:
        candidate_pmid = id_list[st.session_state.study_index % len(id_list)]
        t, a, j = fetch_single_study_xml(candidate_pmid)
        if t and a:
            current_pmid = candidate_pmid
            title, abstract, journal = t, a, j
            break
        st.session_state.study_index += 1
        attempts += 1

# --- Hauptanzeige ---
if current_pmid:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(f"### {title}")
        st.caption(f"**Journal:** {journal} | **PMID:** [{current_pmid}](https://pubmed.ncbi.nlm.nih.gov/{current_pmid}/)")
        
        api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            api_key = st.text_input("Bitte Gemini API Key eingeben:", type="password")
            
        if api_key:
            with st.spinner("Zusammenfassung der Studie wird erstellt..."):
                try:
                    summary = summarize_with_ai(title, abstract, api_key)
                    main_topic = extract_main_topic(summary)
                    
                    st.markdown("---")
                    st.markdown(summary)
                    
                    with col_right:
                        st.markdown("### 📊 Thema & Schema")
                        img_url = fetch_schema_image(main_topic)
                        if img_url:
                            st.image(img_url, caption=f"Schematische Übersicht zum Thema: {main_topic}", use_container_width=True)
                        else:
                            st.info(f"Kein direktes Schema zu „{main_topic}“ in Open-Access-Datenbanken gefunden.")
                        
                        st.markdown("---")
                        
                        if st.button(f"💡 Kurzzusammenfassung zu „{main_topic}“", use_container_width=True):
                            with st.spinner(f"Kurzzusammenfassung zu {main_topic} wird geladen..."):
                                st.session_state.kw_summary_text = summarize_keyword(main_topic, api_key)
                        
                        if "kw_summary_text" in st.session_state:
                            st.info(st.session_state.kw_summary_text)
                        
                        with st.expander("Original Abstract (Englisch) anzeigen"):
                            st.write(abstract)
                except Exception as e:
                    st.error(f"Fehler bei der KI-Generierung: {e}")
        else:
            st.warning("Bitte einen API-Schlüssel angeben, um die tägliche deutsche Zusammenfassung zu generieren.")

else:
    st.error("Es konnten keine weiteren Studien mit Abstract abgerufen werden.")
