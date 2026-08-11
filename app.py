import streamlit as st
import requests
import random
import re
import os
import urllib.parse
import google.generativeai as genai

# --- Konfiguration der Seite ---
st.set_page_config(
    page_title="Klinisches Refresher-Tool",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Klinisches Refresher-Tool: Krankheitsbilder")
st.caption("Umfassende & strukturierte Facharzt-Übersichten (im Stil von AMBOSS / DocCheck)")

# --- Umfangreicher Pool an Krankheitsbildern ---
DISEASES = {
    "Kardiologie": [
        "Aortenklappenstenose", "Vorhofflimmern", "Herzinsuffizienz", 
        "Akutes Koronarsyndrom", "Myokarditis", "Kardiogener Schock", 
        "Infektiöse Endokarditis", "AV-Knoten-Reentry-Tachykardie"
    ],
    "Pneumologie": [
        "COPD", "Bronchialkarzinom", "Idiopatische Lungenfibrose", 
        "Lungenarterienembolie", "Asthma bronchiale", "Pneumonie", 
        "Sarkoidose", "Pneumothorax"
    ],
    "Gastroenterologie & Hepatologie": [
        "Colitis ulcerosa", "Morbus Crohn", "Leberzirrhose", 
        "Akute Pankreatitis", "Gastroösophageale Refluxkrankheit", "Zöliakie", 
        "Hepatozelluläres Karzinom", "Gastrointestinale Blutung"
    ],
    "Endokrinologie & Diabetologie": [
        "Diabetes mellitus", "Hyperthyreose", 
        "Hyperparathyreoidismus", "Cushing-Syndrom", 
        "Nebennierenrindeninsuffizienz", "Diabetische Ketoazidose"
    ],
    "Nephrologie & Rheumatologie": [
        "Akutes Nierenversagen", "Chronische Nierenerkrankung", 
        "Glomerulonephritis", "Rheumatoide Arthritis", "Systemischer Lupus erythematodes", 
        "Gicht", "ANCA-assoziierte Vaskulitis"
    ]
}

ALL_DISEASES = [disease for group in DISEASES.values() for disease in group]

# --- Kuratierte Multi-Bildsuche via Wikipedia REST-API ---
@st.cache_data(ttl=86400)
def fetch_wikipedia_images(disease_de, disease_en, max_images=4):
    """Holt bis zu max_images relevante Abbildungen/Schemata aus dem DE & EN Wikipedia-Artikel."""
    headers = {"User-Agent": "MedRefresherApp/1.0 (medical_education_app)"}
    image_urls = []
    
    EXCLUDED_TERMS = [
        "icon", "logo", "flag", "symbol", "stub", "wikisource", "wikimedia", 
        "commons", "question", "edit", "ambox", "disambig", "padlock", "portal", "svg"
    ]

    def extract_from_article(title, lang="de"):
        urls = []
        if not title or title == "None":
            return urls
        try:
            encoded_title = urllib.parse.quote(title)
            
            # 1. Hauptbild/Titelbild abfragen
            summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
            s_res = requests.get(summary_url, headers=headers, timeout=5)
            if s_res.status_code == 200:
                s_data = s_res.json()
                main_img = s_data.get("originalimage", {}).get("source") or s_data.get("thumbnail", {}).get("source")
                if main_img and not any(ex in main_img.lower() for ex in EXCLUDED_TERMS):
                    urls.append(main_img)

            # 2. Relevante Medienliste der Seite abfragen
            media_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/media-list/{encoded_title}"
            m_res = requests.get(media_url, headers=headers, timeout=5)
            if m_res.status_code == 200:
                m_data = m_res.json()
                for item in m_data.get("items", []):
                    if item.get("type") == "image":
                        srcset = item.get("srcset", [])
                        img_src = srcset[-1].get("src") if srcset else None
                        
                        if not img_src:
                            file_title = item.get("title", "")
                            if file_title.startswith("File:") or file_title.startswith("Datei:"):
                                clean_name = file_title.split(":", 1)[1].strip()
                                img_src = f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(clean_name)}?width=800"
                        
                        if img_src and img_src.startswith("//"):
                            img_src = "https:" + img_src
                            
                        if img_src and not any(ex in img_src.lower() for ex in EXCLUDED_TERMS):
                            if img_src not in urls:
                                urls.append(img_src)
                                if len(urls) >= max_images:
                                    break
        except Exception:
            pass
        return urls

    # 1. Deutsche Wikipedia durchsuchen
    if disease_de:
        image_urls = extract_from_article(disease_de, lang="de")
    
    # 2. Falls noch Plätze frei sind, mit englischem Wikipedia-Artikel auffüllen
    if len(image_urls) < max_images and disease_en and disease_en != "Disease":
        en_urls = extract_from_article(disease_en, lang="en")
        for u in en_urls:
            if u not in image_urls:
                image_urls.append(u)
            if len(image_urls) >= max_images:
                break
                
    return image_urls[:max_images]

# --- Gemini KI API Helper ---
def call_gemini_api(prompt, api_key):
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
        raise RuntimeError("Kein passendes Gemini-Modell verfügbar.")

def generate_disease_summary(disease_name, api_key):
    """Generiert einen tiefgehenden, detaillierten Fachtext im Stil von AMBOSS/DocCheck."""
    prompt = f"""
    Du bist ein führender Oberarzt der Inneren Medizin und Verfasser medizinischer Standardwerke (wie AMBOSS oder DocCheck Flexikon).
    Erstelle ein extrem ausführliches, klinikrelevantes und vollständiges Skript zum Thema: "{disease_name}".
    
    WICHTIG: Schreibe nicht nur Gliederungen oder Stichpunkte ohne Inhalt! Fülle JEDEN Punkt mit konkretem klinischem Wissen, exakten Laborwerten, Scores, Wirkstoffen und Behandlungsschemata aus.

    Gliedere den Text exakt in folgende Abschnitte:

    ## 1. Definition & Epidemiologie
    - Genaue medizinische Definition.
    - Epidemiologische Eckdaten (Inzidenz, Prävalenz, Alters-/Geschlechtsverteilung).

    ## 2. Ätiologie & Pathophysiologie
    - Ursachen, Risikofaktoren und Auslöser.
    - Detaillierter pathophysiologischer Ablauf / Entstehungsmechanismus.

    ## 3. Leitsymptome & Klinisches Bild
    - Typische Leitsymptome, Verlaufsformen und Warnzeichen (Red Flags).
    - Komplikationen bei Fortschreiten.

    ## 4. Diagnostischer Pfad
    - **Anamnese & Körperliche Untersuchung** (spezifische Zeichen/Tests).
    - **Labor**: Spezifische Marker, Zielwerte, Differenzialparameter.
    - **Apparative Diagnostik**: EKG, Bildgebung (Röntgen, CT, Sono, Echo) mit typischen Befunden.
    - **Klassifikation & Scores** (falls vorhanden, z.B. NYHA, CURB-65, CHA₂DS₂-VASc, Child-Pugh).

    ## 5. Therapie & Leitlinien
    - **Allgemein- & Akutmaßnahmen**.
    - **Pharmakotherapie**: Konkrete Wirkstoffgruppen, First-Line-Medikamente und Wirkmechanismen.
    - **Interventionelle / Operative Verfahren**: Indikationen und Optionen.

    ## 6. Differenzialdiagnosen
    - Die wichtigsten 3–5 Differenzialdiagnosen mit prägnantem Unterscheidungsmerkmal.

    Gib ganz am Ende EXAKT diese zwei Steuerzeilen aus:
    SCHLAGWORT_DE: [Der exakte deutsche Wikipedia-Artikelname für dieses Thema, z.B. "Aortenklappenstenose", "Pneumonie", "Myokardinfarkt"]
    SCHLAGWORT_EN: [Der exakte englische Wikipedia-Artikelname für dieses Thema, z.B. "Aortic stenosis", "Pneumonia", "Myocardial infarction"]
    """
    return call_gemini_api(prompt, api_key)

def extract_keywords(text, default_de):
    """Extrahiert die korrekten DE- und EN-Wikipedia-Artikelnamen aus den Steuerzeilen."""
    de_kw = default_de
    en_kw = "Disease"
    
    if "SCHLAGWORT_DE:" in text:
        raw_de = text.split("SCHLAGWORT_DE:")[1].split("\n")[0]
        de_kw = re.sub(r"[\[\]\*\_\.\"]", "", raw_de).strip()
        
    if "SCHLAGWORT_EN:" in text:
        raw_en = text.split("SCHLAGWORT_EN:")[1].split("\n")[0]
        en_kw = re.sub(r"[\[\]\*\_\.\"]", "", raw_en).strip()
        
    return de_kw, en_kw

def clean_display_text(text):
    """Entfernt die Steuerzeilen aus der Anzeige."""
    text = text.split("SCHLAGWORT_DE:")[0]
    text = text.split("SCHLAGWORT_EN:")[0]
    return text.strip()

# --- Session State Initialisierung ---
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = random.choice(ALL_DISEASES)

# --- Sidebar / Steuerung ---
st.sidebar.header("⚙️ Steuerung & Filter")

category_filter = st.sidebar.selectbox(
    "Fachbereich auswählen:",
    ["Alle Fachbereiche"] + list(DISEASES.keys())
)

if st.sidebar.button("🎲 Zufälliges Krankheitsbild laden", use_container_width=True):
    if category_filter == "Alle Fachbereiche":
        st.session_state.selected_disease = random.choice(ALL_DISEASES)
    else:
        st.session_state.selected_disease = random.choice(DISEASES[category_filter])
    st.rerun()

st.sidebar.markdown("---")

# --- Manuelle Suche mit st.form (Enter-Taste & Button unterstützt) ---
with st.sidebar.form("search_form", clear_on_submit=False):
    manual_input = st.text_input("Manuelle Suche (beliebiges Thema):", placeholder="z. B. Herzinfarkt, Hypertonie...")
    search_submitted = st.form_submit_button("🔍 Suchen", use_container_width=True)
    
    if search_submitted and manual_input.strip():
        st.session_state.selected_disease = manual_input.strip()
        st.rerun()

# --- Hauptbereich ---
st.subheader(f"📋 Refresher: {st.session_state.selected_disease}")

api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = st.text_input("Bitte Gemini API Key eingeben:", type="password")

if api_key:
    with st.spinner(f"Erstelle umfassenden Fachtext zu „{st.session_state.selected_disease}“..."):
        try:
            raw_text = generate_disease_summary(st.session_state.selected_disease, api_key)
            de_kw, en_kw = extract_keywords(raw_text, st.session_state.selected_disease)
            display_text = clean_display_text(raw_text)
            
            col_main, col_side = st.columns([2.3, 1])
            
            with col_main:
                st.markdown(display_text)
                
            with col_side:
                st.markdown("### 📊 Schemata & Abbildungen")
                img_urls = fetch_wikipedia_images(de_kw, en_kw)
                
                if img_urls:
                    st.caption(f"{len(img_urls)} relevante Abbildung(en) geladen:")
                    for idx, url in enumerate(img_urls, start=1):
                        st.image(
                            url, 
                            caption=f"Abbildung {idx}: {st.session_state.selected_disease}", 
                            use_container_width=True
                        )
                else:
                    st.info(f"Keine direkten Abbildungen im Wikipedia-Artikel zu „{st.session_state.selected_disease}“ gefunden.")
                    
        except Exception as e:
            st.error(f"Fehler bei der Generierung: {e}")
else:
    st.warning("Bitte gib deinen API-Schlüssel ein, um die Inhalte zu laden.")
