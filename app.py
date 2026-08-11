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
        "Akutes Koronarsyndrom", "Myokarditis", "Kardiogenes Schock", 
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

# --- Kuratierte Bildsuche via Wikipedia REST-API (DE & EN) ---
@st.cache_data(ttl=86400)
def fetch_wikipedia_image(disease_de, disease_en):
    """Holt das kuratierte Hauptbild des entsprechenden Wikipedia-Artikels."""
    headers = {"User-Agent": "MedRefresherApp/1.0 (medical_education_app)"}
    
    # 1. Versuch: Deutsche Wikipedia
    try:
        encoded_de = urllib.parse.quote(disease_de)
        url_de = f"https://de.wikipedia.org/api/rest_v1/page/summary/{encoded_de}"
        res = requests.get(url_de, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "originalimage" in data:
                return data["originalimage"]["source"]
            elif "thumbnail" in data:
                return data["thumbnail"]["source"]
    except Exception:
        pass

    # 2. Versuch: Englische Wikipedia (falls DE kein Bild hat oder Artikelname abweicht)
    if disease_en and disease_en != "Disease":
        try:
            encoded_en = urllib.parse.quote(disease_en)
            url_en = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_en}"
            res = requests.get(url_en, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if "originalimage" in data:
                    return data["originalimage"]["source"]
                elif "thumbnail" in data:
                    return data["thumbnail"]["source"]
        except Exception:
            pass

    return None

# --- Gemini KI API Helper ---
def call_gemini_api(prompt, api_key):
    """Führt Prompts mit automatischer Modellauswahl aus."""
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
    Erstelle ein extrem ausführliches, klinikrelevantes und vollständiges Skript zum Krankheitsbild: "{disease_name}".
    
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

    Gib ganz am Ende EXAKT diese Steuerzeile aus:
    SCHLAGWORT_EN: [Der exakte englische Wikipedia-Artikelname für dieses Thema, z.B. "Aortic stenosis", "Atrial fibrillation", "Pulmonary embolism", "Crohn's disease"]
    """
    return call_gemini_api(prompt, api_key)

def extract_english_keyword(text):
    """Extrahiert den englischen Begriff für die Wikipedia-Suche."""
    if "SCHLAGWORT_EN:" in text:
        raw = text.split("SCHLAGWORT_EN:")[-1].split("\n")[0]
        return re.sub(r"[\[\]\*\_\.\"]", "", raw).strip()
    return "Disease"

def clean_display_text(text):
    """Entfernt die Steuerzeile aus der Anzeige."""
    return text.split("SCHLAGWORT_EN:")[0].strip()

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
manual_input = st.sidebar.text_input("Manuelle Suche (beliebiges Thema):")
if st.sidebar.button("🔍 Suchen", use_container_width=True) and manual_input:
    st.session_state.selected_disease = manual_input
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
            english_kw = extract_english_keyword(raw_text)
            display_text = clean_display_text(raw_text)
            
            col_main, col_side = st.columns([2.3, 1])
            
            with col_main:
                st.markdown(display_text)
                
            with col_side:
                st.markdown("### 📊 Schema / Abbildung")
                img_url = fetch_wikipedia_image(st.session_state.selected_disease, english_kw)
                if img_url:
                    st.image(img_url, caption=f"Kuratierte Abbildung aus Wikipedia: {st.session_state.selected_disease}", use_container_width=True)
                else:
                    st.info(f"Keine direkte Abbildung im Wikipedia-Artikel zu „{st.session_state.selected_disease}“ vorhanden.")
                    
        except Exception as e:
            st.error(f"Fehler bei der Generierung: {e}")
else:
    st.warning("Bitte gib deinen API-Schlüssel ein, um die Inhalte zu laden.")
