import streamlit as st
import requests
import random
import re
import os
import google.generativeai as genai

# --- Konfiguration der Seite ---
st.set_page_config(
    page_title="Klinisches Refresher-Tool",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Klinisches Refresher-Tool: Krankheitsbilder")
st.caption("Kompakte & strukturierte Übersichten für die Innere Medizin")

# --- Umfangreicher Pool an Krankheitsbildern ---
DISEASES = {
    "Kardiologie": [
        "Aortenklappenstenose", "Vorhofflimmern", "Herzinsuffizienz (HFrEF)", 
        "Akutes Koronarsyndrom (NSTEMI/STEMI)", "Myokarditis", "Hypertonie", 
        "Infektiöse Endokarditis", "AV-Knoten-Reentry-Tachykardie (AVNRT)"
    ],
    "Pneumologie": [
        "COPD-Exazerbation", "Bronchialkarzinom", "Idiopatische Lungenfibrose", 
        "Lungenarterienembolie", "Asthma bronchiale", "Ambulant erworbene Pneumonie", 
        "Sarkoidose", "Pneumothorax"
    ],
    "Gastroenterologie & Hepatologie": [
        "Colitis ulcerosa", "Morbus Crohn", "Dekompensierte Leberzirrhose", 
        "Akute Pankreatitis", "Refluxkrankheit (GERD)", "Zöliakie", 
        "Hepatozelluläres Karzinom (HCC)", "Gastrointestinale Blutung"
    ],
    "Endokrinologie & Diabetologie": [
        "Diabetes mellitus Typ 2", "Hyperthyreose / Morbus Basedow", 
        "Primärer Hyperparathyroidismus", "Cushing-Syndrom", 
        "Nebennierenrindeninsuffizienz (Morbus Addison)", "Diabetische Ketoazidose"
    ],
    "Nephrologie & Rheumatologie": [
        "Akutes Nierenversagen", "Chronische Nierenerkrankung (CKD)", 
        "Glomerulonephritis", "Rheumatoide Arthritis", "Systemischer Lupus erythematodes", 
        "Gichtarthritis", "ANCA-assoziierte Vaskulitis"
    ]
}

# Flache Liste aller Krankheitsbilder für die Zufallsauswahl
ALL_DISEASES = [disease for group in DISEASES.values() for disease in group]

# --- Wikimedia Commons Schemasuche ---
@st.cache_data(ttl=86400)
def fetch_schema_image(english_keyword):
    """Sucht nach medizinischen Diagrammen bei Wikimedia Commons."""
    if not english_keyword or english_keyword == "Disease":
        return None
        
    url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": "MedRefresherApp/1.0"}
    
    search_queries = [
        f"{english_keyword} diagram",
        f"{english_keyword} anatomy",
        f"{english_keyword} schema",
        english_keyword
    ]
    
    for query_term in search_queries:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"File:{query_term}",
            "gsrlimit": 5,
            "prop": "imageinfo",
            "iiprop": "url|mime"
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=5).json()
            pages = r.get("query", {}).get("pages", {})
            for _, page in pages.items():
                imageinfo = page.get("imageinfo", [])
                if imageinfo:
                    img_url = imageinfo[0].get("url", "")
                    mime = imageinfo[0].get("mime", "")
                    if any(valid_type in mime for valid_type in ["image/jpeg", "image/png", "image/svg+xml", "image/webp"]):
                        return img_url
        except Exception:
            continue
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
    """Generiert eine strukturierte medizinische Übersicht zu einem Krankheitsbild."""
    prompt = f"""
    Du bist ein erfahrener Facharzt für Innere Medizin. Erstelle eine fundierte, hochgradig praxistaugliche und strukturierte Übersicht zum Krankheitsbild: "{disease_name}".
    
    Gliedere die Antwort präzise in folgende Abschnitte:
    1. **Kurzzusammenfassung & Definition**
    2. **Epidemiologie, Ätiologie & Pathophysiologie**
    3. **Klinik & Leitsymptome**
    4. **Diagnostik** (Labor, Bildgebung, Staging / Funktionsdiagnostik)
    5. **Therapie & Leitlinienprinzipien** (Konservativ, Pharmakologisch, Interventionell/Operativ)
    6. **Wichtige Differenzialdiagnosen**

    Gib am Ende EXAKT diese Steuerzeile aus:
    SCHLAGWORT_EN: [Englisches Haupt-Suchwort für Anatomie/Schema, z.B. "Aortic stenosis", "Atrial fibrillation", "Pulmonary embolism", "Crohn disease"]
    """
    return call_gemini_api(prompt, api_key)

def extract_english_keyword(text):
    """Extrahiert das englische Schlagwort für die Bildersuche."""
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
    with st.spinner(f"Erstelle klinische Übersicht zu „{st.session_state.selected_disease}“..."):
        try:
            raw_text = generate_disease_summary(st.session_state.selected_disease, api_key)
            english_kw = extract_english_keyword(raw_text)
            display_text = clean_display_text(raw_text)
            
            col_main, col_side = st.columns([2.2, 1])
            
            with col_main:
                st.markdown(display_text)
                
            with col_side:
                st.markdown("### 📊 Schemata & Anatomie")
                img_url = fetch_schema_image(english_kw)
                if img_url:
                    st.image(img_url, caption=f"Schema zu: {st.session_state.selected_disease} ({english_kw})", use_container_width=True)
                else:
                    st.info(f"Kein freies Schema zu „{english_kw}“ bei Wikimedia Commons gefunden.")
                    
        except Exception as e:
            st.error(f"Fehler bei der Generierung: {e}")
else:
    st.warning("Bitte gib deinen API-Schlüssel ein, um die Inhalte zu laden.")
