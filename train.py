import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime.scriptrunner_utils")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import time

# ======== Cyberpunk Theme ========
def set_cyberpunk_theme():
    st.markdown("""
    <style>
    :root {
        --primary: #00ff9d;
        --secondary: #ff00aa;
        --bg: #0e1117;
        --text: #f0f0f0;
    }
    
    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Courier New', monospace;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1a1a1a !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 0px !important;
    }
    
    .stButton>button {
        border: 2px solid var(--primary) !important;
        border-radius: 0px !important;
        color: var(--primary) !important;
        background: black !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 10px var(--primary) !important;
        text-shadow: 0 0 5px var(--primary) !important;
    }
    
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #ff00aa, #00ff9d) !important;
    }
    
    .toxic {
        color: #ff00aa !important;
        text-shadow: 0 0 5px #ff00aa;
    }
    
    .offensive {
        color: #ffcc00 !important;
        text-shadow: 0 0 5px #ffcc00;
    }
    
    .safe {
        color: #00ff9d !important;
        text-shadow: 0 0 5px #00ff9d;
    }
    
    .glitch {
        animation: glitch 1s linear infinite;
    }
    
    @keyframes glitch {
        0% { text-shadow: 2px 0 #ff00aa; }
        20% { text-shadow: -2px 0 #00ff9d; }
        40% { text-shadow: 2px 0 #ff00aa; }
        60% { text-shadow: -2px 0 #00ff9d; }
        80% { text-shadow: 2px 0 #ff00aa; }
        100% { text-shadow: -2px 0 #00ff9d; }
    }
    
    .neon-box {
        border: 1px solid var(--primary);
        border-radius: 0px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 10px var(--primary);
    }
    </style>
    """, unsafe_allow_html=True)

# ======== Model Code ========
def load_data(filename='train.csv'):
    try:
        df = pd.read_csv(filename)
        df['toxic'] = df['toxic'] | df['severe_toxic'] | df['threat']
        df['offensive'] = df['obscene'] | df['insult']
        df['safe'] = ~(df['toxic'] | df['offensive'] | df['identity_hate'])
        conditions = [df['toxic'], df['offensive'], df['safe']]
        choices = [2, 1, 0]
        df['label'] = np.select(conditions, choices, default=0)
        return df['comment_text'], df['label']
    except FileNotFoundError:
        st.error(f"Error: Could not find {filename}")
        st.error("Please download the Jigsaw Toxic Comment Classification dataset")
        st.stop()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model():
    X, y = load_data()
    X_clean = [clean_text(text) for text in X]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_vec = vectorizer.fit_transform(X_clean)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_vec, y)
    joblib.dump(vectorizer, 'toxic_vectorizer.joblib')
    joblib.dump(clf, 'toxic_classifier.joblib')
    return vectorizer, clf

def classify_comment(comment):
    try:
        vectorizer = joblib.load('toxic_vectorizer.joblib')
        clf = joblib.load('toxic_classifier.joblib')
    except:
        with st.spinner('Training model... (First run only)'):
            vectorizer, clf = train_model()
    
    cleaned = clean_text(comment)
    vec = vectorizer.transform([cleaned])
    pred = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    return {0: 'SAFE', 1: 'OFFENSIVE', 2: 'TOXIC'}[pred], proba

# ======== Cyberpunk UI ========
set_cyberpunk_theme()

# Glitch title
st.markdown("""
<h1 class='glitch'>TOXIC<span style='color:#ff00aa'>.</span>AI<span style='color:#00ff9d'>.</span>SCAN</h1>
<p style='color:#cccccc'>NEURAL NETWORK ANALYSIS TERMINAL</p>
""", unsafe_allow_html=True)

# Neon tabs
selected = option_menu(
    menu_title=None,
    options=["Scanner", "About"],
    icons=["search", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"background-color": "#0e1117", "border": "1px solid #00ff9d"},
        "nav-link": {"color": "#f0f0f0", "--hover-color": "#1a1a1a"},
        "nav-link-selected": {"background-color": "#00ff9d", "color": "#000"}
    }
)

if selected == "Scanner":
    # Cyberpunk input
    comment = st.text_area(
        "ENTER TEXT FOR ANALYSIS", 
        "Type your comment here...",
        height=150
    )
    
    # Holographic button
    if st.button("INITIATE SCAN", key="analyze"):
        if comment.strip() in ["", "Type your comment here..."]:
            st.warning("INPUT REQUIRED")
        else:
            with st.spinner('SCANNING TEXT PATTERNS...'):
                # Scan animation
                scan_bar = st.progress(0)
                for percent in range(100):
                    time.sleep(0.01)
                    scan_bar.progress(percent + 1)
                
                # Get prediction
                pred, proba = classify_comment(comment)
                
                # Cyberpunk result display
                st.markdown(f"""
                <div class='neon-box'>
                    <h2 class='{pred.lower()}' style='font-size: 2em;'>RESULT: {pred}</h2>
                    <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
                        <div style='width: 30%;'>
                            <h4 style='color:#00ff9d'>SAFE</h4>
                            <div style='height: 10px; background: #1a1a1a; margin: 5px 0;'>
                                <div style='height: 100%; width: {proba[0]*100}%; background: linear-gradient(90deg, #000, #00ff9d);'></div>
                            </div>
                            <p style='color:#00ff9d; font-size: 1.2em;'>{proba[0]*100:.1f}%</p>
                        </div>
                        <div style='width: 30%;'>
                            <h4 style='color:#ffcc00'>OFFENSIVE</h4>
                            <div style='height: 10px; background: #1a1a1a; margin: 5px 0;'>
                                <div style='height: 100%; width: {proba[1]*100}%; background: linear-gradient(90deg, #000, #ffcc00);'></div>
                            </div>
                            <p style='color:#ffcc00; font-size: 1.2em;'>{proba[1]*100:.1f}%</p>
                        </div>
                        <div style='width: 30%;'>
                            <h4 style='color:#ff00aa'>TOXIC</h4>
                            <div style='height: 10px; background: #1a1a1a; margin: 5px 0;'>
                                <div style='height: 100%; width: {proba[2]*100}%; background: linear-gradient(90deg, #000, #ff00aa);'></div>
                            </div>
                            <p style='color:#ff00aa; font-size: 1.2em;'>{proba[2]*100:.1f}%</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # About section
    st.markdown("""
    <div class='neon-box'>
        <h2 style='color:#00ff9d'>SYSTEM SPECS</h2>
        <p style='color:#cccccc'>MODEL: LogisticRegression</p>
        <p style='color:#cccccc'>TRAINING DATA: Jigsaw Toxic Comments</p>
        <p style='color:#ff00aa'>DETECTION CAPABILITIES:</p>
        <ul>
            <li style='color:#00ff9d'>Safe content (Green Zone)</li>
            <li style='color:#ffcc00'>Offensive language (Warning Zone)</li>
            <li style='color:#ff00aa'>Toxic content (Danger Zone)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Terminal footer
st.markdown(f"""
<div style='border-top: 1px solid #00ff9d; margin-top: 50px; padding: 10px;'>
    <p style='color:#00ff9d; font-family: monospace;'>SYSTEM STATUS: <span style='color:#00ff9d'>OPERATIONAL</span></p>
    <p style='color:#00ff9d; font-family: monospace;'>LAST SCAN: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
