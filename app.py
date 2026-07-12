from pathlib import Path

import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="EMHMD - Health Misinformation Detector", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #023047 0%, #028090 100%);
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}
.stTextArea textarea, .stTextInput input {
    color: #1E293B !important;
    background-color: #FFFFFF !important;
}
.main-header {
    background: rgba(255, 255, 255, 0.12);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: #FFFFFF !important; margin: 0; font-size: 3.5rem; }
div.stButton > button {
    background-color: #02C39A;
    color: #023047 !important;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
}
div.stButton > button:hover { background-color: #00A896; }
[data-testid="stMetricValue"] {
    color: #02C39A !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #FFFFFF !important;
    font-weight: 600;
}
.model-card {
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.08);
}
.gauge-wrap { display: flex; justify-content: center; margin: 0.5rem 0; }
.info-bar {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_biobert():
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    weight_path = Path(__file__).resolve().parent / "biobert_full_finetuned_final.pt"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    else:
        st.info("Fine-tuned weights were not found locally; using the base BioBERT model for demo deployment.")

    model.eval()
    return tokenizer, model

@st.cache_resource
def load_tfidf_baseline():
    train_df = pd.read_csv("Data/Constraint_Train.csv")
    label_map = {"real": 1, "fake": 0}
    train_df['label_num'] = train_df['label'].map(label_map)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['tweet'].astype(str))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df['label_num'])
    return vectorizer, clf

def gauge_html(confidence, color):
    return f"""
    <div class="gauge-wrap">
      <div style="width:150px;height:150px;border-radius:50%;
        background:conic-gradient({color} {confidence}%, #ffffff33 {confidence}% 100%);
        display:flex;align-items:center;justify-content:center;">
        <div style="width:115px;height:115px;background:#023047;border-radius:50%;
          display:flex;align-items:center;justify-content:center;
          font-size:22px;font-weight:700;color:{color};">
          {confidence:.1f}%
        </div>
      </div>
    </div>
    """

st.markdown('<div class="main-header"><h1>🩺 EMHMD</h1><p>Explainable Multimodal Health Misinformation Detector</p></div>', unsafe_allow_html=True)

st.markdown('<div class="info-bar">', unsafe_allow_html=True)
i1, i2, i3, i4 = st.columns(4)
i1.metric("Model", "BioBERT (fine-tuned)")
i2.metric("Test Accuracy", "95.14%")
i3.metric("Training Data", "6,420 tweets")
i4.metric("Dataset", "CONSTRAINT@AAAI2021")
st.markdown('</div>', unsafe_allow_html=True)
st.caption("Kamaljeet Kaur — MSE907 Capstone, Yoobee College of Creative Innovation")

with st.spinner("Loading BioBERT model..."):
    tokenizer, biobert_model = load_biobert()
with st.spinner("Training TF-IDF baseline..."):
    tfidf_vectorizer, tfidf_model = load_tfidf_baseline()

st.write("Enter a health-related claim below, or click an example:")

col1, col2 = st.columns(2)
example = None
with col1:
    if st.button("💧 Drinking hot water cures COVID-19"):
        example = "Drinking hot water cures COVID-19"
with col2:
    if st.button("💉 Vaccines prevent measles"):
        example = "Vaccines prevent measles"

user_input = st.text_area("Enter a health claim:", value=example if example else "", placeholder="e.g., Garlic cures cancer")

if st.button("🔍 Check Claim", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing with both models..."):
            encoding = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                output = biobert_model(**encoding)
                probs = torch.softmax(output.logits, dim=-1)
                bio_pred = torch.argmax(probs, dim=-1).item()
                bio_conf = probs[0][bio_pred].item() * 100

            tfidf_vec = tfidf_vectorizer.transform([user_input])
            tfidf_pred = tfidf_model.predict(tfidf_vec)[0]
            tfidf_conf = tfidf_model.predict_proba(tfidf_vec)[0][tfidf_pred] * 100

        bio_label = "REAL ✅" if bio_pred == 1 else "FAKE ⚠️"
        tfidf_label = "REAL ✅" if tfidf_pred == 1 else "FAKE ⚠️"
        bio_color = "#02C39A" if bio_pred == 1 else "#FF6B6B"
        tfidf_color = "#02C39A" if tfidf_pred == 1 else "#FF6B6B"

        st.markdown("### Model Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="model-card"><b>BioBERT (Fine-tuned)</b><br>{bio_label}</div>', unsafe_allow_html=True)
            st.markdown(gauge_html(bio_conf, bio_color), unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="model-card"><b>TF-IDF + Logistic Regression (Baseline)</b><br>{tfidf_label}</div>', unsafe_allow_html=True)
            st.markdown(gauge_html(tfidf_conf, tfidf_color), unsafe_allow_html=True)

        st.info("📊 Statistical significance of BioBERT's improvement over this baseline was confirmed using McNemar's test (p = 0.0045).")

st.markdown("---")
st.caption("EMHMD Research Prototype — Kamaljeet Kaur, Yoobee College of Creative Innovation. This tool is for research demonstration purposes and is not a substitute for professional medical advice.")