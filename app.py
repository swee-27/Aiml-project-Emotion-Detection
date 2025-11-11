import streamlit as st
import pickle
import numpy as np

# Load your model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Emotion Recognition App", page_icon="😊", layout="centered")

# --- Load CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <h1>😊 Emotion Recognition from Text</h1>
    <p>Type a sentence and I’ll detect the <b>emotion</b> behind it!</p>
""", unsafe_allow_html=True)

# --- Text Input ---
user_input = st.text_area("Enter your text 👇", height=150)

# --- Button ---
if st.button("🔍 Detect Emotion"):
    if user_input.strip() != "":
        # Preprocess input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        prediction_proba = np.max(model.predict_proba(input_vector)) * 100

        # --- Emotion Cards ---
        emotion_styles = {
            "joy": ("😄", "#00C853", "white"),
            "anger": ("😡", "#D50000", "white"),
            "sadness": ("😢", "#1565C0", "white"),
            "fear": ("😨", "#6A1B9A", "white"),
            "neutral": ("😐", "#9E9E9E", "white"),
            "surprise": ("😲", "#FF6D00", "white")
        }

        emoji, bg_color, text_color = emotion_styles.get(prediction.lower(), ("🙂", "#607D8B", "white"))

        st.markdown(f"""
            <div class="result-box" style="background-color:{bg_color};color:{text_color}">
                <h2 style="font-size:28px;">{emoji} Predicted Emotion: <b>{prediction.capitalize()}</b></h2>
                <p class="confidence">Confidence: {prediction_proba:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter some text before clicking Detect Emotion.")
