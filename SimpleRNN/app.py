# app.py
# Streamlit app for IMDB Sentiment Analysis (BiLSTM, H5 model)

import os
import traceback
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below to predict whether the sentiment is "
    "**Positive** or **Negative**."
)

# --------------------------------------------------
# Model path (H5 saved by main.py)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_bilstm_imdb_tf220.h5")

@st.cache_resource
def load_model():
    # compile=False avoids loading optimizer state and is safer for inference
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load the Keras model. See details below.")
    st.text(f"MODEL_PATH = {MODEL_PATH}")
    st.text("---- full traceback ----")
    st.text("".join(traceback.format_exc()))
    st.stop()

# --------------------------------------------------
# Parameters (must match main.py)
# --------------------------------------------------
MAX_FEATURES = 10000
MAX_LEN = 500

# Load IMDB word index
word_index = imdb.get_word_index()

def encode_review(text: str):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) for w in words if word_index.get(w, 2) < MAX_FEATURES]
    return pad_sequences([encoded], maxlen=MAX_LEN, padding="pre", truncating="pre")

# --------------------------------------------------
# UI
# --------------------------------------------------
review = st.text_area(
    "📝 Enter your movie review:",
    height=160,
    placeholder="This movie was amazing! The story and acting were excellent."
)

if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        with st.spinner("Predicting sentiment..."):
            encoded = encode_review(review)
            score = model.predict(encoded, verbose=0)[0][0]

        st.markdown("---")
        if score >= 0.5:
            st.success(f"✅ **Positive Review**\n\nConfidence: **{score:.2f}**")
        else:
            st.error(f"❌ **Negative Review**\n\nConfidence: **{1 - score:.2f}**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Keras, and Streamlit")
