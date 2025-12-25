# app.py
# Streamlit app for IMDB Sentiment Analysis using SimpleRNN

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below and predict whether the sentiment is "
    "**Positive** or **Negative**."
)

# --------------------------------------------------
# Load model safely (relative path)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "simple_rnn_imdb")



@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Parameters (must match training)
# --------------------------------------------------
MAX_FEATURES = 10000
MAX_LEN = 500

# Load IMDB word index
word_index = imdb.get_word_index()

# --------------------------------------------------
# Helper function to encode review
# --------------------------------------------------
def encode_review(text: str):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word)
        if index is not None and index < MAX_FEATURES:
            encoded.append(index)

    padded = pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre"
    )

    return padded

# --------------------------------------------------
# Streamlit UI
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
        encoded_review = encode_review(review)
        prediction = model.predict(encoded_review, verbose=0)[0][0]

        st.markdown("---")

        if prediction >= 0.5:
            st.success(
                f"✅ **Positive Review**\n\nConfidence: **{prediction:.2f}**"
            )
        else:
            st.error(
                f"❌ **Negative Review**\n\nConfidence: **{1 - prediction:.2f}**"
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Keras, and Streamlit")
