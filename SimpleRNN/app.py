# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("imdb_simple_rnn_model.h5")

# -----------------------------
# Parameters (must match training)
# -----------------------------
max_features = 10000
max_len = 500

# Load word index
word_index = imdb.get_word_index()

# -----------------------------
# Helper function
# -----------------------------
def encode_review(text):
    words = text.lower().split()
    encoded = []
    for word in words:
        if word in word_index and word_index[word] < max_features:
            encoded.append(word_index[word])
    padded = pad_sequences([encoded], maxlen=max_len, padding='pre')
    return padded

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and predict whether it is **Positive** or **Negative**.")

review = st.text_area("Movie Review", height=150)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        encoded_review = encode_review(review)
        prediction = model.predict(encoded_review)[0][0]

        if prediction > 0.5:
            st.success(f"✅ Positive Review (Confidence: {prediction:.2f})")
        else:
            st.error(f"❌ Negative Review (Confidence: {1 - prediction:.2f})")
