# SimpleRNN/app.py
# Streamlit app for IMDB Sentiment Analysis (RNN/BiLSTM)
# Works with TensorFlow 2.x / Keras 3

import os
import traceback
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
    "Enter a movie review below to predict whether the sentiment is "
    "**Positive** or **Negative**."
)

# --------------------------------------------------
# Model path (CHANGE THIS IF NEEDED)
# --------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_bilstm_imdb.keras")



# --------------------------------------------------
# Load trained model with error visibility
# --------------------------------------------------
@st.cache_resource
def load_model():
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
# Parameters (MUST match training)
# --------------------------------------------------
MAX_FEATURES = 10000   # vocabulary size used in training
MAX_LEN = 500          # sequence length used in training

# Load IMDB word index
word_index = imdb.get_word_index()

# --------------------------------------------------
# Encode review (same logic as training)
# --------------------------------------------------
def encode_review(text: str):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word, 2)  # 2 = <UNK>
        if index < MAX_FEATURES:
            encoded.append(index)

    return pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre"
    )

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
        with st.spinner("Predicting sentiment..."):
            encoded_review = encode_review(review)
            # model output shape assumed (batch, 1) with sigmoid
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
