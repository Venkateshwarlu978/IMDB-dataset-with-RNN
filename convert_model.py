import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "SimpleRNN", "simple_rnn_imdb.keras")

# Parameters (must match app.py)
MAX_FEATURES = 10000
MAX_LEN = 500

# Load IMDB data
(x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)

# Build Keras 3 compatible model
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(MAX_FEATURES, 128),
    SimpleRNN(128),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train briefly (for compatibility)
model.fit(x_train, y_train, epochs=1, batch_size=128)

# Save model (Keras 3 safe)
model.save(MODEL_PATH)

print("✅ Keras 3 compatible model saved successfully")