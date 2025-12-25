import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "SimpleRNN", "simple_rnn_imdb")

# Params
MAX_FEATURES = 10000
MAX_LEN = 500

# Load data
(x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)

# -------- KERAS 3 SAFE MODEL (FUNCTIONAL API) --------
inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
x = layers.Embedding(
    input_dim=MAX_FEATURES,
    output_dim=128
)(inputs)
x = layers.SimpleRNN(128)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train briefly
model.fit(x_train, y_train, epochs=1, batch_size=128)

# 🔥 KERAS 3 NATIVE EXPORT (NO LEGACY CONFIG)
model.export(MODEL_DIR)

print("✅ Model exported in pure Keras 3 format at:", MODEL_DIR)
