import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "SimpleRNN", "simple_rnn_imdb.keras")

MAX_FEATURES = 10000
MAX_LEN = 500

(x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)

inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
x = layers.Embedding(MAX_FEATURES, 128)(inputs)
x = layers.SimpleRNN(128)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=1, batch_size=128)

model.save(MODEL_PATH)
print("DONE")
