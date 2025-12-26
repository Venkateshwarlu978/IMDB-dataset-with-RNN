# main.py
# IMDB Sentiment Analysis using BiLSTM
# Compatible with TensorFlow 2.20 / Keras 3 / Python 3.13
import tensorflow as tf
print("TF version in main.py:", tf.__version__)  # add this line
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Hyperparameters
# --------------------------------------------------
MAX_FEATURES = 10000      # Vocabulary size
MAX_LEN = 500             # Max review length
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

# --------------------------------------------------
# 2. Load IMDB Dataset
# --------------------------------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=MAX_FEATURES
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --------------------------------------------------
# 3. Pad Sequences
# --------------------------------------------------
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN,
    padding="pre",
    truncating="pre"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LEN,
    padding="pre",
    truncating="pre"
)

print("Padded train shape:", X_train.shape)
print("Padded test shape:", X_test.shape)

# --------------------------------------------------
# 4. Build BiLSTM Model
# --------------------------------------------------
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=EMBEDDING_DIM
    ),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# --------------------------------------------------
# 5. Compile Model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 6. Callbacks
# --------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# --------------------------------------------------
# 7. Train Model
# --------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# --------------------------------------------------
# 8. Evaluate Model
# --------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# --------------------------------------------------
# 9. Save Model (Keras 3 SAFE)
# --------------------------------------------------
model.save("sentiment_bilstm_imdb.keras")
print("Model saved as sentiment_bilstm_imdb.keras")
