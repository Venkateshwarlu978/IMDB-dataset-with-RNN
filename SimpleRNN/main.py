# main.py
# IMDB Sentiment Analysis using SimpleRNN

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Hyperparameters
# -----------------------------
max_features = 10000   # Vocabulary size
max_len = 500          # Max review length
embedding_dim = 128
batch_size = 64
epochs = 7

# -----------------------------
# 2. Load IMDB Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# -----------------------------
# 3. Pad Sequences
# -----------------------------
X_train = pad_sequences(
    X_train,
    maxlen=max_len,
    padding='pre',
    truncating='pre'
)

X_test = pad_sequences(
    X_test,
    maxlen=max_len,
    padding='pre',
    truncating='pre'
)

print("After padding:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# -----------------------------
# 4. Build SimpleRNN Model
# -----------------------------
model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(128),  # tanh activation (stable)
    Dense(1, activation='sigmoid')
])

# -----------------------------
# 5. Compile Model
# -----------------------------
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 6. Callbacks
# -----------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# -----------------------------
# 7. Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# -----------------------------
# 9. Save Model
# -----------------------------
model.save("imdb_simple_rnn_model.h5")
print("Model saved as imdb_simple_rnn_model.h5")
