import tensorflow as tf
model = tf.keras.models.load_model('SimpleRNN/simple_rnn_imdb.keras', compile=False)
model.save_weights('SimpleRNN/simple_rnn_imdb_weights.h5')

