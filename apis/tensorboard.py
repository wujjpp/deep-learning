import keras
from keras import models, layers, optimizers, losses, activations
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, name='embed'))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1, activation=activations.sigmoid))

model.summary()

model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train,
          y_train,
          epochs=20,
          batch_size=128,
          validation_split=0.2,
          callbacks=[
              keras.callbacks.TensorBoard(log_dir='results',
                                          histogram_freq=1,
                                          embeddings_freq=1)
          ])
