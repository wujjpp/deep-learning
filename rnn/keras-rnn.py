from keras import models, layers, optimizers, losses, activations
import numpy as np

samples = 15000
max_len = 25
max_words = 100

x_train = np.random.random((samples, max_len))
y_train = np.random.randint(0, 2, (samples, ))

model = models.Sequential()
model.add(layers.Embedding(max_words, 128, input_length=max_len))
model.add(layers.SimpleRNN(16, return_sequences=True))
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(64, return_sequences=True))
# model.add(layers.Flatten())
model.add(layers.SimpleRNN(8))
model.add(layers.Dense(1, activation=activations.sigmoid))
model.summary()

model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.5)
