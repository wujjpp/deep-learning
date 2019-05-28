from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models, layers, losses, optimizers
from keras import activations
import callback_helper

max_feature = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()

model.add(layers.Embedding(max_feature, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))

model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.summary()

callbacks = callback_helper.generate_callbacks()

model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=10,
          validation_split=0.2,
          callbacks=callbacks)

result = model.evaluate(x_test, y_test)
print(result)
