from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models, layers, losses, optimizers
from keras import activations
import callback_helper

# about pad_sequences
# 不足max_len的情况下，默认左边补0（可配置）
# list1 = [[1, 2, 3, 4, 5], [6, 7, 8]]
# list1 = sequence.pad_sequences(list1, maxlen=10)
# print(list1)
# '''
# [
#     [0 0 0 0 0 1 2 3 4 5]
#     [0 0 0 0 0 0 0 6 7 8]
# ]
# '''

max_feature = 10000
max_len = 20

(x_train, y_labels), (x_test, y_test) = imdb.load_data(num_words=max_feature)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()

model.add(layers.Embedding(10000, 8, input_length=max_len))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.summary()

callbacks = callback_helper.generate_callbacks_include_early_stop()

model.fit(x_train,
          y_labels,
          batch_size=32,
          epochs=10,
          validation_split=0.2,
          callbacks=callbacks)

result = model.evaluate(x_test, y_labels)
print(result)