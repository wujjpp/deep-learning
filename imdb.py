from keras.datasets import imdb
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
import numpy as np

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def show_content(content_data):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in content_data])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(train_labels)
print(y_train)

model = models.Sequential()
model.add(layers.Dense(16, activation=activations.relu, input_shape=(10000, )))
model.add(layers.Dense(16, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
print(results)

results2 = model.predict(x_test[:10])
print(results2)
print(test_labels[:10])
