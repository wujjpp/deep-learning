from keras.datasets import imdb
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
import numpy as np
import matplotlib.pyplot as plt

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

x_validation = x_train[:10000]
partial_x_train = x_train[10000:]

y_validation = y_train[:10000]
partial_y_train = y_train[10000:]

# 将训练数据分成2份，一份用于训练，一份用于检验, 找出合适的epochs值，之后再用完整的训练数据做训练
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_validation, y_validation))

history_dict = history.history
print(history_dict.keys())

train_loss_values = history_dict['loss']
validation_loss_values = history_dict['val_loss']

train_acc_values = history_dict['acc']
validation_acc_value = history_dict['val_acc']

epochs = range(1, len(train_loss_values) + 1)

plt.figure()
plt.plot(epochs, train_loss_values, 'bo', label="Training loss")
plt.plot(epochs, validation_loss_values, 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, train_acc_values, 'ro', label="Training acc")
plt.plot(epochs, validation_acc_value, 'r', label="Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print(results)

results2 = model.predict(x_test[:10])
print(results2)
print(test_labels[:10])
