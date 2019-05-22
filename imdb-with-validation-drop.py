from keras.datasets import imdb
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import regularizers
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

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_validation = x_train[:10000]
partial_x_train = x_train[10000:]

y_validation = y_train[:10000]
partial_y_train = y_train[10000:]

epochs = 20


# 正常模型
def fit_original():
    model = models.Sequential()
    model.add(
        layers.Dense(16, activation=activations.relu, input_shape=(10000, )))
    model.add(layers.Dense(16, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    # 将训练数据分成2份，一份用于训练，一份用于检验, 找出合适的epochs值，之后再用完整的训练数据做训练
    return model.fit(partial_x_train,
                     partial_y_train,
                     epochs=epochs,
                     batch_size=512,
                     validation_data=(x_validation, y_validation))


def fit_with_regularizer():
    model = models.Sequential()
    model.add(
        layers.Dense(16,
                     kernel_regularizer=regularizers.l2(0.001),
                     activation=activations.relu,
                     input_shape=(10000, )))
    model.add(
        layers.Dense(16,
                     kernel_regularizer=regularizers.l2(0.001),
                     activation=activations.relu))

    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    # 将训练数据分成2份，一份用于训练，一份用于检验, 找出合适的epochs值，之后再用完整的训练数据做训练
    return model.fit(partial_x_train,
                     partial_y_train,
                     epochs=epochs,
                     batch_size=512,
                     validation_data=(x_validation, y_validation))


# 添加Dropout层
def fit_with_drop():
    model = models.Sequential()
    model.add(
        layers.Dense(16, activation=activations.relu, input_shape=(10000, )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation=activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    # 将训练数据分成2份，一份用于训练，一份用于检验, 找出合适的epochs值，之后再用完整的训练数据做训练
    return model.fit(partial_x_train,
                     partial_y_train,
                     epochs=epochs,
                     batch_size=512,
                     validation_data=(x_validation, y_validation))


history_original_dict = fit_original().history
history_with_regularizer_dict = fit_with_regularizer().history
history_with_drop_dict = fit_with_drop().history

validation_loss_original = history_original_dict['val_loss']
validation_loss_with_regularizer = history_with_regularizer_dict['val_loss']
validation_loss_with_drop = history_with_drop_dict['val_loss']

xaxis = range(1, epochs + 1)

# plt.plot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
plt.figure()
plt.plot(xaxis, validation_loss_original, 'g:', label="Orignal model")
plt.plot(xaxis,
         validation_loss_with_regularizer,
         'r:',
         label="L2-regularized model")
plt.plot(xaxis,
         validation_loss_with_drop,
         'b:',
         label="Dropout-regularized model")
plt.xlabel('epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
