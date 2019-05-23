from keras import models
from keras import layers
from keras import activations
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt


def build_model_1():
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3),
                      activation=activations.relu,
                      input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(10, activation=activations.softmax))

    return model


def build_model_2():
    model = models.Sequential()
    model.add(
        layers.Conv2D(16, (5, 5),
                      padding='same',
                      input_shape=(28, 28, 1),
                      activation=activations.relu))
    model.add(layers.MaxPool2D((2, 1)))
    model.add(
        layers.Conv2D(36, (5, 5), padding='same', activation=activations.relu))
    model.add(layers.MaxPool2D((2, 1)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation=activations.softmax))

    return model


model = build_model_1()

model.summary()

exit()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((len(train_images), 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((len(test_images), 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

val_samples = 30000

data_train_images = train_images[val_samples:]
data_train_labels = train_labels[val_samples:]

val_images = train_images[:val_samples]
val_labels = train_labels[:val_samples]

model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(data_train_images,
                    data_train_labels,
                    epochs=5,
                    batch_size=64,
                    validation_data=(val_images, val_labels))

results = model.evaluate(test_images, test_labels)
print(results)

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
