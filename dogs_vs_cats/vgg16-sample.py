from keras.applications import VGG16
from config import train_dir, validation_dir, test_dir
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory=directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        print('directory:{0}, percent:{1:.2f}%'.format(
            directory, i * batch_size * 100.0 / sample_count))
        if i * batch_size >= sample_count:
            break

    return features, labels


train_sample_size = 2000
validation_sample_size = 1000
test_sample_size = 1000

train_features, train_labels = extract_features(train_dir, train_sample_size)
validation_features, validation_labels = extract_features(
    validation_dir, validation_sample_size)
# test_features, test_labels = extract_features(test_dir, test_sample_size)

train_features = np.reshape(train_features, (train_sample_size, 4 * 4 * 512))
validation_features = np.reshape(validation_features,
                                 (validation_sample_size, 4 * 4 * 512))
# test_features = np.reshape(test_features, (test_sample_size, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation=activations.relu,
                       input_dim=4 * 4 * 512))

model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

history_dict = history.history

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
