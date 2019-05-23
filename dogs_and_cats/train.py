from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras.preprocessing.image import ImageDataGenerator
from dirs import train_dir, validation_dir, test_dir
import matplotlib.pyplot as plt

model = models.Sequential()

model.add(
    layers.Conv2D(32, (3, 3),
                  activation=activations.relu,
                  input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

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
