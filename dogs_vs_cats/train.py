from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from config import train_dir, validation_dir
from config import train_sample_size, validation_sample_size, test_sample_size
import os
import re


def build_model():

    model = models.Sequential()

    model.add(
        layers.Conv2D(filters=32,
                      kernel_size=(3, 3),
                      activation=activations.relu,
                      input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation=activations.relu))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      activation=activations.relu))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      activation=activations.relu))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),  # optimizers.rmsprop(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=['accuracy'])
    return model


batch_size = 20
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)

train_datagen = ImageDataGenerator(rescale=1. / 255)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

for train_batch, labels_batch in validation_generator:
    print(train_batch.shape)
    print(labels_batch.shape)

    # plt.figure(figsize=(10, 10))
    # index = 0
    # for img in train_batch:
    #     index += 1
    #     plt.subplot(4, 5, index)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(img)
    #     plt.xlabel(labels_batch[index - 1])

    # plt.show()

    # print(train_batch[0].shape)
    # print(train_batch[0])
    break


def get_result_file_names():
    # get max index
    results_path = os.path.join(os.path.abspath('.'), 'results')
    index = 0
    for x in os.listdir(results_path):
        matched = re.match(r'^run-(\d{4})$', x)
        if matched:
            cur = int(matched.group(1))
            if cur > index:
                index = cur
    index += 1

    # prepare dirs
    log_dir = 'results/run-{0:0>4d}/logs'.format(index)
    model_file_name = 'results/run-{0:0>4d}/model.h5'.format(index)
    return log_dir, model_file_name


log_dir, model_file_name = get_result_file_names()

model = build_model()

callbacks = [
    # log
    TensorBoard(log_dir=log_dir),
    # save model if necessary
    ModelCheckpoint(filepath=model_file_name,
                    monitor='val_loss',
                    save_best_only=True),
    # early stop if acc is not improvement
    EarlyStopping(monitor='acc', patience=5),
    # early stop if val_loss is not improvement
    EarlyStopping(monitor='val_loss', patience=5)
]

model.fit_generator(train_generator,
                    steps_per_epoch=((train_sample_size * 2) // batch_size),
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=((validation_sample_size * 2) //
                                      batch_size),
                    callbacks=callbacks)
