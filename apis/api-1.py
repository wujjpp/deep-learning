from keras import models, layers, activations, losses, optimizers, Input
import numpy as np

seq_model = models.Sequential()
seq_model.add(layers.Dense(32, activation=activations.relu,
                           input_shape=(64, )))
seq_model.add(layers.Dense(32, activation=activations.relu))
seq_model.add(layers.Dense(10, activation=activations.softmax))
seq_model.summary()

input_tensor = Input(shape=(64, ))
x = layers.Dense(32, activation=activations.relu)(input_tensor)
x = layers.Dense(32, activation=activations.relu)(x)
output_tensor = layers.Dense(10, activation=activations.softmax)(x)

model = models.Model(input_tensor, output_tensor)

model.summary()

model.compile(optimizer=optimizers.Adam(),
              loss=losses.categorical_crossentropy)

x_train = np.random.random((10000, 64))
y_train = np.random.random((10000, 10))

model.fit(x_train, y_train, epochs=5)
