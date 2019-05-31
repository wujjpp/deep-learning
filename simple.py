from keras import layers, models, optimizers, callbacks
import numpy as np

x_train = np.random.random(size=100)
y_train = np.random.random(size=100)

# print(x_train, y_train)

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(1, ), name='layer-1'))

model.compile(loss='mean_squared_error',
              optimizer=optimizers.RMSprop(0.001),
              metrics=['mae'])

model.summary()

model.fit(x_train,
          y_train,
          callbacks=[callbacks.TensorBoard(log_dir='results')])
