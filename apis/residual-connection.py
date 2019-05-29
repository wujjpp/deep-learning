from keras import layers, models, optimizers, losses, activations, Input

inputs = Input(shape=(36, 36, 3))

# 恒等残差连接(identity residual connection), 直接用layers.add([tensors array])
x = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(inputs)

y = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(x)
y = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(y)

y = layers.add([y, x])

model = models.Model(inputs, y)
model.summary()

# 线性残差连接(linear residual connection)
x1 = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(inputs)

y1 = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(x1)
y1 = layers.MaxPooling2D(2, strides=2)(y)

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)

y1 = layers.add([y1, residual])

model1 = models.Model(inputs, y1)
model1.summary()
