from keras import models, optimizers, losses, layers, activations, callbacks, Input
import numpy as np
from keras.layers import InputLayer


class ActivationLogger(callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [
            layer.output for layer in model.layers if type(layer) != InputLayer
        ]
        self.layer_names = [
            layer.name for layer in model.layers if type(layer) != InputLayer
        ]
        # 这边相当于，构建一个新的model, 一个输入，多个输出
        self.activation_model = models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if (self.validation_data is None):
            raise RuntimeError('Require validation data')
        validation_sample = self.validation_data[0][0:1]

        activations = self.activation_model.predict(validation_sample)
        for i, activation in enumerate(activations):
            print(
                '\n\n--------------------{0} outputs(shape: {1})--------------------'
                .format(self.layer_names[i], activation.shape))
            print(activation)


sample_num = 1000

x_train = np.random.random(size=(sample_num, 28, 28, 3))
y_train = np.random.randint(2, size=sample_num)

# 输入
inputs = Input(shape=(28, 28, 3), name='input_layer')

x = layers.Flatten(name="flatten-1")(inputs)

# 28*28*3全连接层
x = layers.Dense(28 * 28 * 3,
                 activation=activations.relu,
                 name='dense_28_28_3_1')(x)

# 转换成Conv2D需要的格式
x = layers.Reshape((28, 28, 3), name='reshape-1')(x)
x = layers.Conv2D(filters=8,
                  kernel_size=(2, 2),
                  padding='same',
                  activation=activations.relu,
                  name='conv2d-1')(x)

# 池化层
x = layers.MaxPooling2D(pool_size=(2, 2), strides=6, name="maxpooling-1")(x)

# # 转换成Embedding层需要的格式
# x = layers.Flatten(name='flatten-2')(x)

# x = layers.Dense(100, name='for_fix_issue', activation=activations.relu)(x)

# x = layers.Embedding(1024, 64, input_length=100)(x)

x = layers.Flatten(name='flatten-last')(x)

x = layers.Dense(1, activation=activations.sigmoid, name='dense_1')(x)

model = models.Model(inputs, x)
model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.summary()

model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=1,
          validation_split=0.1,
          callbacks=[ActivationLogger()])
