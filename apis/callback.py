from keras import models, optimizers, losses, activations, layers
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence


class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        # 这边相当于，构建一个新的model, 一个输入，多个输出
        self.activation_model = keras.models.Model(model.input, layer_outputs)

    def on_train_begin(self, logs=None):
        return super().on_train_begin(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        return super().on_epoch_begin(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        return super().on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        return super().on_batch_end(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if (self.validation_data is None):
            raise RuntimeError('Require validation data')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activation_model.predict(validation_sample)
        for activation in activations:
            print(activation.shape)
        return super().on_epoch_end(epoch, logs=logs)

    def on_train_end(self, logs=None):
        return super().on_train_end(logs=logs)


max_feature = 10000
max_len = 20

(x_train, y_labels), (x_test, y_test) = imdb.load_data(num_words=max_feature)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()

model.add(layers.Embedding(10000, 8, input_length=max_len))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

model.summary()

model.fit(x_train,
          y_labels,
          batch_size=32,
          epochs=2,
          validation_split=0.2,
          callbacks=[ActivationLogger()])
