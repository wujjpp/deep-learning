import tensorflow as tf

mnist = tf.keras.datasets.mnist


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)

        bias_shape = tf.TensorShape((self.output_dim))
        self.bias = self.add_weight(name='bias',
                                    shape=bias_shape,
                                    initializer='uniform',
                                    trainable=True)

        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        # return tf.matmul(inputs, self.kernel)
        return tf.matmul(inputs, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(name='flatten_1'))
    model.add(MyLayer(10, name='custom_layer_1'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def main():
    model = create_model()
    model.fit(x_train, y_train, epochs=1, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir='results')
    ])
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
