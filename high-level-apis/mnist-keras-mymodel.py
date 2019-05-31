import tensorflow as tf
mnist = tf.keras.datasets.mnist


class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), num_classes=10):
        super(MyModel, self).__init__(name='my_model')

        self.num_classes = num_classes

        self.flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self.dense_1 = tf.keras.layers.Dense(
            512, activation=tf.keras.activations.relu)
        self.dense_2 = tf.keras.layers.Dropout(0.2)
        self.dense_3 = tf.keras.layers.Dense(
            num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs):

        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        # self.layers -> tf.keras.Model -> tf.pyhton.keras.engine.Network
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    model = MyModel()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def main():
    model = create_model()
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
