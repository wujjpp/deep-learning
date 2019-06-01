import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def main():
    model = create_model()
    model.fit(x_train, y_train, epochs=2, batch_size=512)
    model.evaluate(x_test, y_test)

    # save weights and convert to json
    model.save_weights('my_model.h5', save_format='h5')
    json_string = model.to_json()

    # save all
    model.save('my_model_all.h5')

    fresh_model = tf.keras.models.model_from_json(json_string)
    fresh_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    fresh_model.load_weights('my_model.h5')
    fresh_model.evaluate(x_test, y_test)

    fresh_model_all = tf.keras.models.load_model('my_model_all.h5')
    fresh_model_all.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()
