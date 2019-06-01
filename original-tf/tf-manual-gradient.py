import tensorflow as tf
import math
from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

lost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

p1 = -tf.reduce_sum(y * tf.log(pred), reduction_indices=1)
p2 = y * tf.log(pred)
p3 = tf.log(pred)

W_grad = -tf.matmul(tf.transpose(x), y - pred)
b_grad = -tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch = x_train.shape[0] // batch_size

    for epoch in range(training_epochs):

        avg_cost = 0.

        # train
        for i in range(total_batch):
            index_start = i * batch_size
            index_end = (i + 1) * batch_size

            batch_x = x_train[index_start:index_end]
            batch_y = y_train[index_start:index_end]

            _, _ , c, p1r, p2r, p3r = sess.run([new_W, new_b, lost, p1, p2, p3], {x: batch_x, y: batch_y})


            print(c)
            print(p3r)

            if i == 1:
                exit()

            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print('Epochs: {0:4d}, cost = {1:.9f}'.format(epoch + 1, avg_cost))


    # test
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

    print('test acc', acc.eval({x: x_test, y: y_test}))

    print('Optimization Finished')