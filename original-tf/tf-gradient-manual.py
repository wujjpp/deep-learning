import tensorflow as tf
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

x_train = x_train / 255
x_test = x_test / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# 计算total_batch
total_batch = int(x_train.shape[0] // batch_size)
# 样本数
total_samples = x_train.shape[0]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W)+b)

# Minimize error using cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 计算梯度
W_grad = -tf.matmul(tf.transpose(x), y - pred)
b_grad = -tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

# 梯度下降
new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.

        # train
        for i in range(total_batch):
            index_start = min(i * batch_size, total_samples)
            index_end = min((i + 1) * batch_size, total_samples)

            batch_x = x_train[index_start:index_end]
            batch_y = y_train[index_start:index_end]

            _, _ , c = sess.run([new_W, new_b, loss], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "lost=", "{:.9f}".format(avg_cost))


    # test
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

    print('test acc', acc.eval({x: x_test, y: y_test}))

    print('Optimization Finished')