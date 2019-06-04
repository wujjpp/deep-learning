# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import tensorflow as tf
from progress.bar import Bar
import numpy as np
#
# data = np.asarray([[1,2],[3,4]])
#
# print(data[:, 0])
# print(data[:, 1])
#
# exit()
# y = 4x + 5

# y = Wx + b
# b = y - Wx
# W = (y - b) / x

# 准备样本数据
x_train = np.random.random(100)
y_train = np.asarray([(4 * x + 5) for x in x_train], dtype='float32')

# 使用tensorflow求解
W = tf.Variable(0, dtype=tf.float64, name='W')
b = tf.Variable(0, dtype=tf.float64, name='b')
x = tf.placeholder(dtype=tf.float64, name='x')
y = tf.placeholder(dtype=tf.float64, name='y')

# 设计模型
model = x * W + b

learning_rate = 0.001
N = float(len(x_train))

# w_error = x * (y - W * x - b) ->
W_gradient = tf.reduce_sum(-2/N * x * (y - W * x - b))
# b_error = y - W * x - b -> 我们期望这个计算值为0，当W的值趋于真实结果的时候，这个等式计算结果趋于0
b_gradient = tf.reduce_sum(-2/N * (y - W * x - b))

new_W = W.assign(W - learning_rate * W_gradient)
new_b = b.assign(b - learning_rate * b_gradient)


# 初始化变量
init = tf.global_variables_initializer()

epochs = 10000
metrics = []

with tf.Session() as sess:
    sess.run(init)

    k = 0
    steps = 200
    with Bar('Training', max=epochs/steps) as bar:

        for i in range(epochs):
            new_W_r, new_b_r = sess.run([new_W, new_b, W_gradient, b_gradient], feed_dict={x: x_train, y: y_train})

            k += 1

            if k % steps == 0:
                bar.next()

    result = sess.run(model, feed_dict={x: 4})

    print(result)

#
# plt.figure()
# plt.plot([i + 1 for i in range(len(metrics))], metrics, label='mean squared error')
# plt.xlabel('epochs')
# plt.ylabel('mse')
# plt.legend()
# plt.show()

