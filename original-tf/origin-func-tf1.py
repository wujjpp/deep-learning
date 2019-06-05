# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import tensorflow as tf
from progress.bar import Bar
import numpy as np

# 准备样本数据
x_train = np.random.random(100)
y_train = np.asarray([(4 * x + 5) for x in x_train], dtype='float32')

# plt.plot(x_train, y_train, '.b')
# plt.show()
# exit()

# 使用tensorflow求解
W = tf.Variable(0, dtype=tf.float64, name='W')
b = tf.Variable(0, dtype=tf.float64, name='b')
x = tf.placeholder(dtype=tf.float64, name='x')
y = tf.placeholder(dtype=tf.float64, name='y')

# 设计模型
model = x * W + b

# 计算损失函数 - mean squared error
loss = tf.sqrt(tf.reduce_sum(tf.square(model - y)))

learning_rate = 0.01
N = float(len(x_train))

# y - model决定方向

# Σ -2/N * (y - (x * W + b))  ->  下面的迭代可以将这个值逼向0
b_gradient = tf.reduce_sum(-2/N * (y - model))
# Σ -2/N * x * (y - (W * x + b)) -> 下面的迭代可以将这个值逼向0
W_gradient = tf.reduce_sum(-2/N * (y - model) * x)

# 梯度下降
new_b = b.assign(b - learning_rate * b_gradient)
new_W = W.assign(W - learning_rate * W_gradient)

# 初始化变量
init = tf.global_variables_initializer()

epochs = 2000
metrics = []

with tf.Session() as sess:
    sess.run(init)

    k = 0
    steps = 200
    with Bar('Training', max=epochs/steps) as bar:

        for i in range(epochs):
            new_W_r, new_b_r, loss_r = sess.run([new_W, new_b, loss], feed_dict={x: x_train, y: y_train})
            metrics.append(loss_r)
            k += 1

            if k % steps == 0:
                bar.next()

    result = sess.run(model, feed_dict={x: 5})

    print(result)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x_train, y_train, 'ro')
plt.subplot(1, 2, 2)
plt.plot([i + 1 for i in range(len(metrics))], metrics, label='mean squared error')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()

