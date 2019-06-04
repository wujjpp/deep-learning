# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import tensorflow as tf
from progress.bar import Bar

# y = 4x + 5

# 准备样本数据
x_train = [1, 2]
y_train = [9, 13]

# 使用tensorflow求解
W = tf.Variable(0, dtype=tf.float64, name='W')
b = tf.Variable(0, dtype=tf.float64, name='b')
x = tf.placeholder(dtype=tf.float64, name='x')
y = tf.placeholder(dtype=tf.float64, name='y')

# 设计模型
model = x * W + b

# 计算损失函数
loss = tf.sqrt(tf.reduce_sum(tf.square(model - y)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

epochs = 20000
metrics = []

with tf.Session() as sess:
    sess.run(init)

    k = 0
    steps = 200
    with Bar('Training', max=epochs/steps) as bar:

        for i in range(epochs):
            _, mae = sess.run([train, loss], feed_dict={x: x_train, y: y_train})

            k += 1

            if k % steps == 0:
                metrics.append(mae)
                bar.next()

    result = sess.run(model, feed_dict={x: 4})

    print('mae:{0}, x:{1}, y:{2}'.format(mae, 4, result))


plt.figure()
plt.plot([i + 1 for i in range(len(metrics))], metrics, label='mean squared error')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()

