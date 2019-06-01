import tensorflow as tf
from progress.bar import Bar
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 3.1 * x + (2 + (0.125 - np.random.random() / 8.))


def prepare_train_data():
    x_train = np.random.random_sample(size=1000)
    y_train = [func(x) for x in x_train]
    return x_train, y_train


x_train, y_train = prepare_train_data()

plt.plot(x_train[:200], y_train[:200], 'bo')

# 创建变量W, b
W = tf.Variable(.1, dtype=tf.float32)
b = tf.Variable(-.1, dtype=tf.float32)

# 创建x节点，用来输入x_train[n]
x = tf.placeholder(tf.float32)

# 这个就是模型函数
linear_model = x * W + b

# 创建y节点，用来输入y_train[n]
y = tf.placeholder(tf.float32)

# 创建损失函数，用于评估模型输出值与期望值差距
loss = tf.reduce_sum(tf.square(linear_model - y))

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
# optimizer = tf.train.AdamOptimizer()

train = optimizer.minimize(loss)


# 初始化变量
init = tf.global_variables_initializer()

# 训练10000次
epochs = 20000
with tf.Session() as sess:
    sess.run(init)
    with Bar('Processing', max=epochs) as bar:
        for i in range(epochs):
            sess.run(train, {x: x_train, y: y_train})
            bar.next()

    r_W, r_b, r_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print('W:{0}, b: {1}, loss: {2}'.format(r_W, r_b, r_loss))

    seeds = np.random.random_sample(size=100)

    x_test = [i * W + b for i in seeds]
    y_test = [func(i) for i in seeds]

    x_results = sess.run(x_test)

    plt.plot(seeds, x_results, 'ro')

plt.show()
