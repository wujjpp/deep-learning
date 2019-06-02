import tensorflow as tf
from progress.bar import Bar
import numpy as np
import matplotlib.pyplot as plt


num_samples = 1000

def func(x):
    return 3.1 * x + (2 + (0.125 - np.random.random() / 8.))
    # return 3.1 * x + 2


def prepare_train_data():
    x_train = np.random.random_sample(size=num_samples)
    y_train = np.array([func(x) for x in x_train])
    return x_train, y_train


x_train, y_train = prepare_train_data()


# 存数学公式推导： 最小二乘解
def fit():
    mean_of_x = x_train.mean()
    mean_of_y = y_train.mean()

    mean_of_xy = (x_train.dot(y_train)) / num_samples
    mean_of_xx = (x_train.dot(x_train)) / num_samples

    W = (mean_of_xy - (mean_of_x * mean_of_y)) / (mean_of_xx - mean_of_x * mean_of_x)
    b = mean_of_y - W * mean_of_x
    return W, b

print('最小二乘解： ', fit())


plt.plot(x_train[:200], y_train[:200], 'bo')

# 创建变量W, b
W = tf.Variable(.1, dtype=tf.float32, name='W')
b = tf.Variable(-.1, dtype=tf.float32, name='b')

# 创建x节点，用来输入x_train[n]
x = tf.placeholder(tf.float32)

# 这个就是模型函数
model = x * W + b

# 创建y节点，用来输入y_train[n]
y = tf.placeholder(tf.float32)

# 创建损失函数，用于评估模型输出值与期望值差距
# tf.math.square(model - y) ---> square loss function, 平均损失函数
# tf.math.reduce_sum(tf.math.square(model - y)) ---> 整个数据集上的平均损失函数
loss = tf.math.reduce_sum(tf.math.square(model - y))

learning_rate = 0.01

# 计算梯度
W_grad = -tf.reduce_mean(x * (y - model))
b_grad = -tf.reduce_mean(x * (y - model))

# 梯度下降
new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)


# # 创建一个梯度下降优化器，学习率为0.001,通过 optimizer + loss 获取到每个变量的，然后应用到变量
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
gradients = optimizer.compute_gradients(loss)
train = optimizer.apply_gradients(gradients)

# 初始化变量
init = tf.global_variables_initializer()

# 训练10000次
epochs = 200
batch_size = 250
steps = num_samples // batch_size

with tf.Session() as sess:

    sess.run(init)

    for i in range(epochs):
        print('epochs:{0}/{1}'.format(i + 1, epochs))
        for j in range(steps):
            start = min(j * batch_size, num_samples)
            end = min((j + 1) * batch_size, num_samples)
            x_batch = x_train[start: end]
            y_batch = y_train[start: end]

            # (value1, value2, l) = sess.run([new_W, new_b,  loss], feed_dict={x: x_batch, y: y_batch})
            sess.run(train, feed_dict={x: x_batch, y: y_batch})

        r_W, r_b, r_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print('W:{0}, b: {1}, loss: {2}'.format(r_W, r_b, r_loss))

    seeds = np.random.random_sample(size=100)
    x_test = [i * W + b for i in seeds]
    y_test = [func(i) for i in seeds]

    x_results = sess.run(x_test)
    plt.plot(seeds, x_results, 'ro')

plt.show()
