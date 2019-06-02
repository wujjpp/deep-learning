import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def func(x):
    return 3 * x + (2 + (0.125 - np.random.random() / 8.))


def prepare_train_data():
    x_train = np.random.random_sample(size=1000)
    y_train = [func(x) for x in x_train]
    return x_train, y_train


x_train, y_train = prepare_train_data()

# dataset = pd.DataFrame({'x': x_train, 'y': y_train})
# print(dataset.head(n=10))
# print(dataset.tail(n=10))


num_epochs = 250

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1))
model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mae'])

history = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=num_epochs)

# 生成评估数据
seeds = np.random.random_sample(size=100)
x_results = model.predict(seeds)


# 下面是画图
# 画样本图
plt.subplot(2, 1, 1)  # 2行1列布局，第1个图（上面一个图）
plt.plot(x_train[:200], y_train[:200], 'bo', label='train')
# 画测试结果图
plt.plot(seeds, x_results, 'ro', label='predict')
plt.legend()

# mae
history_dic = history.history
loss_history = history_dic['loss']
mean_absolute_error_history = history_dic['mean_absolute_error']

plt.subplot(2, 1, 2)  # 2行1列布局，第2个图（下面一个图）
plt.plot(range(num_epochs), loss_history, 'r', label='loss')
plt.plot(range(num_epochs), mean_absolute_error_history, 'b', label='mae')
plt.xlabel('epochs')

plt.legend()

plt.show()
