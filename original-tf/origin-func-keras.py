# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import tensorflow as tf
from progress.bar import Bar

# y = 4x + 5

# 准备样本数据
x_train = [1, 2]
y_train = [9, 13]

# 使用keras求解

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(1, input_dim=1))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.mean_squared_error])

model.summary()

model.fit(
    x_train,
    y_train,
    epochs=10000,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir='results')])

print(model.predict([3]))
