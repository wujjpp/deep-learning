import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode):
    # 构建线性模型
    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square(y - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    global_step = tf.train.get_global_step()

    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=y,
                                      loss=loss,
                                      train_op=train)


estimator = tf.estimator.Estimator(model_fn=model_fn)

# 训练数据
x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21., 25.3])

# 评估数据
x_val = np.array([2., 5., 7., 9.])
y_val = np.array([7.6, 17.2, 23.6, 28.8])

train_input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train},
                                                    y_train,
                                                    batch_size=2,
                                                    num_epochs=None,
                                                    shuffle=True)

train_input_fn2 = tf.estimator.inputs.numpy_input_fn({'x': x_train},
                                                     y_train,
                                                     batch_size=2,
                                                     num_epochs=1000,
                                                     shuffle=False)

val_input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_val},
                                                  x_val,
                                                  batch_size=2,
                                                  num_epochs=1000,
                                                  shuffle=False)

# 训练模型
estimator.train(input_fn=train_input_fn, steps=1000)

# # 使用训练数据评估模型，目的查看训练结果, 这边使用的训练数据，Keras从训练数据中分离一部分数据作为验证数据
# train_metrics = estimator.evaluate(input_fn=train_input_fn2)
# print('train_metrics:', train_metrics)

# # 使用评估数据评估模型，目的是验证模型泛化性能，对应Keras中的测试数据
# val_metrics = estimator.evaluate(input_fn=val_input_fn)
# print('val_metrics:', val_metrics)
