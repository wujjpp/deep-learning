from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

print(train_data.shape)  # (404, 13)
print(train_targets.shape)  # (404,)

print(test_data.shape)  # (102, 13)
print(test_targets.shape)  # (102,)

# d = np.array([[1, 2, 3], [4, 5, 6]])
# m = d.mean(axis=0)  # 压缩行，按列求平均值，得到 [2.5, 3.5, 4.5]
# print(m)
# m = d.mean(axis=1)  # 压缩列，按行球平均值，得到 [2.0, 5.0]
# print(m)

mean = train_data.mean(axis=0)  # 对列求平均值，得到(13,)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

# 测试数据标准化用到的均值和标准差都是在训练数据上得到的
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(
        layers.Dense(64,
                     activation=activations.relu,
                     input_shape=(train_data.shape[1], )))

    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.rmsprop(),
                  loss=losses.mse,
                  metrics=[metrics.mae])

    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    validation_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    validation_targets = train_targets[i * num_val_samples:(i + 1) *
                                       num_val_samples]

    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]
    ],
                                        axis=0)

    partial_train_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]
    ],
                                           axis=0)

    model = build_model()
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        epochs=num_epochs,
                        validation_data=(validation_data, validation_targets),
                        batch_size=16)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

plt.figure()
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[18:])

plt.figure()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

plt.show()
