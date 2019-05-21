from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics

(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

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


model = build_model()
model.fit(train_data, train_targets, epochs=88, batch_size=16)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print('test_mse_score: %s, test_mae_score: %s' %
      (test_mse_score, test_mae_score))
