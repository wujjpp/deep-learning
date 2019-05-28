import os
import numpy as np
from keras import models, layers, optimizers, losses, activations
from callback_helper import generate_callbacks
import pandas as pd

# https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip

# 读取数据
data_dir = '/home/jp/workspace'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

original_dataset = pd.read_csv(fname)
dataset = original_dataset.copy()
dataset.pop('Date Time')

# 数据标准化 - 只使用前200000个数据（训练数据）来计算平均值和标准差，但是计算结果将用于全部数据
data_stats = dataset[:200000].describe()
data_stats = data_stats.transpose()
''' data_stats details
                    count         mean        std      min      25%       50%      75%      max
p (mbar)         200000.0   988.886359   8.480455   913.60   983.90   989.360   994.41  1012.84
T (degC)         200000.0     9.077349   8.852521   -23.01     3.02     9.410    15.47    35.86
Tpot (K)         200000.0   283.146313   8.953264   250.60   277.15   283.505   289.56   309.93
Tdew (degC)      200000.0     4.448547   7.165868   -25.01    -0.19     5.020     9.83    20.58
rh (%)           200000.0    75.354059  16.727358    12.95    64.22    78.700    89.10   100.00
VPmax (mbar)     200000.0    13.382955   7.689165     0.95     7.59    11.820    17.60    59.02
VPact (mbar)     200000.0     9.296955   4.198092     0.79     6.02     8.740    12.16    24.28
VPdef (mbar)     200000.0     4.085917   4.840356     0.00     0.85     2.190     5.52    42.12
sh (g/kg)        200000.0     5.875211   2.665656     0.50     3.80     5.520     7.68    15.49
H2OC (mmol/mol)  200000.0     9.405210   4.252074     0.80     6.10     8.840    12.29    24.67
rho (g/m**3)     200000.0  1217.514297  42.488949  1059.45  1187.45  1213.280  1243.59  1393.54
wv (m/s)         200000.0     2.151004   1.536668     0.00     1.02     1.800     2.89    14.63
max. wv (m/s)    200000.0     3.569509   2.330679     0.00     1.80     3.040     4.77    23.50
wd (deg)         200000.0   176.217034  86.613447     0.00   131.60   199.300   236.30   360.00
'''
dataset = (dataset - data_stats['mean']) / data_stats['std']

# convert pd data frame to numpy
float_data = dataset.to_numpy()


def generator(data,
              lookback,
              delay,
              min_index,
              max_index,
              shuffle=False,
              batch_size=128,
              step=6):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback,
                                     max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i + len(rows)
        '''
        -----------------------------------------------------------------------------------
        rows             : rows保存了即将要准备的数据在原始数据中的开始索引
        len(rows)        : batch_size, 一次取多少笔数据
        lookback         : 5天的数据。注：每10分钟一笔数据，5天的总数据量是： 6 * 24 * 5 = 720
        step             : 数据采样周期（单位：时间步）。取值6表示，每个小时取一个样本
        lookback // step : 5天中总共需要的样本数量
        data.shape[-1]   : 每笔原始数据的column数量。本例中固定是14，从前面的data_stats的行数可以看出
        delay            : 站在当前数据点看未来24小时的那个数据点，索引相差 6 * 24 = 144
        -----------------------------------------------------------------------------------
        samples.shape    : (一次取多少笔数据, 5天中总共需要的样本数量, 每笔原始数据的column数)
                           注：(batch_size, 120, 14)
        targets.shape    : (一次取多少笔数据，)
                           注：(batch_size, )
        -----------------------------------------------------------------------------------
        '''
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            '''
            -----------------------------------------------------------------------------------
            samples      : 看当前数据点的前5天的数据（共720个样本），每隔step（6个）取一个样本， 720 / 6 = 120个样本数据
            taregts      : 取当前数据点24小时之后的数据(未来24小时总共144个数据点)，所以target数据取 data[rows[j] + delay][1]
                           注：索引为1，是因为目标数据（温度）在第2列， zero based index为1
            -----------------------------------------------------------------------------------
            举例说明：
            1. 假如 rows = [720], 这边batch_size相当于为1
            2. samples = [
                            [
                                data[0],
                                data[6],
                                data[12],
                                ....
                                data[702],
                                data[708],
                                data[714]
                            ]
                        ]
               注意： data[n]为长度为14的一维数组, 每隔6个数据点（一小时间隔），取一个样本
            3. tagerts = [
                    data[720 + 144][1]
                ]
            4. 总体逻辑为：站在当前样本的时间点，将过去5天的数据（每个小时取一个样本）作为输入，
                         未来24小时的那个数据点的温度作为目标输出，进行拟合
            -----------------------------------------------------------------------------------
            '''
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


lookback = 720
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

model = models.Sequential()

model.add(
    layers.GRU(32,
               dropout=0.1,
               recurrent_dropout=0.5,
               return_sequences=True,
               input_shape=(None, float_data.shape[-1])))

model.add(layers.GRU(64, activation=activations.relu, dropout=0.1, recurrent_dropout=0.5))

model.add(layers.Dense(1))

model.compile(optimizer=optimizers.RMSprop(), loss=losses.mae)

model.summary()

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              callbacks=generate_callbacks())
