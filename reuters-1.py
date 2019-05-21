from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
import copy

# load data
(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

# load word index, {"foo": 23, "bar": 45}
word_index = reuters.get_word_index()

# reverse word index, {23: "foo", 45: "bar"}
reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def show_content(content_data):
    print(' '.join([reverse_word_index.get(i - 3, '?') for i in content_data]))


# for i in range(10):
#     print('=====================================================')
#     print(train_data[i])
#     print('-----------------------------------------------------')
#     show_content(train_data[i])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# 将标签数据处理成one-hot编码
one_hot_train_lables = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# 定义模型
model = models.Sequential()
model.add(layers.Dense(64, activation=activations.relu, input_shape=(10000, )))
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(46, activation=activations.softmax))

# 编译模型
model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

# 训练
history = model.fit(x_train, one_hot_train_lables, epochs=9, batch_size=512)

history_dict = history.history

# plt绘制训练损失和验证损失曲线图
loss = history_dict['loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'r', label="Training loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt绘制训练精度和验证精度曲线图
acc = history_dict['acc']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, 'r', label="Training acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 在测试集上验证
result1 = model.evaluate(x_test, one_hot_test_labels)
print(result1)

# 在测试数据上预测结果
predications = model.predict(x_test)


def show_predication(predication):
    print('shape:%s, sum: %s, pecent: %s, mean: %s' %
          (predication.shape, np.sum(predication), np.amax(predication),
           np.argmax(predication)))


for i in range(10):
    show_predication(predications[i])
    correct = '😀'
    if np.argmax(predications[i]) != test_labels[i]:
        correct = '😱'
    print('actual: %s, results: %s' % (test_labels[i], correct))

# 下面代码仅仅为了演示随机下的精度
# 完全随机精度
test_label_copy = copy.copy(test_labels)
np.random.shuffle(test_label_copy)
hits_array = (np.array(test_labels) == np.array(test_label_copy))

randomAcc = float(np.sum(hits_array)) / len(test_labels)
print(randomAcc)

# 显示plt
plt.show()
