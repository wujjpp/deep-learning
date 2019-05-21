from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt

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

# 将训练数据分成两份，一份用于训练，一份用于训练时评估
partial_x_train = x_train[1000:]
x_validation = x_train[:1000]

partial_y_train = one_hot_train_lables[1000:]
y_validation = one_hot_train_lables[:1000]

# 训练
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_validation, y_validation))

history_dict = history.history

# plt绘制训练损失和验证损失曲线图
loss = history_dict['loss']
validation_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, validation_loss, 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt绘制训练精度和验证精度曲线图
acc = history_dict['acc']
validation_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, 'bo', label="Training acc")
plt.plot(epochs, validation_acc, 'b', label="Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

result = model.evaluate(x_test, one_hot_test_labels)
print(result)
