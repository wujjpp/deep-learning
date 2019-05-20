from keras.datasets import mnist
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import activations
from keras.utils import to_categorical
from utils import show_ndarray_info

(train_images, train_labels), (test_images, test_lables) = mnist.load_data()

show_ndarray_info(train_images, '[before]:train_images')
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
show_ndarray_info(train_images, '[after]:train_images')

show_ndarray_info(train_labels, '[before]:train_labels')
print('[before]:trains_lables[0]', train_labels[0])
train_labels = to_categorical(train_labels)
print('[after]:trains_lables[0]', train_labels[0])
show_ndarray_info(train_labels, '[after]:train_labels')

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
test_lables = to_categorical(test_lables)

network = models.Sequential()

# 假如前面不对数据做reshape的话，可以添加一层Flatten层
# network.add(layers.Flatten(input_shape=(28, 28)))

network.add(layers.Dense(512, activation=activations.relu))

network.add(layers.Dense(10, activation=activations.softmax))

# network.compile(optimizer=optimizers.rmsprop(),
#                 loss=losses.categorical_crossentropy,
#                 metrics=['accuracy'])

# network.fit(train_images, train_labels, epochs=5, batch_size=128)

# test_loss, test_acc = network.evaluate(test_images, test_lables)
# print('test_loss: %s, test_acc: %s' % (test_loss, test_acc))
