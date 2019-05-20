import numpy as np
from keras.datasets import mnist
from utils import show_ndarray_info
import matplotlib.pyplot as plt

x = np.array(12)

print(x, x.ndim)

x = np.array([1, 2, 3, 4, 56])
print(x, x.ndim)

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x, x.ndim)

x = np.array([[[11, 12, 13], [14, 15, 16], [17, 18, 19]],
              [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
              [[31, 32, 33], [34, 35, 36], [37, 38, 39]]])
print(x, x.ndim)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

show_ndarray_info(train_images, 'train_images')

img = train_images[4]
plt.figure()
plt.imshow(img, cmap=plt.cm.binary)

# train_images[10:,100, :, :] => train_images[10:100, 0:28, 0:28]
my_slice = train_images[10:100]
show_ndarray_info(my_slice, 'my_slice')

my_img = img[7:14, 7:14]
plt.figure()
plt.imshow(my_img, cmap=plt.cm.binary)
plt.show()
