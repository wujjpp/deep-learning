import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from config import train_cats_dir
import matplotlib.pyplot as plt

fnames = [
    os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)
]

img_path = fnames[3]

print(img_path)

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)

print(x.shape)

x = x.reshape((1, ) + x.shape)

print(x)

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 10 == 0:
        break

plt.show()