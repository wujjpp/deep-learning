from keras import models
from config import test_cats_dir
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# load model
model_file_name = 'results/run-0005/model.h5'
model = models.load_model(model_file_name)
model.summary()

# -----------------------------------------------------------
# load img
img_path = os.path.join(test_cats_dir, 'cat.1700.jpg')
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)  # (150, 150, 3)

img_tensor = np.expand_dims(
    img_tensor,
    # 增加一个维度， axis=0 -> (1, 150, 150, 3), axis=1 -> (150, 1, 150, 3)
    axis=0)
print('img_tensor.shape:', img_tensor.shape)

img_tensor /= 255.  # convert to float

plt.imshow(img_tensor[0])
plt.show()

# -----------------------------------------------------------

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# display layers & activations infomation
for layer, activation in zip(model.layers[:8], activations):
    print('-----------------{0}----------------------'.format(layer.name))
    print('input type:{0}, shape:{1}'.format(type(layer.input),
                                             layer.input.shape))
    print('output type:{0}, shape:{1}'.format(type(layer.output),
                                              layer.output.shape))
    print('activation shape: {0}'.format(activation.shape))

images_per_row = 16

for layer, activation in zip(model.layers[:8], activations):
    n_features = activation.shape[-1]
    size = activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, size * images_per_row))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activation[0, :, :, col * images_per_row + row]
            # channel_image -= channel_image.mean()
            # channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) *
                         size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer.name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()