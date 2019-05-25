from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)
model.summary()


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random(
        (1, size, size, 3)) * 20 + 128.  # 从一张带有噪声的灰度图开始

    step = 1.

    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


# plt.imshow(generate_pattern('block3_conv1', 0))
# plt.show()


def show_filters(layer_name):
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3),
                       dtype='uint8')

    for i in range(8):
        for j in range(8):
            filter_image = generate_pattern(layer_name, i + (j * 8), size=size)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size

            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size

            results[horizontal_start:horizontal_end, vertical_start:
                    vertical_end, :] = filter_image

    plt.figure(figsize=(20, 20))
    plt.title(layer_name)
    plt.imshow(results)


for layer_name in [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
        'block5_conv1'
]:
    print('try to show filter for layer: {}'.format(layer_name))
    show_filters(layer_name)

plt.show()
