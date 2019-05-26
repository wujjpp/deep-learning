import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models, layers, activations, losses, optimizers, metrics
import callback_helper

imbd_dir = '/home/jp/workspace/aclImdb'
train_dir = os.path.join(imbd_dir, 'train')
test_dir = os.path.join(imbd_dir, 'test')


def load_data(text_dir):
    texts = []
    labels = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(text_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname)) as f:
                    texts.append(f.read())
                if label_type == 'neg':
                    labels.append(0.)
                else:
                    labels.append(1.)

    return np.asarray(texts), np.asarray(labels)


x_train, y_train = load_data(train_dir)

maxlen = 300
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

print('Found %s unique tokens' % len(word_index))

x_train = tokenizer.texts_to_sequences(x_train)

x_train = pad_sequences(x_train,
                        padding='post',
                        truncating='post',
                        maxlen=maxlen)

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

# 这种方式的赋值，是numpy功能，普通list无效
x_train = x_train[indices]
y_train = y_train[indices]

# 定义模型
model = models.Sequential()
model.add(layers.Embedding(max_words, 100, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))
model.summary()

model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

callbacks = callback_helper.generate_callbacks_include_early_stop()

history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    validation_split=0.5,
                    callbacks=callbacks)

# 准备测试数据
print('Prepare testdata')
x_test, y_test = load_data(test_dir)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test,
                       padding='post',
                       truncating='post',
                       maxlen=maxlen)

# 在测试集上测试
result = model.evaluate(x_test, y_test)
print(result)
