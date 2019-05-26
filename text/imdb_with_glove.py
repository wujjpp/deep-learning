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
                    labels.append(0)
                else:
                    labels.append(1)

    return np.asarray(texts), np.asarray(labels)


texts, labels = load_data(train_dir)

maxlen = 100
train_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens' % len(word_index))

sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences,
                     padding='post',
                     truncating='post',
                     maxlen=maxlen)
labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

# 这种方式的赋值，是numpy功能，普通list无效
data = data[indices]
labels = labels[indices]

# 将数据分成训练和检验两部分
x_train = data[:train_samples]
y_train = labels[:train_samples]
x_validation = data[train_samples:train_samples + validation_samples]
y_validation = labels[train_samples:train_samples + validation_samples]

# # 解析GloVe词嵌入文件
glove_dir = '/home/jp/workspace/glove.6B'
embeddings_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s words in vectors.' % len(embeddings_index))

# # 准备GloVe词嵌入矩阵
embeddings_dim = 100
embeddings_matrix = np.zeros((max_words, embeddings_dim))
for word, index in word_index.items():
    if index < max_words:
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embeddings_matrix[index] = embeddings_vector

# 定义模型
model = models.Sequential()
model.add(layers.Embedding(max_words, 100, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(21, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))
model.summary()

# 将预训练的词嵌入加载到Embedding层
model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

model.compile(optimizer=optimizers.rmsprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

callbacks = callback_helper.generate_callbacks()

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_validation, y_validation),
                    callbacks=callbacks)

# 准备测试数据
x_test, y_test = load_data(test_dir)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test,
                       padding='post',
                       truncating='post',
                       maxlen=maxlen)

# 在测试集上测试
result = model.evaluate(x_test, y_test)
print(result)
