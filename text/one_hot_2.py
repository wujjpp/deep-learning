from keras.preprocessing.text import Tokenizer
import keras

samples = ['The cat sat on the mat', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(samples)  # 构建单词索引
print('index_word:')
print(tokenizer.index_word)

sequences = tokenizer.texts_to_sequences(samples)  # 将字串转换成整数索引组成的列表
print('texts_to_sequences')
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples)
print('texts_to_matrix')
for l in one_hot_results:
    print(l)

for sample in samples:
    hot = keras.preprocessing.text.one_hot(sample, 1000)
    print(hot)
