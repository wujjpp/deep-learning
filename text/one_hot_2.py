from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)  # 构建单词索引
print(tokenizer.index_word)

sequences = tokenizer.texts_to_sequences(samples)  # 将字串转换成整数索引组成的列表
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples)
print(one_hot_results.shape)
for l in one_hot_results:
    print(l)
