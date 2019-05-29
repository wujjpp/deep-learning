from keras import models, layers, losses, optimizers, activations, Input, utils
import numpy as np

# 文本片段词典大小
text_vocabulary_size = 10000
# 问题词典大小
question_vocabulary_size = 10000

maxlen = 500

num_samples = 10000

# 收入分类数
num_income_group = 10

# 处理文本输入
text_input = Input(shape=(None, ), dtype='int32', name='texts')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

# 处理问题输入
question_input = Input(shape=(None, ), dtype='int32', name='questions')
embedded_question = layers.Embedding(question_vocabulary_size,
                                     64)(question_input)
encoded_question = layers.LSTM(64)(embedded_question)

# 连接编码后的问题和文本
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

# 回归问题
age_prediction = layers.Dense(1, name='age')(concatenated)

# 多分类问题
income_prediction = layers.Dense(num_income_group,
                                 activation=activations.softmax,
                                 name='income')(concatenated)

# 二分类问题
gender_prediction = layers.Dense(1,
                                 activation=activations.sigmoid,
                                 name='gender')(concatenated)

model = models.Model([text_input, question_input],
                     [age_prediction, income_prediction, gender_prediction])

model.summary()

model.compile(
    optimizer=optimizers.Adam(),  # 这个注释只是为了自动格式化
    loss={
        'age': losses.mse,
        'income': losses.categorical_crossentropy,
        'gender': losses.binary_crossentropy
    },
    loss_weights={
        'age': 0.25,
        'income': 1.,
        'gender': 10.
    })

# 准备模拟数据
# x_train(s)
texts = np.random.randint(1, text_vocabulary_size, size=(num_samples, maxlen))
questions = np.random.randint(1,
                              question_vocabulary_size,
                              size=(num_samples, maxlen))

# y_train(s)
# 1. 预测age是回归问题， lables.shape = (num_simples,)
age_targets = np.random.randint(16, 40, size=num_samples)

# 2. 预测income是多分类问题，有2种标签向量化方式，这边使用one-hot编码,lables.shape = (num_samples, num_income_group)
income_targets = np.random.randint(num_income_group, size=num_samples)
income_targets = utils.to_categorical(income_targets, num_income_group)

# 3. 预测gender是两分类问题, labels.shape = (num_samples, )
gender_targets = np.random.randint(2, size=num_samples)

model.fit(
    {
        'texts': texts,  # 这个注释只是为了自动格式化
        'questions': questions
    },
    {
        'age': age_targets,
        'income': income_targets,
        'gender': gender_targets
    },
    epochs=10,
    batch_size=64)
