from keras import models, layers, optimizers, activations, losses, Input, utils
import numpy as np

# 输入文字词典大小
vocabulary_size = 50000
# 每个post长度, 至少需要244
maxlen = 500
# 样本量
num_samples = 10000
# 收入分类数
num_income_group = 10

post_input = Input(shape=(None, ), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(post_input)

x = layers.Conv1D(128, 5, activation=activations.relu)(embedded_posts)
x = layers.MaxPooling1D(5)(x)

x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.MaxPooling1D(5)(x)

x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation=activations.relu)(x)

# 回归问题
age_prediction = layers.Dense(1, name='age')(x)

# 多分类问题
income_prediction = layers.Dense(num_income_group,
                                 activation=activations.softmax,
                                 name='income')(x)

# 二分类问题
gender_prediction = layers.Dense(1,
                                 activation=activations.sigmoid,
                                 name='gender')(x)

model = models.Model(post_input,
                     [age_prediction, income_prediction, gender_prediction])

model.summary()

# model.compile(
#     optimizer=optimizers.Adam(),  # 这个注释只是为了自动格式化
#     loss=[
#         losses.mse, losses.categorical_crossentropy, losses.binary_crossentropy
#     ],
#     loss_weights=[0.25, 1., 10.])

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

# 输入posts
posts = np.random.randint(1, vocabulary_size, size=(num_samples, maxlen))

# 预测age是回归问题， lables.shape = (num_simples,)
age_targets = np.random.randint(16, 40, size=num_samples)

# 预测income是多分类问题，有2种标签向量化方式，这边使用one-hot编码,lables.shape = (num_samples, num_income_group)
income_targets = np.random.randint(num_income_group, size=num_samples)
income_targets = utils.to_categorical(income_targets, num_income_group)

# 预测gender是两分类问题, labels.shape = (num_samples, )
gender_targets = np.random.randint(2, size=num_samples)

# model.fit(
#     posts,  # 这个注释只是为了自动格式化
#     [age_targets, income_targets, gender_targets],
#     epochs=10,
#     batch_size=64)

model.fit(
    posts,  # 这个注释只是为了自动格式化
    {
        'age': age_targets,
        'income': income_targets,
        'gender': gender_targets
    },
    epochs=10,
    batch_size=64)
