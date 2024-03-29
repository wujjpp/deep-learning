from keras import models, layers, losses, optimizers, activations, Input, utils, metrics
import numpy as np

# 文本片段词典大小
text_vocabulary_size = 10000
# 问题词典大小
question_vocabulary_size = 10000
# 答案词典大小，即：总答案个数，后面用softmax激活器，类似多分类问题的总分类数
answer_vocabulary_size = 500

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

answer = layers.Dense(answer_vocabulary_size,
                      activation=activations.softmax)(concatenated)

model = models.Model([text_input, question_input], answer)

model.compile(optimizer=optimizers.Adam(),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])

model.summary()
'''
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
texts (InputLayer)              (None, None)         0                                            
__________________________________________________________________________________________________
questions (InputLayer)          (None, None)         0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, None, 64)     640000      texts[0][0]                      
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, None, 64)     640000      questions[0][0]                  
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 32)           12416       embedding_1[0][0]                
__________________________________________________________________________________________________
lstm_2 (LSTM)                   (None, 64)           33024       embedding_2[0][0]                
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 96)           0           lstm_1[0][0]                     
                                                                 lstm_2[0][0]                     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 500)          48500       concatenate_1[0][0]              
==================================================================================================
Total params: 1,373,940
Trainable params: 1,373,940
Non-trainable params: 0
'''

# prepare mock data
num_samples = 10000
maxlen = 100

texts = np.random.randint(1, text_vocabulary_size, size=(num_samples, maxlen))
questions = np.random.randint(1,
                              question_vocabulary_size,
                              size=(num_samples, maxlen))

answers = np.random.randint(answer_vocabulary_size, size=num_samples)
answers = utils.to_categorical(answers, answer_vocabulary_size)

model.fit([texts, questions], answers, epochs=10, batch_size=128)

model.fit({
    'texts': texts,
    'questions': questions
},
          answers,
          epochs=10,
          batch_size=128)
