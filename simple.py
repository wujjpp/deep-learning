from keras import layers, models, optimizers, callbacks
import numpy as np

x_train = np.random.random(size=(10000, 100))
y_train = np.random.random(size=10000)

# print(x_train, y_train)

model = models.Sequential()

model.add(layers.Dense(16, input_shape=(100, ), name='layer-1'))
model.add(layers.Dense(16, name='layer-2'))
model.add(layers.Dense(1, name='layer-3'))

model.compile(loss='mean_squared_error',
              optimizer=optimizers.RMSprop(0.001),
              metrics=['mae'])

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
layer-1 (Dense)              (None, 16)                1616      
_________________________________________________________________
layer-2 (Dense)              (None, 16)                272       
_________________________________________________________________
layer-3 (Dense)              (None, 1)                 17        
=================================================================
Total params: 1,905
Trainable params: 1,905
Non-trainable params: 0
_________________________________________________________________

Param Count: 
    layer-1:
        input shape:  (100, )
        output shape: (16, )
        W shape: (100, 16)
        b shape: (16, )
        So the Param's count is: 100 * 16 + 16 = 1616
    
    layer-2:
        input shape:  (16, )
        output shape: (16, )
        W shape: (16, 16)
        b shape: (16, )
        So the Param's count is: 16 * 16 + 16 = 272
        
    layer-2:
        input shape:  (16, )
        output shape: (1, )
        W shape: (16,)
        b shape: (1, )
        So the Param's count is: 16 * 1 + 1 = 272
        
      
'''

model.fit(x_train,
          y_train,
          callbacks=[callbacks.TensorBoard(log_dir='results')])
