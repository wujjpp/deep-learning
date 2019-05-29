from keras import layers, models, optimizers, losses, activations, Input

x = Input(shape=(36, 36, 3))

branch_a = layers.Conv2D(128, 1, activation=activations.relu, strides=2)(x)

branch_b = layers.Conv2D(128, 1, activation=activations.relu)(x)
branch_b = layers.Conv2D(128,
                         3,
                         activation=activations.relu,
                         strides=2,
                         padding='same')(branch_b)

branch_c = layers.AveragePooling2D(3, strides=2, padding='same')(x)
branch_c = layers.Conv2D(128, 3, activation=activations.relu,
                         padding='same')(branch_c)

branch_d = layers.Conv2D(128, 1, activation=activations.relu)(x)
branch_d = layers.Conv2D(128, 3, activation=activations.relu,
                         padding='same')(branch_d)
branch_d = layers.Conv2D(128,
                         3,
                         activation=activations.relu,
                         strides=2,
                         padding='same')(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

model = models.Model(x, output)
model.summary()
'''
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 36, 36, 3)    0                                            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 36, 36, 128)  512         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 36, 36, 128)  512         input_1[0][0]                    
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 18, 18, 3)    0           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 36, 36, 128)  147584      conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 18, 18, 128)  512         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 18, 18, 128)  147584      conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 18, 18, 128)  3584        average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 18, 18, 128)  147584      conv2d_6[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 18, 18, 512)  0           conv2d_1[0][0]                   
                                                                 conv2d_3[0][0]                   
                                                                 conv2d_4[0][0]                   
                                                                 conv2d_7[0][0]                   
==================================================================================================
Total params: 447,872
Trainable params: 447,872
Non-trainable params: 0
'''
