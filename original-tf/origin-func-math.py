# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt

'''
a = 4, b = 5
a * 5 + b * 3 = 20 + 15 = 35
a * 2 + b * 7 =  8 + 35 = 43

a = 1/5 * (-a * 0 - b * 3 + 35)
b = 1/7 * (-a * 2 - b * 0 + 43)

a  |   0    -3/5   |  35/5
b  |  -2/7     0   |  43/7

'''

# 精确求解
W = np.asarray(
    [
        [5, 3],
        [2, 7]
    ], dtype='float64'
)
b = np.asarray([35, 43], dtype='float64')
result = np.linalg.solve(W, b)
print(result)

# 迭代求解
# 要求矩阵中的值服从 (-1, 1)

W = np.asarray(
    [
        [0,           -3./5.],
        [-2./7.,          0.]
    ], dtype='float32'
)

b = np.asarray([35./5., 43/7.], dtype='float64')

'''
| a |
| b |
'''

expect = np.zeros(2)

epochs = 0
metrics = []
e = 0.000000001

while True:
    epochs += 1
    # save last
    last = expect
    expect = W.dot(expect) + b

    # mean squared error
    mse = np.sqrt(np.square(expect - last).sum())
    metrics.append(mse)
    if mse < e:
        break


print(expect)

plt.plot(range(epochs), metrics, label='mean squared error')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.title('values: (a: {0:.5f}, b: {1:.5f})'.format(expect[0], expect[1]))
plt.legend()
plt.show()
