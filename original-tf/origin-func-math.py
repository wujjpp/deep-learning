# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt

'''
a = 4, b = 5, c = 6
a * 5 + b * 3 + c * 3 = 20 + 15 + 18 = 53
a * 2 + b * 7 + c * 2 =  8 + 35 + 12 = 55
a * 3 + b * 4 + c * 9 = 12 + 20 + 54 = 86

a = 1/5 * (-a * 0 - b * 3 - c * 3 + 53)
b = 1/7 * (-a * 2 - b * 0 - c * 2 + 55)
c = 1/9 * (-a * 3 - b * 4 - c * 0 + 86)

a  |   0    -3/5   -3/5  |  53/5
b  |  -2/7     0   -2/7  |  55/7
c  |  -3/9  -4/9      0  |  86/9

'''

# 精确求解
W = np.asarray(
    [
        [5, 3, 3],
        [2, 7, 2],
        [3, 4, 9]
    ], dtype='float64'
)
b = np.asarray([53, 55, 86], dtype='float64')
result = np.linalg.solve(W, b)
print(result)

# 迭代求解
# 要求矩阵中的值服从 (-1, 1)

W = np.asarray(
    [
        [0,              -3./5.,     -3./5.],
        [-2./7.,             0.,     -2./7.],
        [-3.0/9.0,     -4.0/9.0,          0]
    ], dtype='float32'
)

b = np.asarray([53/5., 55/7., 86/9.], dtype='float64')

expect = np.zeros(3)

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
plt.title('values: (a: {0:.5f}, b: {1:.5f}, c: {2:.5f})'.format(expect[0], expect[1], expect[2]))
plt.legend()
plt.show()
