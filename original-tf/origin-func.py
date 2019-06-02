import numpy as np

'''
a = 4, b = 5, c = 6
a * 1 + b * 3 + c * 3 = 4 + 15 + 18 = 37
a * 2 + b * 1 + c * 2 =  8 + 5 + 12 = 25
a * 3 + b * 4 + c * 1 = 12 + 20 + 6 = 38


a = -a * 0 - b * 3 - c * 3 + 37
b = -a * 2 - b * 0 - c * 2 + 25
c = -a * 3 - b * 4 - c * 0 + 38

a  |   0 -3 -3  |  37
b  |  -2  0 -2  |  25
c  |  -3 -4  0  |  38

'''

# 精确求解
A = np.asarray(
    [
        [1, 3, 3],
        [2, 1, 2],
        [3, 4, 1]
    ], dtype='float64'
)
b = np.asarray([37, 25, 38], dtype='float64')
result = np.linalg.solve(A, b)
print(result)

# 迭代求解
W = np.asarray(
    [
        [ 0, -3, -3],
        [-2,  0, -2],
        [-3, -4,  0]
    ], dtype='float32'
)

print(W.shape)

b = np.asarray([37, 25, 38], dtype='float32')

print(b.shape)

epochs = 2
expect = np.zeros(3)

for i in range(epochs):
    expect = W.dot(expect) + b
    # print(expect)

print(expect)