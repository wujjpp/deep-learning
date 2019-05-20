import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6]])
print('x:\n', x)

x = x.reshape((6))
print('x:\n', x)

x = x.reshape((6, 1))
print('x:\n', x)

x = x.reshape((2, 3))
print('x:\n', x)

x = np.zeros((300, 20))
print(x.shape)
x = x.transpose()
print(x.shape)
