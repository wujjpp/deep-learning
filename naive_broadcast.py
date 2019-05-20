import numpy as np

x = np.random.random((4, 3, 2))
y = np.random.random((3, 2))

z = np.maximum(x, y)

print(x)
print('===================================================')
print(y)
print('===================================================')
print(z)
