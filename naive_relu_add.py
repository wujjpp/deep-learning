import numpy as np

x = np.random.random((5, 3))
y = np.random.random((5, 3))


def naive_relu_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


print(x)
print(y)
print(naive_relu_add(x, y))
