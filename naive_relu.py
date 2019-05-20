import numpy as np

x = np.random.random((5, 3))

print(x)

x = x - 0.5

print('=========================================')
print(x)


def naive_relu(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0.)

    return x


print('=========================================')
print(naive_relu(x))
