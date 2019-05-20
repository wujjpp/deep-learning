import numpy as np

x = np.random.random((2, ))
y = np.random.random((2, ))

z = np.dot(x, y)

print('x:\n', x)
print('y:\n', y)
print('np.dot(x, y):\n', z)


def naive_verctor_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]

    return z


z1 = naive_verctor_dot(x, y)
print('naive_verctor_dot(x, y):\n', z1)


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


def naive_matrix_vector_dot2(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        z[i] = naive_verctor_dot(x[i], y)
    return z


print('----------------------------------------------------')

x = np.random.random((3, 2))
y = np.random.random(2)
z = np.dot(x, y)
z1 = naive_matrix_vector_dot(x, y)
z2 = naive_matrix_vector_dot2(x, y)
print('x:\n', x)
print('y:\n', y)
print('np.dot(x, y):\n', z)
print('naive_matrix_vector_dot(x, y):\n', z1)
print('naive_matrix_vector_dot2(x, y):\n', z1)

print('----------------------------------------------------')


def naive_verctor_dot_n(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            print(row_x.shape)
            print(column_y.shape)
            z[i, j] = naive_verctor_dot(row_x, column_y)
    return z


x = np.random.random((3, 2))
y = np.random.random((2, 4))
z = np.dot(x, y)
z1 = naive_verctor_dot_n(x, y)
print('x:\n', x)
print('y:\n', y)
print('np.dot(x, y):\n', z)
print('naive_verctor_dot_n:\n', z1)

print('----------------------------------------------------')
x = np.random.random((2, 3, 4))
y = np.random.random((4, 5))
z = np.dot(x, y)
assert z.shape == (2, 3, 5)
