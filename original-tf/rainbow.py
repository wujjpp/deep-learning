# -*-coding:utf-8-*-

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(-5, 5, 0.1)
y = np.arange(-1, 5, 0.1)

X, Y = np.meshgrid(x, y)

Z = X*X + Y*Y + 2*Y

plt.xlabel('x')
plt.ylabel('y')

ax.plot_surface(X, Y, Z, rstride=2, cstride=1, cmap='rainbow')

plt.show()