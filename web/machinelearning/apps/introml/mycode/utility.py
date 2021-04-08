'''
======================
3D surface (color map)
======================
Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
X = np.arange(0, 100, 10)
Y = np.arange(0, 100, 10)
X, Y = np.meshgrid(X, Y)

a_x = 0.2
a_y = 0.9
b_x = -0.5
b_y = -0.5
c_x = 0.7
c_y = -0.1

def Z(X, Y):
    u = (X**a_x * Y**a_y) + (X**b_x * Y**b_y) + (X**c_x * Y**c_y)
    return u
#
print(Z(100, 100))
print('----')
print(Z(1, 1))
#
# Plot the surface.
surf = ax.plot_surface(X, Y, Z(X, Y), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 300)
ax.zaxis.set_major_locator(LinearLocator())
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
