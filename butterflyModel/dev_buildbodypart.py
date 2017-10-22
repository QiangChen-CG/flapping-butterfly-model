import numpy as np
from scipy import interpolate
# import scipy
import pickle
import matplotlib.pyplot as plt

n_ele = 20

my_len = 0.0289
my_diam = 0.006


def get_ellipse_elements(length, diameter, n):
    """DOCSTRING"""
    w_ele = length / n  # width of each element
    # x points at center of elements
    x_ele = np.linspace((w_ele / 2),
                        (length - w_ele / 2),
                        n)
    # diameter of each element using shifted ellipse equation
    a = length / 2
    b = diameter / 2
    d_ele = []
    for x in x_ele:
        d_i = 2 * (b / a) * np.sqrt((2 * a * x) -
                                         (x ** 2))
        d_ele.append(d_i)
    d_ele = np.asarray(d_ele)
    return x_ele, d_ele, w_ele




x_150, d_150, w_150 = get_ellipse_elements(my_len, my_diam, 150)

x_20, d_20, w_20 = get_ellipse_elements(my_len, my_diam, 5)

plt.plot(x_150, d_150 / 2, x_150, -d_150 / 2)
plt.plot(x_20, d_20 / 2, 'ro', x_20, -d_20 / 2, 'ro')
plt.axis('equal')

plt.show()
# plt.close('all')
# plt.plot(wc['x'], wc['y_le'])
# plt.plot(wc['x'], wc['y_te'])
# plt.plot(x_mids, yi_le, 'go')
# plt.plot(x_mids, yi_te, 'go')
# plt.plot(x_mids, centers, 'ro')
# plt.plot(x_mids, chord_25, 'bo')
# plt.plot(x_mids, chord_75, 'bo')
# plt.plot(wing_centroid[0], wing_centroid[1], 'kv')
# plt.show()
