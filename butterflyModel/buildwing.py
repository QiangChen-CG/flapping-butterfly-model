import numpy as np
from scipy import interpolate
# import scipy
import pickle
import matplotlib.pyplot as plt

bf_file = open('treeNymph.pkl', 'rb')
bf = pickle.load(bf_file)
wc = bf.wing.wingcoords

n_ele = 25

def interp_elements(x, y, n, axis=1, method='linear'):
    """Takes 1-D x array and N-D y arrays corresponding to points along the
    wing edge and divides the wing into n spanwise elements, returning
    interpolated x and y values at the center of those elements.
    Interpolation axis and method can be optionally specified, with axis=1
    and method='cubic' being the defaults.
    """
    f = interpolate.interp1d(x,y, axis=axis, kind=method)

    w_ele = (x[-1] - x[0]) / n # width of each element
    # x points at center of elements
    x_ele = np.linspace((x[0] + w_ele / 2),
                        (x[-1] - w_ele / 2),
                        n)
    y_ele = f(x_ele)

    return x_ele, y_ele


fy_le = interpolate.interp1d(wc['x'], wc['y_le'])#), kind='cubic')
fy_te = interpolate.interp1d(wc['x'], wc['y_te'])#, kind='cubic')
#
w_ele = (wc['x'][-1] - wc['x'][0]) / n_ele
xmids = np.linspace((wc['x'][0] + w_ele / 2),
                    (wc['x'][-1] - w_ele / 2),
                    n_ele)

x_mids, [yi_le, yi_te] = interp_elements(wc['x'], [wc['y_le'], wc['y_te']], 25)


# FUNCTION FOR centers, chord25, chord75 HERE (reusable function??)

# chord = fy_le(xmids) - fy_te(xmids)
chord = yi_le - yi_te
centers = yi_le - 0.5 * chord
chord_25 = yi_le - 0.25 * chord
chord_75 = yi_le - 0.75 * chord

# centroid
# x_bar = np.divide(np.sum(np.multiply(xmids, chord)),
#                   np.sum(chord))
# y_bar = np.divide(np.sum(np.multiply(centers, chord)),
#                   np.sum(chord))

# print(x_bar, y_bar)


def find_centroid(points, weight=None):
    """Takes a set of N-dimensional points and, optionally, an array of
    weights for those points, and returns an N-dimensional centroid.

    points = m-by-n array where m is the number of dimensions and n is the
    number of points
    weight = n-by-1 array of weighted values for the n-dimensional points
    """
    if weight is None:
        weight = np.ones(len(points[0]))
    centroid = []
    for axis in points:
        x_bar = np.divide(np.sum(np.multiply(axis, weight)), np.sum(weight))
        centroid.append(x_bar)
    return centroid


wing_centroid = find_centroid(np.stack((xmids, centers)), chord)


plt.close('all')
plt.plot(wc['x'], wc['y_le'])
plt.plot(wc['x'], wc['y_te'])
plt.plot(xmids, fy_le(xmids), 'go')
plt.plot(xmids, fy_te(xmids), 'go')
plt.plot(xmids, centers, 'ro')
plt.plot(xmids, chord_25, 'bo')
plt.plot(xmids, chord_75, 'bo')
plt.plot(wing_centroid[0], wing_centroid[1], 'kv')
plt.show()
