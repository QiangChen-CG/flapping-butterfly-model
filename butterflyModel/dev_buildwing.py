import numpy as np
from scipy import interpolate
# import scipy
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import buildbutterfly


def interp_elements(x, y, n, axis=1, method='linear'):
    """Takes 1-D x array and N-D y arrays corresponding to points along the
    wing edge and divides the wing into n spanwise elements, returning
    interpolated x and y values at the center of those elements, as well as
    the width of the elements. Interpolation axis and method can be
    optionally specified, with axis=1 and method='cubic' being the defaults.
    """
    f = interpolate.interp1d(x, y, axis=axis, kind=method)

    w_ele = (x[-1] - x[0]) / n  # width of each element
    # x points at center of elements
    x_ele = np.linspace((x[0] + w_ele / 2),
                        (x[-1] - w_ele / 2),
                        n)
    y_ele = f(x_ele)

    return x_ele, y_ele, w_ele


def get_chord_length(y_le, y_te):
    """Takes y_le, y_te arrays representing the y values of points along the
    leading edge and trailing edge of a wing and calculates the chord length
    at those positions by subtracting the trailing edge y-values from the
    leading edge y-values.
    """
    return y_le - y_te


def get_chord_pos(y_le, y_te, chord_pos):
    """Takes y_le, y_te arrays representing the y values of points along the
    leading edge and trailing edge of a wing and finds the points that are
    chord_pos*chord_length back from the leading edge.  Using for
    calculating the aerodynamic center and center of mass of wing elements
    """
    chord_length = y_le - y_te
    return y_le - chord_pos * chord_length


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


def get_moi(y_ele, x_ele, w_ele, chords, mass):
    """Returns moment of inertia tensor of the wing at the origin of the
    wing coordinate system.  First calculates the moment of inertia
    about center of mass of each element, then about the wing root using
    the parallel axis theorem.  Sums the tensors of each element to
    determine the total moment of inertia"""
    # sum element areas for total wing area
    a = w_ele * sum(chords)

    # get element mass by dividing element area by total area and
    # multiplying by total mass
    m_ele = mass * w_ele * np.divide(chords, a)

    tensor_moi = np.zeros((3, 3))
    for i in range(len(y_ele)):
        m = m_ele[i]
        # I_xx
        h = w_ele
        d_xx = y_ele[i]
        tensor_moi[0][0] += (m / 12) * (h ** 2) + m * (d_xx ** 2)
        # I_yy
        b = chords[i]
        d_yy = x_ele[i]
        tensor_moi[1][1] += (m / 12) * (b ** 2) + m * (d_yy ** 2)
        # I_zz
        d_zz = np.linalg.norm([d_xx, d_yy])
        tensor_moi[2][2] += (m / 12) * ((h ** 2) + (b ** 2)) + \
                            (m * (d_zz ** 2))

    return tensor_moi


bf_file = open('treeNymph.pkl', 'rb')
bf = pickle.load(bf_file)

print(bf)

print(bf.wing.trailing_edge[4][1])

def gen3dplot(axobj, pts):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
gen3dplot(ax, bf.wing.trailing_edge)
gen3dplot(ax, bf.wing.leading_edge)
gen3dplot(ax, bf.wing.midchord)
gen3dplot(ax, bf.wing.centroid)
gen3dplot(ax, bf.wing.chord25)
gen3dplot(ax, bf.wing.chord75)
plt.show()

# wc = bf.wing.wingcoords
#
# n_ele = 20
#
# x_mids, [yi_le, yi_te], w_ele = interp_elements(wc['x'], [wc['y_le'],
#                                                           wc['y_te']], 25)
#
# chord = get_chord_length(yi_le, yi_te)
# centers = get_chord_pos(yi_le, yi_te, 0.5)
# chord_25 = get_chord_pos(yi_le, yi_te, 0.25)
# chord_75 = get_chord_pos(yi_le, yi_te, 0.75)
#
# wing_centroid = find_centroid(np.stack((x_mids, centers)), chord)
#
# # wing_moi = get_moi(x_mids, centers, w_ele, chord, 0.0587)
# test_moi = get_moi([7], [2], 3, [4], 1)
#
# print(test_moi)
# # print(wing_moi)
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