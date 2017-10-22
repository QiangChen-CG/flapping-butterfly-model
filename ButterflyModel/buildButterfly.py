"""

DOCSTRING HERE

"""
import numpy as np
from scipy import interpolate


def create_3d_array(x_pts, y_pts, z_pts=None):
    if z_pts is None:
        z_pts = np.zeros(len(x_pts))

    return np.stack((x_pts, y_pts, z_pts), axis=1)


class BodyEllipse(object):
    """Define an elliptical body part with a mass, length, and diameter
    properties.  Assumes uniform density.

    mass = mass of the body part in grams(g), must be positive.
    length = length of the body part in meters(m), cannot be negative.
    diameter = diameter of the body part in meters(m), cannot be negative.
    """

    def __init__(self, mass, length, diameter, n_elements):
        self.mass = mass
        self.length = length
        self.diameter = diameter
        self.n_elements = n_elements

        # Raise exceptions to enforce acceptable parameter values
        if not (mass > 0):
            raise ValueError("mass must be positive")
        if length < 0:
            raise ValueError("length cannot be negative")
        if diameter < 0:
            raise ValueError("radius cannot be negative")

        self.x_ele, self.d_ele, self.w_ele = self.get_ellipse_elements(
            self.length, self.diameter, self.n_elements)

        self.moi = self.get_moi(self.mass, self.length, self.diameter)

    @staticmethod
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

    @staticmethod
    def get_moi(mass, length, diameter):
        """Returns moment of inertia tensor of the body part at the root of
        the body part.  First calculates the moment of inertia about the
        center of mass of the part, then at the root using the parallel axis
        theorem
        """
        tensor_moi = np.zeros((3, 3))
        tensor_moi[1][1] = (mass / 20) * (length ** 2 + diameter ** 2) + \
                           (mass / 4) * length ** 2
        return tensor_moi


class BodySphere(object):
    """Define a spherical body part with a mass and diameter properties.
    Assumes uniform density.

    mass = mass of the body part in grams(g), must be positive.
    diameter = diameter of the body part in meters(m), cannot be negative.
    """

    def __init__(self, mass, diameter):
        self.mass = mass  # grams
        self.diameter = diameter  # meters

        # Raise exceptions to enforce acceptable parameter values
        if not (mass > 0):
            raise ValueError("mass must be positive")
        if diameter < 0:
            raise ValueError("radius cannot be negative")

        self.moi = self.get_moi(self.mass, self.diameter)

    @staticmethod
    def get_moi(mass, diameter):
        """Returns moment of inertia tensor of the body part at the root of
        the body part.  First calculates the moment of inertia about the
        center of mass of the part, then at the root using the parallel axis
        theorem
        """
        tensor_moi = np.zeros((3, 3))
        tensor_moi[1][1] = (mass / 10) * diameter ** 2 + \
                           (mass / 4) * diameter ** 2
        return tensor_moi


class Wing(object):
    """Define a wing with a mass and shape.  The shape is defined by
    coordinates along the leading and trailing edges.  Modeled as uniform
    density. Methods are available to divide the wing into a number of
    smaller elements, and store information about those elements to be used
    for later calculation.

    Coordinate system:  The wing coordinate system has its origin at the
    root of the wing where it connects to the body.  The x-axis is parallel
    to the wing chord and is positive from the trailing edge to the leading
    edge.  The y-axis is parallel to the wing span and is positive from the
    origin moving away from the body.  When the wing angles are zero,
    the wing coordinate system and body coordinate system have parallel axes.

    Attributes:
        mass: Mass of the wing.
        y_ele: Spanwise location of element centerpoints.
        ele_width: Width of each individual element.
        chord_length: Chord length(distance from leading to trailing edge) of
        each element.
        centroid: Centroid(also center of mass, assuming uniform density) of
        the entire wing.
        chord25: Points on the elements that are 25% of the chord length
        behind the leading edge.  Assumed to be the aerodynamic center where
        the resultant lift force acts on the element.
        chord75: Used when the angle of attack goes beyond 90 degrees,
        resulting in the morphological trailing edge acting as the
        aerodynamic leading edge.  The aerodynamic center is therefore 25%
        of the chord length behind the trailing edge, or 75% of the chord
        length in front of the leading edge.
        moi: Moment of inertia tensor at the root of the wing in the wing
        coordinate system.

        UNITS - All lengths and positions are in meters.  Mass in in grams.

    """

    def __init__(self, raw_wing_coords, mass, n_elements):
        """Creates the initial wing object with input wing coordinates,
        mass and number of elements.  Divides the wing into n elements of
        equal length along the wingspan, then calculates properties of those
        elements needed in the model.

        Args:
            raw_wing_coords (dict): Contains 'x', 'y_le', 'y_te' entries
            containing arrays representing the x,y coordinates of points
            along the leading edge and trailing edge of the wing.
            mass: Mass of the wing in grams(g).
            n_elements: Number of elements to divide the wing into.

        NOTE: Because the wing point coordinate data is in the y = f(x)
        format, the x and y axes in some of the functions are switched from
        what is used in the wing coordinate system(i.e. the x data for the
        wing points are along the span of the wing, which is the y-axis in
        the wing frame, and visa versa).  This is handled in the code but
        should be noted as it is here, to prevent confusion.

        """
        self.mass = mass

        # Generate y-coordinates of the element centers and x-coordinates of
        #  the leading and trailing edges of those elements
        self.y_ele, [self.x_le, self.x_te], self.ele_width = \
            self.interp_elements(raw_wing_coords['x'],
                                 [raw_wing_coords['y_le'],
                                  raw_wing_coords['y_te']],
                                 n_elements)

        # Generate chord lengths of each element
        self.chord_length = self.get_chord_length(self.x_le, self.x_te)

        # Find centers of elements then get centroid for center of mass
        self.midchord = self.get_chord_pos(self.x_le, self.x_te, 0.5)
        self.centroid = self.find_centroid(np.stack((self.y_ele,
                                                     self.midchord)),
                                           self.chord_length)

        # Generate y-coordinates for the 25% chordline and 75% chordline
        self.chord25 = self.get_chord_pos(self.x_le, self.x_te, 0.25)
        self.chord75 = self.get_chord_pos(self.x_le, self.x_te, 0.75)

        # Generate moment of inertia tensor of the entire wing
        self.moi = self.get_moi(self.y_ele, self.midchord, self.ele_width,
                                self.chord_length, mass)

        # Convert x and y data in N-by-3 arrays in the wing frame
        self.leading_edge = create_3d_array(self.x_le, self.y_ele)
        self.trailing_edge = create_3d_array(self.x_te, self.y_ele)
        self.midchord = create_3d_array(self.midchord, self.y_ele)
        self.centroid = create_3d_array([self.centroid[1]], [self.centroid[0]])
        self.chord25 = create_3d_array(self.chord25, self.y_ele)
        self.chord75 = create_3d_array(self.chord75, self.y_ele)

    @staticmethod
    def interp_elements(x, y, n, axis=1, method='linear'):
        """Takes 1-D x array and N-D y arrays corresponding to points along
        the wing edge and divides the wing into n spanwise elements, returning
        interpolated x and y values at the center of those elements, as well
        as the width of the elements. Interpolation axis and method can be
        optionally specified, with axis=1 and method='cubic' being the
        defaults.

        """
        f = interpolate.interp1d(x, y, axis=axis, kind=method)

        w_ele = (x[-1] - x[0]) / n  # width of each element
        # x points at center of elements
        x_ele = np.linspace((x[0] + w_ele / 2),
                            (x[-1] - w_ele / 2),
                            n)
        y_ele = f(x_ele)

        return x_ele, y_ele, w_ele

    @staticmethod
    def get_chord_length(y_le, y_te):
        """Takes y_le, y_te arrays representing the y values of points along
        the leading edge and trailing edge of a wing and calculates the chord
        length at those positions by subtracting the trailing edge y-values
        from the leading edge y-values.
        """
        return y_le - y_te

    @staticmethod
    def get_chord_pos(y_le, y_te, chord_pos):
        """Takes y_le, y_te arrays representing the y values of points along the
        leading edge and trailing edge of a wing and finds the points that are
        chord_pos*chord_length back from the leading edge.  Using for
        calculating the aerodynamic center and center of mass of wing elements
        """
        chord_length = y_le - y_te
        return y_le - chord_pos * chord_length

    @staticmethod
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
            x_bar = np.divide(np.sum(np.multiply(axis, weight)),
                              np.sum(weight))
            centroid.append(x_bar)
        return centroid

    @staticmethod
    def get_moi(y_ele, x_ele, w_ele, chords, mass):
        """Returns moment of inertia tensor of the wing at the origin of the
        wing coordinate system.  First calculates the moment of inertia
        about center of mass of each element, then about the wing root using
        the parallel axis theorem.  Sums the tensors of each element to
        determine the total moment of inertia
        """
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


class Butterfly(object):
    def __init__(self, head, thorax, abdomen, wing):
        self.head = head  # BodyPart class
        self.thorax = thorax  # BodyPart class
        self.abdomen = abdomen  # BodyPart class
        self.wing = wing  # Wing class

# TEST CODE, DELETE LATER
