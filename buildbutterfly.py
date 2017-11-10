#!/usr/bin/env python
"""

DOCSTRING HERE

"""
from functools import reduce
from types import SimpleNamespace

import numpy as np
import pyquaternion as pq
from scipy import interpolate

import settings


def main():
    """DOCSTRING HERE"""
    bf = build_butterfly()


def build_butterfly():
    bf = Butterfly(settings.BODY_FRAME_UNIT_VECS)
    head = BodySphere(settings.HEAD_MASS,
                      settings.HEAD_DIAMETER)
    thorax = BodyEllipse(settings.THORAX_MASS,
                         settings.THORAX_LENGTH,
                         settings.THORAX_DIAMETER,
                         settings.THORAX_ELEMENTS)
    abdomen = BodyEllipse(settings.ABDOMEN_MASS,
                          settings.ABDOMEN_LENGTH,
                          settings.ABDOMEN_DIAMETER,
                          settings.ABDOMEN_ELEMENTS)
    wing = Wing(settings.WING_MASS,
                settings.WING_PTS,
                settings.WING_AXES,
                settings.WING_ELEMENTS)

    bf.add_body_part('head',
                     head,
                     settings.HEAD_ROOT,
                     settings.HEAD_UNIT_VECS)
    bf.add_body_part('thorax',
                     thorax,
                     settings.THORAX_ROOT,
                     settings.THORAX_UNIT_VECS)
    bf.add_body_part('abdomen',
                     abdomen,
                     settings.ABDOMEN_ROOT,
                     settings.ABDOMEN_UNIT_VECS)
    bf.add_body_part('wing',
                     wing,
                     settings.WING_ROOT,
                     settings.WING_UNIT_VECS)

    bf.body.wing.flap = Sinusoid(settings.FLA_AMP,
                                 settings.FLA_MEAN,
                                 settings.FLA_PHA,
                                 settings.FREQUENCY)
    bf.body.wing.sweep = Sinusoid(settings.SWE_AMP,
                                  settings.SWE_MEAN,
                                  settings.SWE_PHA,
                                  settings.FREQUENCY)
    bf.body.wing.feath = Sinusoid(settings.FEA_AMP,
                                  settings.FEA_MEAN,
                                  settings.FEA_PHA,
                                  settings.FREQUENCY)

    return bf


def create_3d_points(x_pts, y_pts, z_pts=None):
    """Given arrays of x, y, and (optionally) z coordinates, return a
    list of arrays as 3D coordinates."""
    if z_pts is None:
        z_pts = np.zeros(len(x_pts))

    return list(np.stack((x_pts, y_pts, z_pts), axis=1))


def quat_rotate(q, v, inverse=False):
    """Rotate vector 'v' by quaternion 'q'.  If inverse=True, Rotate by the
    quaternion inverse(opposite rotation)."""
    if inverse:
        return q.inverse.rotate(v)
    else:
        return q.rotate(v)


def rotate_vectors(q, vec):
    """Rotate 3D vectors by quaternion 'q'.  'vec' should be an N-by-3 numpy
    array, where N is the number of vectors to rotate.  Returns an array of
    the same size as 'vec', with each vector rotated by 'q'
    """
    rot_vec = []
    for i, v in enumerate(vec):
        rot_vec.append(q.rotate(v))
    return rot_vec


# OBSOLETE - np.cross already can do element wise cross product, REMOVE LATER
# def get_cross(w, x_list):
#     """Get tangential velocity/acceleration for a list of position
#     vectors via the cross product w (cross) x"""
#     v_list = []
#     for i, x in enumerate(x_list):
#         v_list.append(np.cross(w, x))
#     return v_list


class Butterfly(object):
    def __init__(self, body_unit_vecs):
        self.unit_vecs = body_unit_vecs
        self.body = SimpleNamespace()  # Create empty object
        self.root = SimpleNamespace()
        self.quat = SimpleNamespace()

    def add_body_part(self, name, body_part, root, part_unit_vecs):
        """Adds body part at specified location"""
        setattr(self.body, name, body_part)
        setattr(self.root, name, root)
        setattr(self.quat, name, self.get_quaternion(self.unit_vecs,
                                                     part_unit_vecs))

    def get_com(self, t):
        """DOCSTRING"""

    @staticmethod
    def get_quaternion(lst1, lst2, matchlist=None):
        """Find quaternion between two coordinate systems EXPAND HERE

        Taken from here(cleaned up a bit):
        https://stackoverflow.com/a/23760608/8774464

        Which is a Python implementation from an algorithm in:
        Paul J. Besl and Neil D. McKay "Method for registration of 3-D
        shapes", Sensor Fusion IV: Control Paradigms and Data Structures,
        586 (April 30, 1992); http://dx.doi.org/10.1117/12.57955
        """
        if not matchlist:
            matchlist = range(len(lst1))
        m = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        for i, coord1 in enumerate(lst1):
            x = np.matrix(np.outer(coord1, lst2[matchlist[i]]))
            m = m + x

        n11 = float(m[0][:, 0] + m[1][:, 1] + m[2][:, 2])
        n22 = float(m[0][:, 0] - m[1][:, 1] - m[2][:, 2])
        n33 = float(-m[0][:, 0] + m[1][:, 1] - m[2][:, 2])
        n44 = float(-m[0][:, 0] - m[1][:, 1] + m[2][:, 2])
        n12 = float(m[1][:, 2] - m[2][:, 1])
        n13 = float(m[2][:, 0] - m[0][:, 2])
        n14 = float(m[0][:, 1] - m[1][:, 0])
        n23 = float(m[0][:, 1] + m[1][:, 0])
        n24 = float(m[2][:, 0] + m[0][:, 2])
        n34 = float(m[1][:, 2] + m[2][:, 1])

        n = np.matrix([[n11, n12, n13, n14],
                       [n12, n22, n23, n24],
                       [n13, n23, n33, n34],
                       [n14, n24, n34, n44]])

        values, vectors = np.linalg.eig(n)
        w = list(values)
        mw = max(w)
        quat = vectors[:, w.index(mw)]
        quat = np.array(quat).reshape(-1, )
        return pq.Quaternion(quat)


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
        x_ele: Spanwise location of element centerpoints.
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

    def __init__(self, mass, wing_pts, wing_axes, n_elements):
        """Creates the initial wing object with input wing coordinates,
        mass and number of elements.  Divides the wing into n elements of
        equal length along the wingspan, then calculates properties of those
        elements needed in the model.

        Args: XXX FINISH THIS XXX
            mass: Mass of the wing in grams(g).
            wing_pts: dictionary containing 'x', 'y_le', and 'y_te' entries,
            which contain arrays of coordinate points defining the leading
            and trailling edge of the wing.
            y_le: y-coordinates of points along the leading edge
            corresponding with 'x'
            y_te: y-coordinates of points along the trailing edge
            corresponding with 'x'
            n_elements: Number of elements to divide the wing into.

        NOTE: Because the wing point coordinate data is in the y = f(x)
        format, the x and y axes in some of the functions are switched from
        what is used in the wing coordinate system(i.e. the x data for the
        wing points are along the span of the wing, which is the y-axis in
        the wing frame, and visa versa).  This is handled in the code but
        should be noted as it is here, to prevent confusion.

        """
        self.mass = mass

        # Generate x-coordinates of the element centers and x-coordinates of
        #  the leading and trailing edges of those elements
        self.x_ele, [self.y_le, self.y_te], self.ele_width = \
            self.interp_elements(wing_pts['x'],
                                 [wing_pts['y_le'],
                                  wing_pts['y_te']],
                                 n_elements)

        # Wing length
        self.wing_length = wing_pts['x']
        self.nondim_x = self.x_ele / self.wing_length

        # Generate chord lengths of each element
        self.chord_length = self.get_chord_length(self.y_le, self.y_te)
        self.nondim_chord = self.chord_length / self.wing_length

        # Mean chord, assumes constant element width
        self.mean_chord = np.mean(self.chord_length)

        # Element area
        self.ele_area = self.ele_width * self.chord_length
        self.wing_area = np.sum(self.ele_area)

        # Find centers of elements then get centroid for center of mass
        self.midchord = self.get_chord_pos(self.y_le, self.y_te, 0.5)
        self.centroid = self.find_centroid(np.stack((self.x_ele,
                                                     self.midchord)),
                                           self.chord_length)

        # Generate y-coordinates for the 25% chordline and 75% chordline
        self.chord25 = self.get_chord_pos(self.y_le, self.y_te, 0.25)
        self.chord75 = self.get_chord_pos(self.y_le, self.y_te, 0.75)

        # Generate moment of inertia tensor of the entire wing
        self.moi = self.get_moi(self.x_ele, self.midchord, self.ele_width,
                                self.chord_length, mass)

        # Convert x and y data in N-by-3 arrays in the wing frame
        self.wingtip = create_3d_points(wing_pts['x'][-1],
                                        wing_pts['y_le'][-1])
        self.leading_edge = create_3d_points(self.x_ele, self.y_le)
        self.trailing_edge = create_3d_points(self.x_ele, self.y_te)
        self.midchord = create_3d_points(self.x_ele, self.midchord)
        self.centroid = create_3d_points([self.centroid[0]],
                                         [self.centroid[1]])
        self.chord25 = create_3d_points(self.x_ele, self.chord25)
        self.chord75 = create_3d_points(self.x_ele, self.chord75)

        self.ax_flap = wing_axes['flap']
        self.ax_sweep = wing_axes['sweep']
        self.ax_feath = wing_axes['feath']

        self.ax_span = wing_axes['span']
        self.ax_chord = wing_axes['chord']
        self.ax_up_surf = wing_axes['up_surf']

        self.flap = None
        self.sweep = None
        self.feath = None

        # 't' class containing values calculated at time step 't'.  Cleared at
        # the end of each time step.
        self.t = SimpleNamespace()

    def time_depend(self, t, rho, v_frame, w_d1_frame, w_d2_frame):
        """EXPAND

        Aggregates information about the wing at time 't'

        """

        # Get rotation quaternion and rotational velocity/acceleration vectors
        #  at time 't'
        self.t.q = self.get_q_total(t)
        self.t.rot_vel, self.t.rot_acc = self.get_rot_vecs(t)

        # Rotate wing axes
        self.t.ax_span = rotate_vectors(self.t.q, self.ax_span)
        self.t.ax_chord = rotate_vectors(self.t.q, self.ax_chord)
        self.t.ax_up_surf = rotate_vectors(self.t.q, self.ax_up_surf)

        # Rotate wing points
        self.t.midchord = rotate_vectors(self.t.q_tot, self.midchord)
        self.t.chord25 = rotate_vectors(self.t.q_tot, self.chord25)
        self.t.chord75 = rotate_vectors(self.t.q_tot, self.chord75)
        self.t.wingtip = rotate_vectors(self.t.q_tot, self.wingtip)

        # Get wing element velocities, accounting for velocity and rotation
        # of the frame
        self.t.v_ele = np.cross((self.t.rot_vel + w_d1_frame),
                                self.t.midchord) + v_frame
        self.t.v_ele = np.cross((self.t.rot_vel + w_d1_frame),
                                self.t.wingtip) + v_frame

        # Get projected velocity in chord plane
        self.t.v_proj = self.plane_projection(self.t.v_ele, self.t.ax_span)
        self.t.wt_vel_proj = self.plane_projection(self.t.wingtip_vel,
                                                   self.t.ax_span)

        # Get angles of attack and aerodynamic upper surface normals
        [self.t.aoa, self.t.aero_up_surf] = self.get_aoa(self.t.v_proj,
                                                         self.t.ax_chord,
                                                         self.t.ax_up_surf)

        # Get point on wing element where aero force acts
        self.t.force_loc = self.get_force_loc(self.t.aoa,
                                              self.t.chord25,
                                              self.t.chord75,
                                              settings.TANH_FACTOR)

        # Get magnitudes of aerodynamic forces per element
        self.t.f_trans = self.get_force_trans(rho,
                                              self.t.aoa,
                                              self.t.v_proj)
        self.t.f_rot = self.get_force_rot(rho, t)
        self.t.f_add = self.get_force_added(rho, t,
                                            self.t.rot_vel, self.t.rot_acc)

        # Get force vectors by summing magnitudes of the aerodynamic forces
        # and multiplying by the unit vector of the upper surface normal(the
        #  direction the force is assumed to act)
        self.t.f_mag = self.t.f_trans + self.t.f_rot + self.t.f_add
        self.t.f_vec = np.multiply(self.t.f_mag, self.t.force_loc)

        # Get moments on elements by crossing force vectors with the
        # position vectors of where the forces act
        self.t.moments = np.cross(self.t.force_loc, self.t.f_vec)

        # Sum element forces and moments to get total force and moment on
        # the wing
        self.t.f_total = np.sum(self.t.f_vec)
        self.t.m_total = np.sum(self.t.moments)

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

    def get_q_total(self, t):
        """Returns resultant quaternion of 3 rotation quaternions.  First gets
        the resultant quaternion of q_swe*q_fla and rotates the wing axis by
        that quaternion.  Then gets q_fea, which is a rotation about the wing
        axis(which is why q_swe*q_fla needs to be calculated first).  The gets
        resultant quaternion q_fea*q_swe*q_fla.
        """
        q_flap = pq.Quaternion(axis=self.ax_flap, angle=self.flap.a(t))
        q_sweep = pq.Quaternion(axis=self.ax_sweep, angle=self.sweep.a(t))
        q_swe_fla = q_sweep.__mul__(q_flap)
        ax_feath = q_swe_fla.rotate(self.ax_span)
        q_fea = pq.Quaternion(axis=ax_feath, angle=self.feath.a(t))

        return q_fea.__mul__(q_swe_fla)

    def get_rot_vecs(self, t):
        """Calculates total rotational velocity and acceleration of the wing
        at time 't'.  Determines components from the motions of the wings in
        the wing frame, as well as the rotational velocity/acceleration of
        the wing frame itself"""
        rot_vel = (self.ax_flap * self.flap.da_dt(t) +
                   self.ax_sweep * self.sweep.da_dt(t) +
                   self.t.ax_span * self.feath.da_dt(t))

        rot_acc = (self.ax_flap * self.flap.d2a_dt2(t) +
                   self.ax_sweep * self.sweep.d2a_dt2(t) +
                   self.t.ax_span * self.feath.d2a_dt2(t))
        return rot_vel, rot_acc

    @staticmethod
    def plane_projection(vectors, plane_norm):
        """DOCSTRING"""
        proj = []
        for vec in vectors:
            proj.append(vec -
                        (np.multiply(np.dot(vec, plane_norm),
                                     plane_norm)))
        return proj

    # def get_wing_force(self, t, v_frame, w_vel_frame, w_acc_frame):
    #     """DOCSTRING"""
    #
    #     # get angle of attack
    #
    #
    #     # get force magnitude
    #
    #     # get force vectors
    #
    #     # sum force vectors
    #     return

    @staticmethod
    def get_aoa(v_proj, chord_axis, up_surf_norm):
        """Get angle of attack for each wing element.  Also determine the
        aerodynamic upper surface based on relative velocity vector(
        aerodynamic upper surface can be different than morphological upper
        surface if the AoA is 'negative'.
        """

        # get angle of attack and determine direction of aerodynamic upper
        # surface (EXPAND HERE TO EXPLAIN)
        aoa = []
        aero_up_surf = []
        for vel in v_proj:
            aoa.append(np.dot(np.linalg.norm(vel), chord_axis))
            flip = -np.sign(np.dot(np.linalg.norm(vel), up_surf_norm))
            aero_up_surf.append(np.multiply(flip, up_surf_norm))
        return [aoa, aero_up_surf]

    @staticmethod
    def get_force_loc(aoa, chord25, chord75, tanh_scale):
        """DOCSTRING

        As the location of the force vectors transitions for the 25%
        chordline to the 75% chordline as the angle of attack crosses 90deg, a
        hyperbolic tangent function is used to transition between those two
        locations smoothly.

        """

        a = 0.5 * (1 - np.tanh(tanh_scale * (aoa - np.pi / 2)))
        force_loc = a * chord25 + (1 - a) * chord75
        return force_loc

    def get_force_trans(self, rho, aoa, v_proj):
        """DOCSTRING"""

        exp_factor = settings.EXP_FACTOR
        cl = (1.5 * np.sin(2 * aoa - 0.06) +
              0.3 * np.cos(aoa - 0.485) +
              0.012)
        cd = (1.4 * np.sin(2 * aoa - 1.4) +
              0.3 * np.cos(aoa - 1.328) +
              1.5)
        c_t = np.linalg.norm([cl, cd], axis=0)
        c_t = c_t * (1 - np.exp(- exp_factor * aoa))

        u_2 = np.square(np.linalg.norm(v_proj, axis=1))
        x_hat = self.nondim_x
        c_hat = self.nondim_chord
        r = self.wing_length
        c_bar = self.mean_chord
        dr = self.ele_width

        f_trans = ((0.5 * rho * r * c_bar * dr) *
                   np.sum(reduce(np.multiply, [c_t, u_2,
                                               x_hat, c_hat])))
        return f_trans

    def get_force_rot(self, rho, t):
        """DOCSTRING"""

        c_rot = settings.ROT_FORCE_COEFF
        v_wt = self.t.wt_vel_proj
        a_d1 = self.feath.da_dt(t)
        c_bar = self.mean_chord
        r = self.wing_length
        dr = self.ele_width
        x_hat = self.nondim_x
        c_hat = self.nondim_chord

        f_rot = (c_rot * rho * v_wt * a_d1 *
                 (c_bar ** 2) * r * dr *
                 np.sum(reduce(np.multiply, [x_hat, c_hat, c_hat])))
        return f_rot

    def get_force_added(self, rho, t, wing_rot_vel, wing_rot_acc):
        """DOCSTRING"""

        c_bar = self.mean_chord
        r = self.wing_length
        dr = self.ele_width
        x_hat = self.nondim_x
        c_hat = self.nondim_chord
        phi_d1 = np.linalg.norm(wing_rot_vel)
        phi_d2 = np.linalg.norm(wing_rot_acc)
        a = self.feath.a(t)
        a_d1 = self.feath.da_dt(t)
        a_d2 = self.feath.d2a_dt2(t)

        f_add = ((0.25 * rho * (c_bar ** 2) * r * dr) *
                 (r * (phi_d2 * np.sin(a) + phi_d1 * a_d1 * np.cos(a)) *
                  np.sum(reduce(np.multiply, [x_hat, c_hat, c_hat]))) +
                 (0.25 * a_d2 * c_bar *
                  np.sum(reduce(np.multiply, [c_hat, c_hat]))))
        return f_add

    def clear_time_variants(self):
        """"""
        pass


class Sinusoid(object):
    """Define a sinusoid with an amplitude, mean, phase,
    and frequency.  Has methods evaluate the function at a given time 't',
    as well as the first and second derivatives of the function
    """

    def __init__(self, amp, mean, phase, freq):
        """Define parameters of sinusoid.

        :param amp: Amplitude
        :param mean: Mean
        :param phase: Phase(radians)
        :param freq: Frequency(Hz)
        """
        self.amp = amp
        self.mean = mean
        self.phase = phase
        self.freq = freq
        self.omega = 2 * np.pi * freq

    def a(self, t):
        """Evaluates the function at time 't'"""
        a = self.amp * np.sin(self.omega * t + self.phase) + self.mean
        return a

    def da_dt(self, t):
        """Evaluates the first derivative of the function at time 't'"""
        da_dt = self.amp * self.omega * np.cos(self.omega * t + self.phase)
        return da_dt

    def d2a_dt2(self, t):
        """Evaluates the second derivative of the function at time 't'"""
        d2a_dt2 = -self.amp * self.omega ** 2 * np.sin(self.omega * t +
                                                       self.phase)
        return d2a_dt2


if __name__ == '__main__':
    main()
