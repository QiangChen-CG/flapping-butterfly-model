"""DOCSTRING"""
import buildbutterfly as bb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyquaternion as pq
import settings


def get_q_total(fla_q, swe_q, theta_fea, wing_axis):
    """Returns resultant quaternion of 3 rotation quaternions.  First gets
    the resultant quaternion of q_swe*q_fla and rotates the wing axis by
    that quaternion.  Then gets q_fea, which is a rotation about the wing
    axis(which is why q_swe*q_fla needs to be calculated first).  The gets
    resultant quaternion q_fea*q_swe*q_fla.
    """
    q_swe_fla = swe_q.__mul__(fla_q)
    wa_rot = q_swe_fla.rotate(wing_axis)
    q_fea = pq.Quaternion(axis=wa_rot, angle=theta_fea)
    return q_fea.__mul__(q_swe_fla)


def rotate_vectors(q, vec):
    """Rotate 3D vectors by quaternion 'q'.  'vec' should be an N-by-3 numpy
    array, where N is the number of vectors to rotate.  Returns an array of
    the same size as 'vec', with each vector rotated by 'q'
    """
    rot_vec = []
    for i, v in enumerate(vec):
        rot_vec.append(q.rotate(v))
    return rot_vec


def plot_coords(coord_list, ax_obj):
    coord_array = np.array(coord_list)
    ax_obj.scatter(np.array(coord_array)[:, 0],
                   np.array(coord_array)[:, 1],
                   np.array(coord_array)[:, 2])


bf = bb.build_butterfly()

# velocity, rotation, rotational acceleration at wing root
v_root = [-0.3, -0.5, 0.7]
theta_d1_root = [0, -0.5, 0]
theta_d2_root = [0, 0.29, 0]

wing_axis_init = [1, 0, 0]

# sinusoid functions
fea = bb.Sinusoid(settings.FEA_AMP,
                  settings.FEA_MEAN,
                  settings.FEA_PHA,
                  settings.FREQUENCY)
swe = bb.Sinusoid(settings.SWE_AMP,
                  settings.SWE_MEAN,
                  settings.SWE_PHA,
                  settings.FREQUENCY)
fla = bb.Sinusoid(settings.FEA_AMP,
                  settings.FEA_MEAN,
                  settings.FEA_PHA,
                  settings.FREQUENCY)

fla_axis = np.array([0, 1, 0])
swe_axis = np.array([0, 0, 1])

t = 2  # time

# rotation vector

theta = np.array([0,
                  fea.a(t),
                  swe.a(t)])

theta_d1 = theta_d1_root + np.array([0,
                                     fea.da_dt(t),
                                     swe.da_dt(t)])

theta_d2 = theta_d2_root + np.array([0,
                                     fea.d2a_dt2(t),
                                     swe.d2a_dt2(t)])

# TEST ANGLES
fla_a = np.deg2rad(0)
swe_a = np.deg2rad(45)
fea_a = np.deg2rad(0)

q_fla = pq.Quaternion(axis=fla_axis, angle=fla_a)
q_swe = pq.Quaternion(axis=swe_axis, angle=swe_a)

q_tot = get_q_total(q_fla, q_swe, fea_a, wing_axis_init)

rot_mid = rotate_vectors(q_tot, bf.body.wing.midchord)
rot_le = rotate_vectors(q_tot, bf.body.wing.leading_edge)
rot_te = rotate_vectors(q_tot, bf.body.wing.trailing_edge)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, 'kv')
plot_coords(rot_mid, ax)
plot_coords(rot_le, ax)
plot_coords(rot_te, ax)
plt.show()
