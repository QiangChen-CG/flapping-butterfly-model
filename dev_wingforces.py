"""DOCSTRING"""
import buildbutterfly as bb
import numpy as np
import pyquaternion as pq
import settings

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

# rotation quaternions
"""First rotate wing axis by flapping and sweeping angle to find wing 
axis, which is needed as the feathering angle is rotated about that wing 
axis"""

q_fla = pq.Quaternion(axis=fla_axis, angle=fla.a(t))
q_swe = pq.Quaternion(axis=swe_axis, angle=swe.a(t))
#
# wing_axis_t = bb.quat_rotate(q_swe,
#                              bb.quat_rotate(q_fla, wing_axis_init))
#
# q_fla_swe = q_fla.__mul__(q_swe)
# q_swe_fla = q_swe.__mul__(q_fla)
# print(wing_axis_t)
# print(q_fla_swe.rotate(wing_axis_init))  # not same!
# print(q_swe_fla.rotate(wing_axis_init))  # same as wing_axis_t


# q2*q1 = q2.__mul__(q1)


def get_q_total(q_fla, q_swe, theta_fea, wing_axis):
    """Returns resultant quaternion of 3 rotation quaternions.  First gets
    the resultant quaternion of q_swe*q_fla and rotates the wing axis by
    that quaternion.  Then gets q_fea, which is a rotation about the wing
    axis(which is why q_swe*q_fla needs to be calculated first).  The gets
    resultant quaternion q_fea*q_swe*q_fla.
    """
    q_swe_fla = q_swe.__mul__(q_fla)
    wa_rot = q_swe_fla.rotate(wing_axis)
    q_fea = pq.Quaternion(axis=wa_rot, angle=theta_fea)
    return q_fea.__mul__(q_swe_fla)


def rotate_vectors(q, vec):
    """Rotate 3D vectors by quaternion 'q'.  'vec' should be an N-by-3 numpy
    array, where N is the number of vectors to rotate.  Returns an array of
    the same size as 'vec', with each vector rotated by 'q'
    """
    rot_vec = np.empty(np.shape(vec))
    for i in range(vec.shape[0]):
        rot_vec[i] = q.rotate(vec[i])
    return rot_vec

q_tot = get_q_total(q_fla, q_swe, np.pi / 4, wing_axis_init)

rot_mid = rotate_vectors(q_tot, bf.body.wing.midchord)

print(q_tot)
print(rot_mid)
print(type(rot_mid.tolist()[0]))