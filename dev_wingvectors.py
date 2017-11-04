import matplotlib.pyplot as plt
import numpy as np

# scl = 10
# lim = np.pi / 8
#
# a = np.linspace(-lim + np.pi / 2, lim + np.pi / 2, 100)
#
# b = 0.5 * (1 - np.tanh(scl * (a - np.pi / 2)))
#
# fig = plt.figure()
# ax = fig.gca()
# ax.set_xticks(np.arange(0, np.pi, np.pi / 8))
# plt.plot(a, b)
# plt.grid()
# plt.show()

exp_factor = 40

a = np.linspace(0, np.pi / 2, 100)

cl = 1.5 * np.sin(2 * a - 0.06) + 0.3 * np.cos(a - 0.485) + 0.012
cd = 1.4 * np.sin(2 * a - 1.4) + 0.3 * np.cos(a - 1.328) + 1.5
ct = np.linalg.norm([cl, cd], axis=0)
ct_mod = ct * (1-np.exp(- exp_factor * a))

fig = plt.figure()
plt.subplot(131)
plt.plot(a,cl)
plt.grid()
plt.subplot(132)
plt.plot(a,cd)
plt.grid()
plt.subplot(133)
plt.plot(a,ct)
plt.plot(a,ct_mod, 'kv')
plt.grid()
plt.show()