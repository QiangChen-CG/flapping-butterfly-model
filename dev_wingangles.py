import numpy as np
import matplotlib.pyplot as plt

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

    def a_t(self, t):
        """Evaluates the function at time 't'"""
        a_t = self.amp * np.sin(self.omega * t + self.phase) + self.mean
        return a_t

    def da_dt(self, t):
        """Evaluates the first derivative of the function at time 't'"""
        da_dt = self.amp * self.omega * np.cos(self.omega * t + self.phase)
        return da_dt

    def d2a_dt2(self, t):
        """Evaluates the second derivative of the function at time 't'"""
        d2a_dt2 = -self.amp * self.omega ** 2 * np.sin(self.omega * t +
                                                       self.phase)
        return d2a_dt2


fea = Sinusoid(.5, .4, .4, 10)

t = np.linspace(0, 1, 1000)
fea_t = np.empty(np.shape(t))
for i, t in enumerate(fea_t):
    fea_t[i] = fea.a_t(t[i])

plt.plot(t, fea_t)
plt.show()