"""Definition of the nonlinear system under study."""

import numpy as np
from scipy import linalg

# Linear system matrices
M = np.eye(2)
C = np.array([[3, -1], [-1, 3]])
K = 1E4 * np.array([[2, -1], [-1, 2]])


def f_nl(x, x_dot):
    """Nonlinear force, as identified by the RFS method."""
    return np.array([3.6e5*x[0]**2 + 7.6e6*x[0]**3, 9E6*x[1]**5])


class NLSystem:
    """NLSystem -- Implement a nonlinear system."""

    def __init__(self, M, K, f_nl):
        """Generic equations of motion of an undamped, free nonlinear system.

        M*x_ddot(t) + K*x(t) + f_nl(x, x_dot) = 0
        """
        self.ndof = M.shape[0]

        self.M = M
        self.K = K
        self.f_nl = f_nl

    def add_damping(self, C):
        """Add linear damping to the NL system.

        M*x_ddot(t) + C*x_dot(t) + K*x(t) + f_nl(x, x_dot) = 0
        """
        self.C = C

    def add_harmonic_excitation(self, amplitude=1):
        """Add an external harmonic excitation to the NL system.

        M*x_ddot(t) + C*x_dot(t) + K*x(t) + f_nl(x, x_dot) = f_ext(w, t)
        """
        # WARN: only add to node 1
        def f_ext(w, t):
            return np.array([amplitude*np.sin(w*t), 0])
        self.f_ext = f_ext

    def build_undamped_free_state_space(self):
        """Recast in first order state-space form:

        y_dot(t) = L*y(t) - g_nl(y) , with y = [x, x_dot]
        """
        identity = np.eye(self.ndof)
        null = np.zeros((self.ndof, self.ndof))
        M_inv = linalg.inv(self.M)
        self.ndof_ss = 2*self.ndof

        self.L = np.vstack((np.hstack((null, identity)), np.hstack((-M_inv@self.K, null))))
        def g_nl(y):
            return np.concatenate((np.zeros(self.ndof), M_inv@self.f_nl(y[:len(y)//2], y[len(y)//2:])))
        self.g_nl = g_nl

        def integrand(t, y, w):
            return self.L@y - self.g_nl(y)
        self.integrand = integrand

    def build_damped_forced_state_space(self):
        """Recast in first order state-space form:

        y_dot(t) = L*y(t) - g_nl(y) + g_ext(w, t) , with y = [x, x_dot]
        """
        identity = np.eye(self.ndof)
        null = np.zeros((self.ndof, self.ndof))
        M_inv = linalg.inv(self.M)
        self.ndof_ss = 2*self.ndof

        self.L = np.vstack((np.hstack((null, identity)), np.hstack((-M_inv@self.K, -M_inv@self.C))))
        def g_nl(y):
            return np.concatenate((np.zeros(self.ndof), M_inv@self.f_nl(y[:len(y)//2], y[len(y)//2:])))
        self.g_nl = g_nl
        def g_ext(w, t):
            return np.concatenate((np.zeros(self.ndof), M_inv@self.f_ext(w, t)))
        self.g_ext = g_ext

        def integrand(t, y, w):
            return self.L@y - self.g_nl(y) + self.g_ext(w, t)
        self.integrand = integrand


def build_undamped_free_system(f_nl):
    sys = NLSystem(M, K, f_nl)
    sys.build_undamped_free_state_space()

    return sys


def build_damped_forced_system(f_nl, f_ext_ampl):
    sys = NLSystem(M, K, f_nl)
    sys.add_damping(C)
    sys.add_harmonic_excitation(amplitude=f_ext_ampl)
    sys.build_damped_forced_state_space()

    return sys


class TimeDivision():
    """TimeDivision -- Handling of time division conversions."""

    def __init__(self):
        self._period = None
        self._circular_frequency = None
        self._frequency = None

    @property
    def T(self):
        return self._period
    @T.setter
    def T(self, period):
        self._period = period
        self._circular_frequency = (2*np.pi) / period
        self._frequency = 1 / period

    @property
    def w(self):
        return self._circular_frequency
    @w.setter
    def w(self, circular_frequency):
        self._period = (2*np.pi) / circular_frequency
        self._circular_frequency = circular_frequency
        self._frequency = circular_frequency / (2*np.pi)

    @property
    def f(self):
        return self._frequency
    @f.setter
    def f(self, frequency):
        self._period = 1 / frequency
        self._circular_frequency = 2*np.pi * frequency
        self._frequency = frequency
