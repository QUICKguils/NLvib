"""shooting -- Collection of nonlinear solvers based on the shooting method."""

# TODO:
# - think of a better handling of forces arguments
# - better docstring, clean up obsolete comments

from typing import NamedTuple

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize_scalar

from utils.tdiv import TimeDivision


class NLSystem:
    """NLSystem -- Implement a nonlinear system."""

    def __init__(self, M, C, K, f_nl):
        """Generic equations of motion of a free nonlinear system.

        M*x_ddot(t) + C*x_dot(t) + K*x(t) + f_nl(x, x_dot) = 0
        """
        self.ndof = M.shape[0]

        self.M = M
        self.C = C
        self.K = K
        self.f_nl = f_nl

    def add_harmonic_excitation(self, amplitude=1):
        """Add an external harmonic excitation to the NL system.

        M*x_ddot(t) + C*x_dot(t) + K*x(t) + f_nl(x, x_dot) = f_ext(w, t)
        """
        # WARN: only add to node 1
        self.f_ext = lambda w, t: np.array([amplitude*np.sin(w*t), 0])

    def build_state_space(self):

        """Recast in first order state-space form:

        y_dot(t) = L*y(t) - g_nl(y) + g_ext(w, t) , with y = [x, x_dot]
        """
        identity = np.eye(self.ndof)
        null = np.zeros((self.ndof, self.ndof))
        M_inv = linalg.inv(self.M)
        self.ndof_ss = 2*self.ndof

        self.L = np.vstack((np.hstack((null, identity)), np.hstack((-M_inv@self.K, -M_inv@self.C))))
        self.g_nl = lambda y: np.concatenate((np.zeros(self.ndof), M_inv@self.f_nl(y[:len(y)//2], y[len(y)//2:])))
        self.g_ext = lambda w, t: np.concatenate((np.zeros(self.ndof), M_inv@self.f_ext(w, t)))

        self.integrand = lambda t, y, w: self.L@y - self.g_nl(y) + self.g_ext(w, t)


def shooting(sys: NLSystem, y0_guess, tdiv: TimeDivision):
    """Shooting method to solve the BVP."""

    def objective(y0):
        """Function to be minimized: difference between y(0) and y(T)."""
        sol = solve_ivp(sys.integrand, [0, tdiv.T], y0, t_eval=[tdiv.T], args=(tdiv.w,))
        yT = sol.y[:, 0]
        return yT - y0

    return root(objective, y0_guess).x


def basic_continuation(sys: NLSystem, y0_guess, tdiv_range):
    """Basic sequential continuation, without prediction and correction."""

    class Solution(NamedTuple):
        """Solution of the basic continuation.
        y0_range  -- IC solutions of the BVP, for the desired w_range.
        max_range -- Corresponding DOFs maximum displacement.
        """
        tdiv_range: np.ndarray
        y0_range: np.ndarray
        max_range: np.ndarray

    y0_range = np.zeros((sys.ndof_ss, tdiv_range.size))
    max_range = np.zeros((sys.ndof, tdiv_range.size))

    for (idx, tdiv) in enumerate(tdiv_range):
        # BVP solution for the excitation frequency w
        y0 = shooting(sys, y0_guess, tdiv)
        y0_range[:, idx] = y0

        # Amplitude maxima
        y = solve_ivp(sys.integrand, [0, tdiv.T], y0, args=(tdiv.w,), dense_output=True).sol
        min_dof1 = minimize_scalar(lambda t: y(t)[0], bounds=(0, tdiv.T))
        min_dof2 = minimize_scalar(lambda t: y(t)[1], bounds=(0, tdiv.T))
        max_range[:, idx] = [-min_dof1.fun, -min_dof2.fun]

        # Update guess for next frequency
        y0_guess = y0

    return Solution(tdiv_range=tdiv_range, y0_range=y0_range, max_range=max_range)


def advanced_continuation(sys: NLSystem, y0_guess, w_range):
    """Sequential continuation, with prediction and correction."""
    # TODO: if time, obtain nice spikes
