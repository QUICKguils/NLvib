"""shooting -- Collection of nonlinear solvers based on the shooting method."""

# TODO:
# - think of a better handling of forces arguments
# - more robust y and x deps
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
        self.g_nl  = lambda y:    np.concatenate((np.zeros(self.ndof), M_inv@self.f_nl(y[:len(y)//2], y[len(y)//2:])))
        self.g_ext = lambda w, t: np.concatenate((np.zeros(self.ndof), M_inv@self.f_ext(w, t)))

        self.integrand = lambda t, y, w: self.L@y - self.g_nl(y) + self.g_ext(w, t)


class ShootingSolution(NamedTuple):
    """Solution of a BVP via the shooting method.
    y0  -- IC solution of the BVP, for the desired tdiv.
    max -- Corresponding DOFs maximum displacement.
    """
    tdiv: TimeDivision
    y0:   np.ndarray
    max:  np.ndarray


def shooting(sys: NLSystem, y0_guess, tdiv: TimeDivision) -> ShootingSolution:
    """Shooting method to solve the BVP."""

    def objective(y0):
        """Function to be minimized: difference between y(0) and y(T)."""
        sol = solve_ivp(sys.integrand, [0, tdiv.T], y0, t_eval=[tdiv.T], args=(tdiv.w,))
        yT = sol.y[:, -1]
        return yT - y0

    y0 = root(objective, y0_guess).x

    y = solve_ivp(sys.integrand, [0, tdiv.T], y0, args=(tdiv.w,), dense_output=True).sol
    min_dof1 = minimize_scalar(lambda t: y(t)[0], bounds=(0, tdiv.T))
    min_dof2 = minimize_scalar(lambda t: y(t)[1], bounds=(0, tdiv.T))
    max = [-min_dof1.fun, -min_dof2.fun]

    return ShootingSolution(tdiv=tdiv, y0=y0, max=max)


class ContinuationSolution(NamedTuple):
    """Solution of a sequential continuation computation.
    y0_range  -- IC solutions of the BVP, for the desired tdiv_range.
    max_range -- Corresponding DOFs maximum displacement.
    """
    tdiv_range: np.ndarray
    y0_range:   np.ndarray
    max_range:  np.ndarray


def basic_continuation(sys: NLSystem, y0_guess, tdiv_range) -> ContinuationSolution:
    """Basic sequential continuation, without prediction and correction."""

    y0_range  = np.zeros((sys.ndof_ss, tdiv_range.size))
    max_range = np.zeros((sys.ndof,    tdiv_range.size))

    for (idx, tdiv) in enumerate(tdiv_range):
        sol = shooting(sys, y0_guess, tdiv)
        y0_range[:, idx]  = sol.y0
        max_range[:, idx] = sol.max

        # Basic continuation: the prediction of y0 for the next frequency
        # is simply the current solution.
        y0_guess = sol.y0

    return ContinuationSolution(
        tdiv_range = tdiv_range,
        y0_range   = y0_range,
        max_range  = max_range
    )


def secant_continuation(sys: NLSystem, y0_guess, tdiv_range) -> ContinuationSolution:
    """Sequential continuation, with secant prediction."""

    y0_range  = np.zeros((sys.ndof_ss, tdiv_range.size))
    max_range = np.zeros((sys.ndof,    tdiv_range.size))

    sol_pprev = shooting(sys, y0_guess, tdiv_range[0])
    y0_range[:, 0]  = y0_pprev = sol_pprev.y0
    max_range[:, 0] = sol_pprev.max

    sol_prev = shooting(sys, y0_pprev, tdiv_range[0])
    y0_range[:, 1]  = y0_prev = sol_prev.y0
    max_range[:, 1] = sol_prev.max

    for (shifted_idx, tdiv) in enumerate(tdiv_range[2:]):
        idx = shifted_idx + 2  # TODO: not elegant

        # Secant continuation: the prediction of y0 for the next frequency
        # is the linear extrapolation of the two previous solutions.
        # WARN: assume a linear spacing between the tdiv
        y0_guess = 2*y0_prev - y0_pprev

        sol = shooting(sys, y0_guess, tdiv)
        y0_range[:, idx]  = sol.y0
        max_range[:, idx] = sol.max

        y0_pprev = y0_prev
        y0_prev  = sol.y0

    return ContinuationSolution(
        tdiv_range = tdiv_range,
        y0_range   = y0_range,
        max_range  = max_range
    )
