"""Collection of nonlinear solvers based on the shooting method."""

# TODO:
# - think of a better handling of forces arguments
# - more robust y and x deps

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from nlsys import NLSystem, TimeDivision


class ShootingSolution(NamedTuple):
    """Solution of a BVP via the shooting method.
    y0  -- IC solution of the BVP, for the desired tdiv.
    max -- Corresponding DOFs maximum displacement.
    min -- Corresponding DOFs maximum displacement.
    """
    tdiv: TimeDivision
    y0:   np.ndarray
    max:  np.ndarray
    min:  np.ndarray


def shooting(sys: NLSystem, y0_guess, tdiv: TimeDivision) -> ShootingSolution:
    """Shooting method to solve the BVP."""

    def objective(y0):
        """Function to be minimized: difference between y(0) and y(T)."""
        sol = solve_ivp(sys.integrand, [0, tdiv.T], y0, t_eval=[tdiv.T], args=(tdiv.w,))
        yT = sol.y[:, -1]
        return (yT - y0)

    sol = root(objective, y0_guess)
    print(f"shooting success: {sol.success}, nfev: {sol.nfev}, freq: {tdiv.f:.4f}Hz")
    y0 = sol.x

    # Find the extremum displacements of each DOFs
    y = solve_ivp(
        sys.integrand,
        [0, tdiv.T],
        y0,
        args=(tdiv.w,),
        t_eval=np.linspace(0, tdiv.T, 300)).y
    max_dof1 = np.max(y[0, :])
    max_dof2 = np.max(y[1, :])
    max = [max_dof1, max_dof2]
    min_dof1 = np.min(y[0, :])
    min_dof2 = np.min(y[1, :])
    min = [min_dof1, min_dof2]

    return ShootingSolution(tdiv=tdiv, y0=y0, max=max, min=min)


class ContinuationSolution(NamedTuple):
    """Solution of a sequential continuation computation.
    y0_range  -- IC solutions of the BVP, for the desired tdiv_range.
    max_range -- Corresponding DOFs maximum displacement.
    min_range -- Corresponding DOFs minimum displacement.
    """
    tdiv_range: np.ndarray
    y0_range:   np.ndarray
    max_range:  np.ndarray
    min_range:  np.ndarray


def basic_continuation(sys: NLSystem, y0_guess, tdiv_range) -> ContinuationSolution:
    """Basic sequential continuation, without prediction and correction."""

    y0_range  = np.zeros((sys.ndof_ss, tdiv_range.size))
    max_range = np.zeros((sys.ndof,    tdiv_range.size))
    min_range = np.zeros((sys.ndof,    tdiv_range.size))

    for (idx, tdiv) in enumerate(tdiv_range):
        sol = shooting(sys, y0_guess, tdiv)
        y0_range[:, idx]  = sol.y0
        max_range[:, idx] = sol.max
        min_range[:, idx] = sol.min

        # Basic continuation: the prediction of y0 for the next frequency
        # is simply the current solution.
        y0_guess = sol.y0

    return ContinuationSolution(
        tdiv_range = tdiv_range,
        y0_range   = y0_range,
        max_range  = max_range,
        min_range  = min_range
    )


def secant_continuation(sys: NLSystem, y0_guess, tdiv_range) -> ContinuationSolution:
    """Sequential continuation, with secant prediction."""

    y0_range  = np.zeros((sys.ndof_ss, tdiv_range.size))
    max_range = np.zeros((sys.ndof,    tdiv_range.size))
    min_range = np.zeros((sys.ndof,    tdiv_range.size))

    sol_pprev = shooting(sys, y0_guess, tdiv_range[0])
    y0_range[:, 0]  = y0_pprev = sol_pprev.y0
    max_range[:, 0] = sol_pprev.max
    min_range[:, 0] = sol_pprev.min

    sol_prev = shooting(sys, y0_pprev, tdiv_range[0])
    y0_range[:, 1]  = y0_prev = sol_prev.y0
    min_range[:, 1] = sol_prev.min

    for (shifted_idx, tdiv) in enumerate(tdiv_range[2:]):
        idx = shifted_idx + 2  # TODO: not elegant

        # Secant continuation: the prediction of y0 for the next frequency
        # is the linear extrapolation of the two previous solutions.
        # WARN: assume a linear spacing between the tdiv
        y0_guess = 2*y0_prev - y0_pprev

        sol = shooting(sys, y0_guess, tdiv)
        y0_range[:, idx]  = sol.y0
        max_range[:, idx] = sol.max
        min_range[:, idx] = sol.min

        y0_pprev = y0_prev
        y0_prev  = sol.y0

    return ContinuationSolution(
        tdiv_range = tdiv_range,
        y0_range   = y0_range,
        max_range  = max_range,
        min_range  = min_range
    )


def plot_BVP(sys: NLSystem, y0_guess, bspan: TimeDivision):
    """Plot the BVP shooting solution."""

    # Solve the BVP throught the shooting method
    sol_shooting = shooting(sys, y0_guess, bspan)
    print(f"IC solution of the BVP: {sol_shooting.y0}")
    print(f"DOF maximas: {sol_shooting.max}")
    print(f"DOF minimas: {sol_shooting.min}")

    # Verify that the BVP has been solved correctly
    sol = solve_ivp(
        sys.integrand,
        [0, bspan.T],
        sol_shooting.y0,
        args=(bspan.w,),
        t_eval=np.linspace(0, bspan.T, 300))
    y = sol.y
    t_sample = sol.t

    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    ax.plot(t_sample, y[:2, :].T)
    ax.legend(['x1', 'x2'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('State vector')
    ax.set_title(f"BVP solution (f = {sol_shooting.tdiv.f} Hz)")
    ax.grid()
    fig.show()

    return sol_shooting
