# TODO:
# - must have an init method for TimeDivision, otherwise it is painful to set
#   an array of TimeDivision (see the test_basic_continuation function).

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from utils.tdiv import TimeDivision
import shooting

# plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
# plt.rcParams['figure.figsize'] = (6.34, 3.34)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 200


def test_system():
    """Build the system given by the statement."""

    # Linear system matrices
    M = np.eye(2)
    C = np.array([[3, -1], [-1, 3]])
    K = 1E4 * np.array([[2, -1], [-1, 2]])

    # Nonlinearities
    f_nl = lambda x, x_dot: np.array([5E4*x[0]**3, 0])

    # Build the unforced NL system
    sys = shooting.NLSystem(M, C, K, f_nl)

    # Add an external excitation
    sys.add_harmonic_excitation(amplitude=100)

    # Build the state-space repr of the NL system
    sys.build_state_space()

    return sys


def test_shooting(sys):
    """Shooting for a particular excitation frequency."""

    # Choose an excitation frequency
    tdiv = TimeDivision()
    tdiv.f = 18

    # Make an initial guess on the IC state
    y0_guess = 1E-6 * np.array([1, 1, 1, 1])

    # Solve the BVP throught the shooting method
    sol_shooting = shooting.shooting(sys, y0_guess, tdiv)

    # Verify that the BVP has been solved correctly
    y = solve_ivp(sys.integrand, [0, tdiv.T], sol_shooting.y0, args=(tdiv.w,), dense_output=True).sol
    t_sample = np.linspace(0, tdiv.T, 300)
    plt.plot(t_sample, y(t_sample).T)
    plt.xlabel('time (s)')
    plt.ylabel('State vector')
    plt.legend(['x1', 'x2', 'x1_dot', 'x2_dot'])
    plt.title(f"BVP solution (f = {sol_shooting.tdiv.f} Hz)")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return sol_shooting


def test_continuation(sys, continuation=shooting.basic_continuation):
    """Sequential continuation through a frequency range."""

    # NOTE::
    # building a quick model of the linear system in NI2D show that the
    # natural frequencies are about 16Hz and 27.6Hz.
    # TODO: think of a way to compute solutions in parallel
    f_ranges = [
        np.linspace(0.1,  17,   100),   # From 0Hz to the first natural frequency
        np.linspace(17,   21,   2010),  # Refine around the fundamental peak of the first freq
        np.linspace(16.4, 27.4, 100),   # Restart from low amplitude, between the two fundamental peaks
        np.linspace(27.4, 30,   100),   # Refine around the fundamental peak of the second freq
        np.linspace(27.8, 35,   100),   # Restart from low amplitude, after the second fundamental peak
    ]
    # Try to detail the low freq range.
    # f_ranges = [np.linspace(0.1, 17/4, 1000)]
    # Try to spot superharmonic of first freq
    # f_ranges = [
    #     np.linspace(0.1, 5, 100),
    #     np.linspace(5, 6.5, 500),
    #     np.linspace(5.8, 8, 100),
    # ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    #IC state guess
    y0_guess = 1E-6 * np.array([1, 1, 1, 1])

    # Sequential continuation over the specified frequency range
    solutions = [0 for _ in f_ranges]
    solutions[0] = continuation(sys, y0_guess,                     tdiv_ranges[0])
    solutions[1] = continuation(sys, solutions[0].y0_range[:, -1], tdiv_ranges[1])
    solutions[2] = continuation(sys, y0_guess,                     tdiv_ranges[2])
    solutions[3] = continuation(sys, solutions[2].y0_range[:, -1], tdiv_ranges[3])
    solutions[4] = continuation(sys, y0_guess,                     tdiv_ranges[4])

    # Plot the NFRC of DOF x1
    for solution in solutions:
        plt.plot([sol.f for sol in solution.tdiv_range], solution.max_range[0, :])
    plt.xlabel('Excitation frequency (Hz)')
    plt.ylabel('DOF amplitude (m)')
    plt.title("Nonlinear frequency response curve")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return solutions


if __name__ == '__main__':
    sys = test_system()
    test_shooting(sys)
    # solutions = test_continuation(sys)
