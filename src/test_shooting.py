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
    f_nl = lambda x, x_dot: np.array([1E2*x[0]**3, 0])

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
    tdiv.f = 30

    # Make an initial guess on the IC state
    y0_guess = 1E-4 * np.array([1, 1, 0, 0])

    # Solve the BVP : find the IC state y0
    y0 = shooting.shooting(sys, y0_guess, tdiv)

    # Verify that the BVP has been solved correctly
    y = solve_ivp(sys.integrand, [0, tdiv.T], y0, args=(tdiv.w,), dense_output=True).sol
    t_sample = np.linspace(0, tdiv.T, 300)
    plt.plot(t_sample, y(t_sample)[:2, :].T)
    plt.xlabel('time (s)')
    plt.ylabel('State vector')
    plt.legend(['x1', 'x2', 'x1_dot', 'x2_dot'])
    plt.title(f"BVP solution (f = {tdiv.f} Hz)")
    plt.tight_layout()
    plt.show()

    # Verify that the max displ. of x1 is correct
    min_dof1 = minimize_scalar(lambda t: y(t)[0], bounds=(0, tdiv.T))
    max_dof1 = -min_dof1.fun
    print(f"Max. displ. of dof1: {max_dof1} m")


def test_basic_continuation(sys):
    """Basic sequential continuation through a frequency range."""

    # Chosen frequency range
    f_range = np.linspace(0.1, 35, 300)
    tdiv_range = np.array([TimeDivision() for _ in f_range])
    for idx, f in enumerate(f_range):
        tdiv_range[idx].f = f

    #IC state guess
    y0_guess = 1E-4 * np.array([1, 1, 0, 0])

    # Basic sequential continuation over the specified frequency range
    solution = shooting.basic_continuation(sys, y0_guess, tdiv_range)

    # Plot the NFRC of DOF x1
    plt.plot([sol.f for sol in solution.tdiv_range], solution.max_range[0, :])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (m)')
    plt.title("NFRC")
    plt.tight_layout()
    plt.show()

    return solution


if __name__ == '__main__':
    sys = test_system()
    test_shooting(sys)
    solution = test_basic_continuation(sys)
