# TODO:
# - must have an init method for TimeDivision, otherwise it is painful to set
#   an array of TimeDivision (see the test_basic_continuation function). Maybe
#   one can inspire from the aeroE code:
#   fft_signals = np.zeros(extracted_signals.shape[:2], dtype=ld.ExtractedSignal)

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from utils.tdiv import TimeDivision
import shooting

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


# TODO: make this cleaner. Not hardcode RFS values here
def f_nl(x, x_dot):
    """Nonlinear force, as identified by the RFS method."""
    return np.array([3.6e5*x[0]**2 + 7.6e6*x[0]**3, 9E6*x[1]**5])

# TODO: M, C, K shoud be global constants

def build_undamped_free_system(f_nl):
    # Linear system matrices
    M = np.eye(2)
    K = 1E4 * np.array([[2, -1], [-1, 2]])

    # Build the unforced NL system
    sys = shooting.NLSystem(M, K, f_nl)

    # Build the state-space repr of the NL system
    sys.build_undamped_free_state_space()

    return sys


def build_damped_forced_system(f_nl, f_ext_ampl):
    # Linear system matrices
    M = np.eye(2)
    C = np.array([[3, -1], [-1, 3]])
    K = 1E4 * np.array([[2, -1], [-1, 2]])

    # Build the unforced NL system
    sys = shooting.NLSystem(M, K, f_nl)

    # Add damping
    sys.add_damping(C)

    # Add an external excitation
    sys.add_harmonic_excitation(amplitude=f_ext_ampl)

    # Build the state-space repr of the NL system
    sys.build_damped_forced_state_space()

    return sys


def test_shooting(sys, y0_guess, freq_Hz):
    """Shooting for a particular excitation frequency."""

    # Set the excitation frequency
    tdiv = TimeDivision()
    tdiv.f = freq_Hz

    # Solve the BVP throught the shooting method
    sol_shooting = shooting.shooting(sys, y0_guess, tdiv)
    print(f"IC solution of the BVP: {sol_shooting.y0}")
    print(f"DOF maximas: {sol_shooting.max}")
    print(f"DOF minimas: {sol_shooting.min}")

    # Verify that the BVP has been solved correctly
    sol = solve_ivp(sys.integrand, [0, tdiv.T], sol_shooting.y0, args=(tdiv.w,), t_eval=np.linspace(0, tdiv.T, 300))
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


def test_nfrc(sys, y0_guess, continuation=shooting.basic_continuation):
    """Compute NFRCs with sequential continuation."""

    # Manually refine near the peaks
    f_ranges = [
        np.linspace(0.5,  5,     20),  # Segment 0
        np.linspace(5,    5.5,   50),  # Segment 1
        np.linspace(5.5,  7.5,   50),  # Segment 2
        np.linspace(7.5,  8.5,   50),  # Segment 3
        np.linspace(8.5,  10,    200), # Segment 4
        np.linspace(10,   13.5,  50),  # Segment 5
        np.linspace(13.5, 14.3,  400), # Segment 6
        np.linspace(14.3, 18.9,  300), # Segment 7
        np.linspace(20,   17,    200), # Segment 8
        np.linspace(20,   29.3,  300), # Segment 9
        np.linspace(30,   28.04, 100), # Segment 10
        np.linspace(30,   40,    50),  # Segment 11
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    # Sequential continuation over the specified frequency range
    solutions = [0 for _ in f_ranges]
    solutions[0]  = continuation(sys, y0_guess,                     tdiv_ranges[0])
    solutions[1]  = continuation(sys, solutions[0].y0_range[:, -1], tdiv_ranges[1])
    solutions[2]  = continuation(sys, solutions[1].y0_range[:, -1], tdiv_ranges[2])
    solutions[3]  = continuation(sys, solutions[2].y0_range[:, -1], tdiv_ranges[3])
    solutions[4]  = continuation(sys, solutions[3].y0_range[:, -1], tdiv_ranges[4])
    solutions[5]  = continuation(sys, solutions[4].y0_range[:, -1], tdiv_ranges[5])
    solutions[6]  = continuation(sys, solutions[5].y0_range[:, -1], tdiv_ranges[6])
    solutions[7]  = continuation(sys, solutions[6].y0_range[:, -1], tdiv_ranges[7])
    solutions[8]  = continuation(sys, y0_guess,                     tdiv_ranges[8])
    solutions[9]  = continuation(sys, y0_guess,                     tdiv_ranges[9])
    solutions[10] = continuation(sys, y0_guess,                     tdiv_ranges[10])
    solutions[11] = continuation(sys, y0_guess,                     tdiv_ranges[11])

    # Plot the NFRC of DOF x1
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for solution in solutions:
        ax.plot([sol.f for sol in solution.tdiv_range], solution.max_range[0, :])
    ax.set_xlabel('Excitation frequency (Hz)')
    ax.set_ylabel('DOF amplitude (m)')
    ax.set_title("NFRC")
    ax.grid()
    fig.show()

    return solutions


def test_nnm(sys, y0_guess, continuation=shooting.basic_continuation):
    """Compute NNMs and backbones with sequential continuation."""

    f_ranges = [
        np.linspace(15.915, 15, 300),
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    # Sequential continuation over the specified frequency range
    solutions = [0 for _ in f_ranges]
    solutions[0] = continuation(sys, y0_guess, tdiv_ranges[0])

    # Plot the natural freq amplitude of DOF x1
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for solution in solutions:
        ax.plot([sol.f for sol in solution.tdiv_range], solution.max_range[0, :])
    ax.set_xlabel('Natural frequency (Hz)')
    ax.set_ylabel('DOF amplitude (m)')
    ax.set_title("Backbone of the NNMs")
    ax.grid()
    fig.show()

    return solutions


if __name__ == '__main__':
    # Set simulation parameters
    f_ext_ampl = 50
    f_ext_freq = 13.9851
    y0_guess = 1E-2 * np.array([1, 1, 0, 0])
    # f_nl = lambda x, x_dot: np.array([-2E4*x[0]**3, 0])

    # Test the code
    # sys_free = build_undamped_free_system(f_nl)
    # shooting_sol_free = test_shooting(sys_free, y0_guess, f_ext_freq)
    sys_forced = build_damped_forced_system(f_nl, f_ext_ampl)
    # shooting_sol_forced = test_shooting(sys_forced, y0_guess, f_ext_freq)
    nfrc_sol = test_nfrc(sys_forced, y0_guess)
    # nnm_sol = test_nnm(sys_free, y0_guess)
