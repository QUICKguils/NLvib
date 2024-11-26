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

# plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
# plt.rcParams['figure.figsize'] = (6.34, 3.34)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 200


# def f_nl(x, x_dot):
#     """Nonlinear force previously identified with RFS."""
#     nl_coeffs = np.array([
#         -2.09521445e+01,  1.36036473e+04,
#         -6.60445565e-01, -3.30382563e+03,
#         -9.68259415e-02, -1.15548181e+00,
#          2.71928261e+00, -2.52949130e+00,
#         -1.54618632e+05, -1.15104547e-03,
#     ])
#     f_assumed = np.array([
#         [1,                0],
#         [x[0],             0],
#         [x_dot[0],         0],
#         [x[0]**2,          0],
#         [x_dot[0]**2,      0],
#         [x[0]*x_dot[0],    0],
#         [x[0]**2*x_dot[0], 0],
#         [x[0]*x_dot[0]**2, 0],
#         [x[0]**3,          0],
#         [x_dot[0]**3,      0],
#     ]).T
#     return f_assumed @ nl_coeffs

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
    print(sol_shooting.y0)

    # Verify that the BVP has been solved correctly
    y = solve_ivp(sys.integrand, [0, 2*tdiv.T], sol_shooting.y0, args=(tdiv.w,), dense_output=True).sol
    t_sample = np.linspace(0, 2*tdiv.T, 200)
    plt.plot(t_sample, y(t_sample)[:2, :].T)
    plt.legend(['x1', 'x2'])
    # plt.plot(y(t_sample)[0, :].T, y(t_sample)[1, :].T)
    # plt.plot(t_sample, y(t_sample).T)
    # plt.legend(['x1', 'x2', 'x1_dot', 'x2_dot'])
    plt.xlabel('time (s)')
    plt.ylabel('State vector')
    plt.title(f"BVP solution (f = {sol_shooting.tdiv.f} Hz)")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return sol_shooting


def test_nfrc(sys, y0_guess, continuation=shooting.basic_continuation):
    """Compute NFRCs with sequential continuation."""

    # NOTE::
    # building a quick model of the linear system in NI2D show that the
    # natural frequencies are about 16Hz and 27.6Hz.
    f_ranges = [
        np.linspace(0.1,  17,   100),  # From 0Hz to the first natural frequency
        np.linspace(17,   21,   300),  # Refine around the fundamental peak of the first freq
        np.linspace(16.4, 27.4, 100),  # Restart from low amplitude, between the two fundamental peaks
        np.linspace(27.4, 30,   100),  # Refine around the fundamental peak of the second freq
        np.linspace(27.8, 35,   100),  # Restart from low amplitude, after the second fundamental peak
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

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


def test_nnm(sys, y0_guess, continuation=shooting.basic_continuation):
    """Compute NNMs and backbones with sequential continuation."""

    f_ranges = [
        np.linspace(18, 14, 200),
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
    for solution in solutions:
        plt.plot([sol.f for sol in solution.tdiv_range], solution.max_range[0, :])
    plt.xlabel('Natural frequency (Hz)')
    plt.ylabel('DOF amplitude (m)')
    plt.title("Backbone of the NNMs")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return solutions


if __name__ == '__main__':
    # Set simulation parameters
    f_ext_ampl = 50
    f_ext_freq = 18
    y0_guess = 1E-6 * np.array([15, 20, 0, 0])
    f_nl = lambda x, x_dot: np.array([-2E4*x[0]**3, 0])

    # Test the code
    sys_free = build_undamped_free_system(f_nl)
    shooting_sol_free = test_shooting(sys_free, y0_guess, f_ext_freq)
    sys_forced = build_damped_forced_system(f_nl, f_ext_ampl)
    shooting_sol_forced = test_shooting(sys_forced, y0_guess, f_ext_freq)
    # nfrc_sol = test_nfrc(sys_forced, y0_guess)
    # nnm_sol = test_nnm(sys_free, y0_guess)
