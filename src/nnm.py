"""Computation of the nonlinear normal modes (NNMs)."""

# TODO:
# - must have an init method for TimeDivision, otherwise it is painful to set
#   an array of TimeDivision (see the test_basic_continuation function).
#   Maybe one can inspire from the aeroE code:
#   fft_signals = np.zeros(extracted_signals.shape[:2], dtype=ld.ExtractedSignal)

import matplotlib.pyplot as plt
import numpy as np

import nlsys
import shooting


def compute_nnm(sys, continuation=shooting.basic_continuation):
    """Compute NNMs and backbones with sequential continuation."""

    y0_guesses = [
        1E-3 * np.array([-7.435, -6.906, 0, 0]),
        1E-2 * np.array([-7.435,  6.906, 0, 0]),
    ]

    f_ranges = [
        np.linspace(15.94, 18.9, 500),  # 1st linear freq: 15.915Hz
        np.linspace(27.6,  29.4, 500),  # 2nd linear freq: 27.566Hz
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([nlsys.TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    # Sequential continuation over the specified frequency range
    sol_nnm = [0 for _ in f_ranges]
    sol_nnm[0] = continuation(sys, y0_guesses[0], tdiv_ranges[0])
    sol_nnm[1] = continuation(sys, y0_guesses[1], tdiv_ranges[1])

    return sol_nnm


def plot_backbone(sol_nnm) -> None:
    # Plot the natural freq amplitude of DOF x1
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for sol in sol_nnm:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :])
    ax.set_xlabel('Natural frequency (Hz)')
    ax.set_ylabel('DOF amplitude (m)')
    ax.set_title("Backbone of the NNMs")
    ax.grid()
    fig.show()


def plot_nnm(sol_nnm) -> None:
    pass


if __name__ == '__main__':
    # Build the nonlinear free system
    sys_free = nlsys.build_undamped_free_system(nlsys.f_nl)

    # Compute the NFRC
    sol_nnm = compute_nnm(sys_free)
    plot_backbone(sol_nnm)

    # Verify one of the system response, at the given
    # system time division (natural frequency)
    sys_tdiv = nlsys.TimeDivision()
    sys_tdiv.f = 15.94
    y0_guess = 1E-3 * np.array([-7.435, -6.906, 0, 0])
    shooting_sol_free = shooting.plot_BVP(sys_free, y0_guess, sys_tdiv)
