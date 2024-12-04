# TODO:
# - must have an init method for TimeDivision, otherwise it is painful to set
#   an array of TimeDivision (see the test_basic_continuation function).
#   Maybe one can inspire from the aeroE code:
#   fft_signals = np.zeros(extracted_signals.shape[:2], dtype=ld.ExtractedSignal)

"""Computation of the nonlinear normal modes (NNMs)."""

import matplotlib.pyplot as plt
import numpy as np

import nlsys
import shooting


def test_nnm(sys, y0_guess, continuation=shooting.basic_continuation):
    """Compute NNMs and backbones with sequential continuation."""

    f_ranges = [
        np.linspace(15.94, 18.9, 500),  # 1st linear freq: 15.915Hz
        # np.linspace(15.94, 18.9, 500),  # 2nd linear freq: 27.566Hz
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([nlsys.TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    # Sequential continuation over the specified frequency range
    sol_nnm = [0 for _ in f_ranges]
    sol_nnm[0] = continuation(sys, y0_guess, tdiv_ranges[0])

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


if __name__ == '__main__':
    # Set simulation parameters
    f_ext_freq = 15.94
    y0_guess = 1E-3 * np.array([-7.435, -6.906, 0, 0])
    # f_ext_freq = 16.187
    # y0_guess = np.array([-0.0219, -0.0253, 0, 0])
    # f_ext_freq = 36.54
    # y0_guess = np.array([-0.12, -0.29, 0, 0])

    # Free, undamped system
    sys_free = nlsys.build_undamped_free_system(nlsys.f_nl)
    shooting_sol_free = shooting.plot_BVP(sys_free, y0_guess, f_ext_freq)
    sol_nnm = test_nnm(sys_free, y0_guess)
    plot_backbone(sol_nnm)
