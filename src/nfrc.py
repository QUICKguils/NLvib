# TODO:
# - must have an init method for TimeDivision, otherwise it is painful to set
#   an array of TimeDivision (see the test_basic_continuation function).
#   Maybe one can inspire from the aeroE code:
#   fft_signals = np.zeros(extracted_signals.shape[:2], dtype=ld.ExtractedSignal)

"""Computation of the nonlinear frequency response curves (NFRCs)."""

import matplotlib.pyplot as plt
import numpy as np

import nlsys
import shooting


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
        np.linspace(20,   16.98, 200), # Segment 8
        np.linspace(20,   29.3,  300), # Segment 9
        np.linspace(30,   28.03, 100), # Segment 10
        np.linspace(30,   40,    50),  # Segment 11
    ]

    # Build the associated time divisions
    tdiv_ranges = [0 for _ in f_ranges]
    for idx_r, f_range in enumerate(f_ranges):
        tdiv_ranges[idx_r] = np.array([nlsys.TimeDivision() for _ in f_range])
        for idx_f, f in enumerate(f_range):
            tdiv_ranges[idx_r][idx_f].f = f

    # Sequential continuation over the specified frequency range
    sol_nfrc = [0 for _ in f_ranges]
    sol_nfrc[0]  = continuation(sys, y0_guess,                    tdiv_ranges[0])
    sol_nfrc[1]  = continuation(sys, sol_nfrc[0].y0_range[:, -1], tdiv_ranges[1])
    sol_nfrc[2]  = continuation(sys, sol_nfrc[1].y0_range[:, -1], tdiv_ranges[2])
    sol_nfrc[3]  = continuation(sys, sol_nfrc[2].y0_range[:, -1], tdiv_ranges[3])
    sol_nfrc[4]  = continuation(sys, sol_nfrc[3].y0_range[:, -1], tdiv_ranges[4])
    sol_nfrc[5]  = continuation(sys, sol_nfrc[4].y0_range[:, -1], tdiv_ranges[5])
    sol_nfrc[6]  = continuation(sys, sol_nfrc[5].y0_range[:, -1], tdiv_ranges[6])
    sol_nfrc[7]  = continuation(sys, sol_nfrc[6].y0_range[:, -1], tdiv_ranges[7])
    sol_nfrc[8]  = continuation(sys, y0_guess,                    tdiv_ranges[8])
    sol_nfrc[9]  = continuation(sys, y0_guess,                    tdiv_ranges[9])
    sol_nfrc[10] = continuation(sys, y0_guess,                    tdiv_ranges[10])
    sol_nfrc[11] = continuation(sys, y0_guess,                    tdiv_ranges[11])

    return sol_nfrc


# TODO: allow to plot for DOF2
def plot_nfrc(sol_nfrc) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for sol in sol_nfrc:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :])
    ax.set_xlabel('Excitation frequency (Hz)')
    ax.set_ylabel('DOF amplitude (m)')
    ax.set_title("NFRC")
    ax.grid()
    fig.show()


if __name__ == '__main__':
    # Set simulation parameters
    f_ext_ampl = 50
    f_ext_freq = 15
    y0_guess = 1E-2 * np.array([1, 1, 0, 0])


    # Forced, damped system
    sys_forced = nlsys.build_damped_forced_system(nlsys.f_nl, f_ext_ampl)
    shooting_sol_forced = shooting.plot_BVP(sys_forced, y0_guess, f_ext_freq)
    sol_nfrc = test_nfrc(sys_forced, y0_guess)
    plot_nfrc(sol_nfrc)

