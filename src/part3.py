"""Derive results and plots needed to answer the third part of the project."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from scipy.integrate import solve_ivp

import mplrc  # Set the plot dressing
import nlsys
import nfrc
import nnm


def compute_nfrc_backbone():
    # Build the nonlinear forced and free systems
    f_ext_ampl = 50  # Excitation force amplitude (N)
    sys_free = nlsys.build_undamped_free_system(nlsys.f_nl)
    sys_forced = nlsys.build_damped_forced_system(nlsys.f_nl, f_ext_ampl)

    # Compute the NFRC and NNM
    y0_guess = 1E-2 * np.array([1, 1, 0, 0])  # Default IC guess
    sol_nfrc = nfrc.compute_nfrc(sys_forced, y0_guess)
    sol_nnm = nnm.compute_nnm(sys_free)

    return sys_free, sol_nfrc, sol_nnm

def plot_nfrc_backbone(sol_nfrc, sol_nnm) -> None:
    fig, ax = plt.subplots()
    for sol in sol_nnm:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :], color='C1', linewidth=0.6)
    for sol in sol_nfrc:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :], color='C0', linewidth=0.6)
    ax.set_xlabel(r"Excitation frequency [Hz]")
    ax.set_ylabel(r"DOF amplitude [m]")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.05)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.show()


def extract_simulation_data(fname: str):
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    RES_DIR = ROOT_DIR / "res"
    DATA_PATH = RES_DIR / fname
    DATA = io.loadmat(str(DATA_PATH))
    t = DATA['t']
    x = DATA['x']
    freq_start = DATA['ext_force'][0, 2][0][0]
    freq_end = DATA['ext_force'][0, 3][0][0]

    return x, t, freq_start, freq_end


def plot_nfrc_envelope(sol_nfrc, x, t, freq_start, freq_end) -> None:
    freq_sample = np.linspace(freq_start, freq_end, len(t[0]))
    fig, ax = plt.subplots()
    ax.plot(freq_sample, x[0, :], color='C0', linewidth=0.6)
    for sol in sol_nfrc:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :], color='C1', linewidth=0.8)
        ax.plot([sol.f for sol in sol.tdiv_range], sol.min_range[0, :], color='C1', linewidth=0.8)
    ax.set_xlabel(r"Excitation frequency [Hz]")
    ax.set_ylabel(r"DOF amplitude [m]")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, 40)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.show()


def plot_nnm(sys_free: nlsys.NLSystem, sol_nnm, id_mode=0) -> None:
    sol_bb = sol_nnm[id_mode]
    y0_low = sol_bb.y0_range[:, 0]
    y0_high = sol_bb.y0_range[:, -1]
    tdiv_low = sol_bb.tdiv_range[0]
    tdiv_high = sol_bb.tdiv_range[-1]

    sol_low = solve_ivp(
        sys_free.integrand,
        [0, tdiv_low.T],
        y0_low,
        args=(tdiv_low.w,),
        t_eval=np.linspace(0, tdiv_low.T, 300))
    sol_high = solve_ivp(
        sys_free.integrand,
        [0, tdiv_high.T],
        y0_high,
        args=(tdiv_high.w,),
        t_eval=np.linspace(0, tdiv_high.T, 300))

    fig, (ax_low, ax_high) = plt.subplots(1, 2)

    ax_low.plot(sol_low.y[0, :], sol_low.y[1, :], color='C0', linewidth=0.6)
    ax_low.set_xlabel(r"$x_1$ [m]")
    ax_low.set_ylabel(r"$x_2$ [m]")
    ax_low.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax_low.grid(True, linewidth=0.5, alpha=0.3)

    ax_high.plot(sol_high.y[0, :], sol_high.y[1, :], color='C0', linewidth=0.6)
    ax_high.set_xlabel(r"$x_1$ [m]")
    ax_high.set_ylabel(r"$x_2$ [m]")
    ax_high.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax_high.grid(True, linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.show()

    fig, (ax_low, ax_high) = plt.subplots(1, 2)

    ax_low.plot(sol_low.t, sol_low.y[:2, :].T, linewidth=0.6)
    ax_low.set_xlabel(r"Time [s]")
    ax_low.set_ylabel(r"$x_1, x_2$ [m]")
    ax_low.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_low.grid(True, linewidth=0.5, alpha=0.3)

    ax_high.plot(sol_high.t, sol_high.y[:2, :].T, linewidth=0.6)
    ax_high.set_xlabel(r"Time [s]")
    ax_high.set_ylabel(r"$x_1, x_2$ [m]")
    ax_high.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_high.grid(True, linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    mplrc.load_rcparams()

    sys_free, sol_nfrc, sol_nnm = compute_nfrc_backbone()
    sim_data = extract_simulation_data("group4_test3_2.mat")

    plot_nfrc_backbone(sol_nfrc, sol_nnm)
    plot_nfrc_envelope(sol_nfrc, *sim_data)
    plot_nnm(sys_free, sol_nnm, id_mode=0)
    plot_nnm(sys_free, sol_nnm, id_mode=1)
