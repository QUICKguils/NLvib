"""Derive results and plots needed to answer the third part of the project."""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from scipy import io

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

    return sol_nfrc, sol_nnm

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
    fig.tight_layout()
    fig.show()


def extract_simulation_data(fname: str):
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    RES_DIR = ROOT_DIR / "res"
    DATA_PATH = RES_DIR / fname
    DATA = io.loadmat(str(DATA_PATH))
    t = DATA['t']
    x = DATA['x']

    return x, t


def plot_nfrc_envelope(sol_nfrc, x, t) -> None:
    f = np.linspace(5, 60, len(t[0]))
    fig, ax = plt.subplots()
    ax.plot(f, x[0, :], color='C0', linewidth=0.6)
    for sol in sol_nfrc:
        ax.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :], color='C1', linewidth=0.8)
        ax.plot([sol.f for sol in sol.tdiv_range], sol.min_range[0, :], color='C1', linewidth=0.8)
    ax.set_xlabel(r"Excitation frequency [Hz]")
    ax.set_ylabel(r"DOF amplitude [m]")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, 40)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    sol_nfrc, sol_nnm = compute_nfrc_backbone()
    x, t = extract_simulation_data("group4_test3_2.mat")

    plot_nfrc_backbone(sol_nfrc, sol_nnm)
    plot_nfrc_envelope(sol_nfrc, x, t)
