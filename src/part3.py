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


def compute_nnm_bounds(sys_free: nlsys.NLSystem, sol_nnm, id_mode=0):
    # TODO: not robust: break each time the elements order of sol_nnm are modified
    if id_mode == 0:
        y0_lf = sol_nnm[0].y0_range[:, -1]
        y0_hf = sol_nnm[1].y0_range[:, -1]
        tdiv_lf = sol_nnm[0].tdiv_range[-1]
        tdiv_hf = sol_nnm[1].tdiv_range[-1]
    if id_mode == 1:
        y0_lf = sol_nnm[2].y0_range[:, 0]
        y0_hf = sol_nnm[4].y0_range[:, -1]
        tdiv_lf = sol_nnm[2].tdiv_range[0]
        tdiv_hf = sol_nnm[4].tdiv_range[-1]

    nnm_lf = solve_ivp(
        sys_free.integrand,
        [0, tdiv_lf.T],
        y0_lf,
        args=(tdiv_lf.w,),
        t_eval=np.linspace(0, tdiv_lf.T, 300))
    nnm_hf = solve_ivp(
        sys_free.integrand,
        [0, tdiv_hf.T],
        y0_hf,
        args=(tdiv_hf.w,),
        t_eval=np.linspace(0, tdiv_hf.T, 300))

    return nnm_lf, nnm_hf, tdiv_lf, tdiv_hf


def plot_nnm_periodic_sol(nnm_lf, nnm_hf, tdiv_lf, tdiv_hf) -> None:
    fig, (ax_lf, ax_hf) = plt.subplots(1, 2)

    ax_lf.plot(nnm_lf.t, nnm_lf.y[:2, :].T, linewidth=0.6)
    ax_lf.set_title(f"{tdiv_lf.f:.2f} Hz")
    ax_lf.set_xlabel(r"Time [s]")
    ax_lf.set_ylabel(r"$x_1, x_2$ [m]")
    ax_lf.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_lf.grid(True, linewidth=0.5, alpha=0.3)

    ax_hf.plot(nnm_hf.t, nnm_hf.y[:2, :].T, linewidth=0.6)
    ax_hf.set_title(f"{tdiv_hf.f:.2f} Hz")
    ax_hf.set_xlabel(r"Time [s]")
    ax_hf.set_ylabel(r"$x_1, x_2$ [m]")
    ax_hf.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_hf.grid(True, linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.show()


def plot_nnm_config_space(nnm_lf, nnm_hf, tdiv_lf, tdiv_hf) -> None:
    fig, (ax_lf, ax_hf) = plt.subplots(1, 2)

    ax_lf.plot(nnm_lf.y[0, :], nnm_lf.y[1, :], color='C0', linewidth=0.6)
    ax_lf.set_title(f"{tdiv_lf.f:.2f} Hz")
    ax_lf.set_xlabel(r"$x_1$ [m]")
    ax_lf.set_ylabel(r"$x_2$ [m]")
    ax_lf.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax_lf.grid(True, linewidth=0.5, alpha=0.3)

    ax_hf.plot(nnm_hf.y[0, :], nnm_hf.y[1, :], color='C0', linewidth=0.6)
    ax_hf.set_title(f"{tdiv_hf.f:.2f} Hz")
    ax_hf.set_xlabel(r"$x_1$ [m]")
    ax_hf.set_ylabel(r"$x_2$ [m]")
    ax_hf.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax_hf.grid(True, linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.show()


def plot_attractor_2() -> None:
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    OUT_DIR = ROOT_DIR / "out"
    AT = np.loadtxt(str(OUT_DIR / "above_left_max_dof1.txt"))
    BT = np.flip(np.loadtxt(str(OUT_DIR / "above_right_max_dof1.txt")), 1)
    CT = np.flip(np.loadtxt(str(OUT_DIR / "below_left_max_dof1.txt")), 1)
    DT = np.loadtxt(str(OUT_DIR / "below_right_max_dof1.txt"))

    ss_ampl = np.vstack((np.hstack((AT, CT)), np.hstack((BT, DT)))).T

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    dof_sample = np.linspace(-0.2, 0.2, 200)
    dof1_mat, dof2_mat = np.meshgrid(dof_sample, dof_sample)

    ax.pcolormesh(dof1_mat, dof2_mat, ss_ampl, cmap='bwr')

    ax.set_xlabel(r'Initial displacement $x_{1}(0)$ [m]')
    ax.set_ylabel(r'Initial displacement $x_{2}(0)$ [m]')

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    mplrc.load_rcparams()

    # sys_free, sol_nfrc, sol_nnm = compute_nfrc_backbone()
    # sim_data = extract_simulation_data("group4_test3_2.mat")
    # nnm_fbounds = compute_nnm_bounds(sys_free, sol_nnm, id_mode=0)

    # plot_nfrc_backbone(sol_nfrc, sol_nnm)
    # plot_nfrc_envelope(sol_nfrc, *sim_data)
    # plot_nnm_periodic_sol(*nnm_fbounds)
    # plot_nnm_config_space(*nnm_fbounds)

    plot_attractor_2()
