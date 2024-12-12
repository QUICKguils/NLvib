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


ROOT_DIR = pathlib.Path(__file__).parent.parent
RES_DIR = ROOT_DIR / "res"


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
    # NOTE:
    # the backbones computed by ni2d are the absolute maximum between the
    # negative and positive amplitudes, while the nfrc are computed as the mean
    # value of the maximal negative and positive amplitude. It thus does not
    # make sense to plot them here.
    nfrc_dof1_ni2d = np.loadtxt(str(RES_DIR/"nfrc_dof1.csv"), skiprows=1, delimiter=',')
    nfrc_dof2_ni2d = np.loadtxt(str(RES_DIR/"nfrc_dof2.csv"), skiprows=1, delimiter=',')

    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(6.34, 6.34), layout="constrained")

    ax_1.plot(nfrc_dof1_ni2d[:, 0], nfrc_dof1_ni2d[:, 1], color='C7', linewidth=0.4)
    ax_2.plot(nfrc_dof2_ni2d[:, 0], nfrc_dof2_ni2d[:, 1], color='C7', linewidth=0.4)
    # ax_2.plot(backbone_dof2_nnm1_ni2d[:, 0], backbone_dof2_nnm1_ni2d[:, 1], color='C7', linewidth=0.6, linestyle='--')
    # ax_2.plot(backbone_dof2_nnm2_ni2d[:, 0], backbone_dof2_nnm2_ni2d[:, 1], color='C7', linewidth=0.6, linestyle='--')

    for sol in sol_nnm:
        avg_max_1 = (np.abs(sol.max_range[0, :]) + np.abs(sol.min_range[0, :]))/2
        avg_max_2 = (np.abs(sol.max_range[1, :]) + np.abs(sol.min_range[1, :]))/2
        ax_1.plot([sol.f for sol in sol.tdiv_range], avg_max_1, color='C1', linewidth=0.6)
        ax_2.plot([sol.f for sol in sol.tdiv_range], avg_max_2, color='C1', linewidth=0.6)
    for sol in sol_nfrc:
        avg_max_1 = (np.abs(sol.max_range[0, :]) + np.abs(sol.min_range[0, :]))/2
        avg_max_2 = (np.abs(sol.max_range[1, :]) + np.abs(sol.min_range[1, :]))/2
        ax_1.plot([sol.f for sol in sol.tdiv_range], avg_max_1, color='C0', linewidth=0.6)
        ax_2.plot([sol.f for sol in sol.tdiv_range], avg_max_2, color='C0', linewidth=0.6)

    ax_2.set_xlabel(r"Excitation frequency [Hz]")
    ax_1.set_ylabel(r"$q_1$ amplitude [m]")
    ax_2.set_ylabel(r"$q_2$ amplitude [m]")
    ax_1.grid(True, linewidth=0.5, alpha=0.3)
    ax_2.grid(True, linewidth=0.5, alpha=0.3)
    ax_1.set_xlim(0, 40)
    ax_2.set_xlim(0, 40)
    ax_1.set_ylim(0, 0.055)
    ax_2.set_ylim(0, 0.085)
    ax_1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    fig.show()


def plot_nfrc_envelope(sol_nfrc) -> None:
    sim_data = io.loadmat(str(RES_DIR/"group4_test3_1.mat"))
    x = sim_data['x']
    freq_start = sim_data['ext_force'][0, 2][0][0]
    freq_end = sim_data['ext_force'][0, 3][0][0]
    freq_sample = np.linspace(freq_start, freq_end, len(x[0]))

    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(6.34, 6.34), layout="constrained")

    ax_1.plot(freq_sample, x[0, :], color='C0', linewidth=0.6)
    ax_2.plot(freq_sample, x[1, :], color='C0', linewidth=0.6)
    for sol in sol_nfrc:
        ax_1.plot([sol.f for sol in sol.tdiv_range], sol.max_range[0, :], color='C1', linewidth=0.8)
        ax_1.plot([sol.f for sol in sol.tdiv_range], sol.min_range[0, :], color='C1', linewidth=0.8)
        ax_2.plot([sol.f for sol in sol.tdiv_range], sol.max_range[1, :], color='C1', linewidth=0.8)
        ax_2.plot([sol.f for sol in sol.tdiv_range], sol.min_range[1, :], color='C1', linewidth=0.8)

    ax_2.set_xlabel(r"Excitation frequency [Hz]")
    ax_1.set_ylabel(r"$q_1$ amplitude [m]")
    ax_2.set_ylabel(r"$q_2$ amplitude [m]")
    ax_1.grid(True, linewidth=0.5, alpha=0.3)
    ax_2.grid(True, linewidth=0.5, alpha=0.3)
    ax_1.set_xlim(0, 40)
    ax_2.set_xlim(0, 40)
    ax_1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    fig.show()


def compute_largest_nnm(sys_free: nlsys.NLSystem, sol_nnm, n_mode=1):
    # TODO: not robust: break each time the elements order of sol_nnm are modified
    # Take the nnm computed at the highest amplitude
    if n_mode == 1:
        y0 = sol_nnm[1].y0_range[:, -1]
        largest_tdiv = sol_nnm[1].tdiv_range[-1]
    if n_mode == 2:
        y0 = sol_nnm[4].y0_range[:, -1]
        largest_tdiv = sol_nnm[4].tdiv_range[-1]

    largest_nnm = solve_ivp(
        sys_free.integrand,
        [0, largest_tdiv.T],
        y0,
        args=(largest_tdiv.w,),
        t_eval=np.linspace(0, largest_tdiv.T, 300))

    return largest_nnm, largest_tdiv


def plot_nnm_backbone(sol_nnm, nnm, tdiv, n_mode=1) -> None:
    backbone_dof1_ni2d = np.loadtxt(str(RES_DIR/f"backbone_dof1_nnm{n_mode}.csv"), skiprows=1, delimiter=',')
    backbone_dof2_ni2d = np.loadtxt(str(RES_DIR/f"backbone_dof2_nnm{n_mode}.csv"), skiprows=1, delimiter=',')

    fig, axs = plt.subplot_mosaic(
        [['bb1', 'bb1'], ['bb2', 'bb2'], ['nnm_ps', 'nnm_ss']],
        figsize=(7, 7),
        layout='constrained')

    axs['bb1'].plot(backbone_dof1_ni2d[:, 0], backbone_dof1_ni2d[:, 1], color='C7', linewidth=0.6, linestyle='--')
    axs['bb2'].plot(backbone_dof2_ni2d[:, 0], backbone_dof2_ni2d[:, 1], color='C7', linewidth=0.6, linestyle='--')

    # TODO: not robust
    if n_mode == 1:
        backbone = sol_nnm[:2]
    if n_mode == 2:
        backbone = sol_nnm[2:]

    for bb in backbone:
        axs['bb1'].plot([bb.f for bb in bb.tdiv_range], np.abs(bb.min_range[0, :]), color='C0', linewidth=0.6)
        axs['bb2'].plot([bb.f for bb in bb.tdiv_range], np.abs(bb.min_range[1, :]), color='C0', linewidth=0.6)

    # axs['bb1'].scatter(tdiv.f, np.abs(bb.min_range[0, -1]), color='C2', linewidths=1, marker='x')
    # axs['bb2'].scatter(tdiv.f, np.abs(bb.min_range[1, -1]), color='C2', linewidths=1, marker='x')

    axs['bb1'].set_title("Backbone, DOF 1")
    axs['bb2'].set_title("Backbone, DOF 2")
    axs['bb2'].set_xlabel("Natural frequency [Hz]")
    axs['bb1'].set_ylabel(r"$q_1$ [m]")
    axs['bb2'].set_ylabel(r"$q_2$ [m]")
    axs['bb1'].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs['bb2'].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs['bb1'].grid(True, linewidth=0.5, alpha=0.3)
    axs['bb2'].grid(True, linewidth=0.5, alpha=0.3)
    if n_mode == 1:
        axs['bb1'].set_ylim(0, 0.065)
        axs['bb2'].set_ylim(0, 0.1)
    if n_mode == 2:
        axs['bb1'].set_ylim(0, 0.065)
        axs['bb2'].set_ylim(0, 0.045)

    axs['nnm_ps'].plot(nnm.t, nnm.y[0, :].T, linewidth=0.6, label=r"$q_1$")
    axs['nnm_ps'].plot(nnm.t, nnm.y[1, :].T, linewidth=0.6, label=r"$q_2$")
    axs['nnm_ps'].set_title("Periodic solution")
    axs['nnm_ps'].set_xlabel(r"Time [s]")
    axs['nnm_ps'].set_ylabel(r"$q_1, q_2$ [m]")
    axs['nnm_ps'].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs['nnm_ps'].grid(True, linewidth=0.5, alpha=0.3)
    axs['nnm_ps'].legend()

    axs['nnm_ss'].plot(nnm.y[0, :], nnm.y[1, :], color='C0', linewidth=0.6)
    axs['nnm_ss'].set_title("Configuration space")
    axs['nnm_ss'].set_xlabel(r"$q_1$ [m]")
    axs['nnm_ss'].set_ylabel(r"$q_2$ [m]")
    axs['nnm_ss'].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs['nnm_ss'].grid(True, linewidth=0.5, alpha=0.3)

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

    mplrc.load_rcparams(style='custom')

    sys_free, sol_nfrc, sol_nnm = compute_nfrc_backbone()
    largest_1 = compute_largest_nnm(sys_free, sol_nnm, n_mode=1)
    largest_2 = compute_largest_nnm(sys_free, sol_nnm, n_mode=2)

    plot_nfrc_backbone(sol_nfrc, sol_nnm)
    plot_nfrc_envelope(sol_nfrc)
    plot_nnm_backbone(sol_nnm, *largest_1, n_mode=1)
    plot_nnm_backbone(sol_nnm, *largest_2, n_mode=2)
    plot_attractor_2()
