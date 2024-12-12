"""Quantification of damping nonlinearities with the restoring force surface (RFS) method."""

import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
from scipy import io

from nlsys import M, C, K

ROOT_DIR = pathlib.Path(__file__).parent.parent
RES_DIR = ROOT_DIR / "res"
OUT_DIR = ROOT_DIR / "out"


def build_row(n, vect1, vect2, idx_time):
    row = []

    for i in range(1, n + 1):
        row.append((vect1[idx_time])**i)
    for i in range(1, n + 1):
        row.append((vect2[idx_time])**i)
    for i in range(1, n + 1):
        row.append((vect1[idx_time] - vect2[idx_time])**i)

    return row


def func_matrix(n, vect1, vect2):
    # n = order of polynomial
    # vect1 and vect2 = displacements or velocities vectors at every time step

    nb_rows = len(vect1)

    # start building matrix from index 0
    f_mtrx = build_row(n, vect1, vect2, 0)

    for i in range(1, nb_rows):
        row_i = build_row(n, vect1, vect2, i)

        f_mtrx = np.vstack((f_mtrx, row_i))

    return f_mtrx


def measure_matrix(f_ext_vect, q_dot_dot_vect, q_dot_vect, q_vect):
    # all vectors contain only the selected time interval
    measured_mtrx = (
        np.transpose(f_ext_vect)
            - np.transpose(q_dot_dot_vect) @ M
            - np.transpose(q_dot_vect) @ C
            - np.transpose(q_vect) @ K
    )

    return measured_mtrx


data = io.loadmat(str(RES_DIR/"group4_test3_1.mat"))

dt = 5e-5

# -- select only part of the time response --

t_min = 110
t_max = 140

idx_t_min = int(t_min/dt)
idx_t_max = int(t_max/dt)

skip_nb = 10

acc_vect = data['xdd'][:, idx_t_min:idx_t_max:skip_nb]
vel_vect = data['xd'][:, idx_t_min:idx_t_max:skip_nb]
disp_vect = data['x'][:, idx_t_min:idx_t_max:skip_nb]
f_ext_vect = data['pex'][:, idx_t_min:idx_t_max:skip_nb]

# damping -> use velocities
vect1 = vel_vect[0][:]
vect2 = vel_vect[1][:]

# --------coefficients convergence--------
# base order
n_base = 5

# -- order of polynomial --
n_max = 20

# arrays that will store the coefficients
# one row = one order
x1_coeffs1 = np.zeros((n_max - n_base + 1, n_base))
x2_coeffs1 = np.zeros((n_max - n_base + 1, n_base))
x1_minus_x2_coeffs1 = np.zeros((n_max - n_base + 1, n_base))

x1_coeffs2 = np.zeros((n_max - n_base + 1, n_base))
x2_coeffs2 = np.zeros((n_max - n_base + 1, n_base))
x1_minus_x2_coeffs2 = np.zeros((n_max - n_base + 1, n_base))

# compute the coefficients for each order and the new values of the coefficients
orders_array = np.arange(n_base, n_max + 1, 1)

for i in range(len(orders_array)):
    print(f"RFS dampings: order {orders_array[i]}")
    n_order = orders_array[i]

    Ai = func_matrix(n_order, vect1, vect2)
    bi = measure_matrix(f_ext_vect, acc_vect, vel_vect, disp_vect)

    ki = np.linalg.pinv(Ai) @ bi

    # for dof 1 - first column of ki
    x1_coeffs1[i, :] = abs(ki[:n_base, 0])
    x2_coeffs1[i, :] = abs(ki[n_order:n_order + n_base, 0])
    x1_minus_x2_coeffs1[i, :] = abs(ki[2*n_order:2*n_order + n_base, 0])

    # for dof 2 - second column of ki
    x1_coeffs2[i, :] = abs(ki[:n_base, 1])
    x2_coeffs2[i, :] = abs(ki[n_order:n_order + n_base, 1])
    x1_minus_x2_coeffs2[i, :] = abs(ki[2*n_order:2*n_order + n_base, 1])

x1_coeffs1 = np.loadtxt('damping_k_coeffs/x1_1.txt')
x2_coeffs1 = np.loadtxt('damping_k_coeffs/x1_1.txt')
x1_minus_x2_coeffs1 = np.loadtxt('damping_k_coeffs/x1_minus_x2_1.txt')

x1_coeffs2 = np.loadtxt('damping_k_coeffs/x1_2.txt')
x2_coeffs2 = np.loadtxt('damping_k_coeffs/x2_2.txt')
x1_minus_x2_coeffs2 = np.loadtxt('damping_k_coeffs/x1_minus_x2_2.txt')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

dofs = [1, 2]

size_dots = 8
size_lines = 0.6

for ax, dof in zip(axs, dofs):
    if dof == 1:
        for i in range(np.shape(x1_coeffs1)[1]):
            ax.plot(orders_array, x1_coeffs1[:, i], linewidth=size_lines)
            ax.plot(orders_array, x2_coeffs1[:, i], linewidth=size_lines)
            ax.plot(orders_array, x1_minus_x2_coeffs1[:, i], linewidth=size_lines)

            ax.scatter(orders_array, x1_coeffs1[:, i], s=size_dots)
            ax.scatter(orders_array, x2_coeffs1[:, i], s=size_dots)
            ax.scatter(orders_array, x1_minus_x2_coeffs1[:, i], s=size_dots)

            # ax.annotate(r'$k_{q_1^2}$', (14, 3.6e5 + 5e5), font='Times New Roman', fontsize=11, color='#9467bd')
            # ax.annotate(r'$k_{q_1^3}$', (14, 7.6e6 + 8e6), font='Times New Roman', fontsize=11, color='#828282')
    if dof == 2:
        for i in range(np.shape(x1_coeffs2)[1]):
            ax.plot(orders_array, x1_coeffs2[:, i], linewidth=size_lines)
            ax.plot(orders_array, x2_coeffs2[:, i], linewidth=size_lines)
            ax.plot(orders_array, x1_minus_x2_coeffs2[:, i], linewidth=size_lines)

            ax.scatter(orders_array, x1_coeffs2[:, i], s=size_dots)
            ax.scatter(orders_array, x2_coeffs2[:, i], s=size_dots)
            ax.scatter(orders_array, x1_minus_x2_coeffs2[:, i], s=size_dots)

            # ax.annotate(r'$k_{q_2^5}$', (14, 9e6 + 9e6), font='Times New Roman', fontsize=11, color='#d62728')

    ax.grid(True, linewidth=0.5, alpha=0.3)

    ax.set_xticks([5, 10, 15, 20])

    fs_ticks = 9
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs_ticks)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs_ticks)

    fs = 12
    ax.set_xlabel(r'Maximum polynomial order $n$ [-]', font='Times New Roman', fontsize=fs)
    ax.set_ylabel(r'Coefficient value $k$' if dof == 1 else '', font='Times New Roman', fontsize=fs)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[], numticks=10))

plt.subplots_adjust(wspace=0.08)

plt.minorticks_off()

# plt.ylim(1e-5, 1e3)
# plt.yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

plt.show()

# measured_forces = np.transpose(f_ext_vect - M @ acc_vect - C @ vel_vect - K @ disp_vect)
#
# # choose which k coefficients to include
# k_ret_1 = np.zeros(np.shape(k)[0])
# k_ret_2 = np.zeros(np.shape(k)[0])
#
# # place 1 where you retain k coefficient
# k_ret_1[1] = 1
# k_ret_1[2] = 1
#
# k_ret_2[14] = 1
#
# f_nl_fitted = np.zeros((np.shape(measured_forces)[0], np.shape(measured_forces)[1]))
# # loop over the Q times measured
# for i in range(np.shape(f_nl_fitted)[0]):
#
#     row_j = build_row(n, vect1, vect2, i)
#     for j in range(np.shape(k)[0]):
#         f_nl_fitted[i][0] += k[j][0]*row_j[j]*k_ret_1[j]
#         f_nl_fitted[i][1] += k[j][1]*row_j[j]*k_ret_2[j]
#
# plt.scatter(np.linspace(0, 1, len(measured_forces[:, 0])), measured_forces[:, 0])
# plt.show()
#
# plt.plot(np.linspace(0, 1, len(f_nl_fitted[:, 0])), f_nl_fitted[:, 0])
# plt.show()
#
# MSE_error_1 = 0
# MSE_error_2 = 0
# for i in range(len(f_nl_fitted[:, 0])):
#     MSE_error_1 += (measured_forces[i, 0] - f_nl_fitted[i, 0])**2
#     MSE_error_2 += (measured_forces[i, 1] - f_nl_fitted[i, 1])**2
#
# MSE_error_1 = MSE_error_1*100/(len(measured_forces[:, 0])*np.var(measured_forces[:, 0]))
# MSE_error_2 = MSE_error_2*100/(len(measured_forces[:, 1])*np.var(measured_forces[:, 1]))
#
# print('errors:', MSE_error_1, MSE_error_2)
#
#
# plt.scatter(disp_vect[0, :] - disp_vect[1, :], - acc_vect[0, :])
#
# y1 = lambda x1: x1**2 + x1**3 # because of the chosen k coefficients
# plt.plot(disp_vect[0, :] - disp_vect[1, :], y1(disp_vect[0, :] - disp_vect[1, :]))
