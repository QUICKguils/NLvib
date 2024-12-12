"""Quantification of stiffness nonlinearities with the restoring force surface (RFS) method."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


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


# Constant structural matrices of the linear system
M = [[1, 0], [0, 1]]
C = [[3, -1], [-1, 3]]
K = [[2*1e4, -1*1e4], [-1*1e4, 2*1e4]]

def measure_matrix(f_ext_vect, q_dot_dot_vect, q_dot_vect, q_vect):
    # all vectors contain only the selected time interval
    measured_mtrx = (
        np.transpose(f_ext_vect)
            - np.transpose(q_dot_dot_vect) @ M
            - np.transpose(q_dot_vect) @ C
            - np.transpose(q_vect) @ K
    )

    return measured_mtrx

file_name = 'data_test/group4_test3_1.mat'
data = sio.loadmat(file_name)

dt = 5e-5

# -- select only part of the time response --

t_min = 110
t_max = 140

idx_t_min = int(t_min/dt)
idx_t_max = int(t_max/dt)

acc_vect = data['xdd'][:, idx_t_min:idx_t_max:10]
vel_vect = data['xd'][:, idx_t_min:idx_t_max:10]
disp_vect = data['x'][:, idx_t_min:idx_t_max:10]
f_ext_vect = data['pex'][:, idx_t_min:idx_t_max:10]

# stiffness -> use displacements
vect1 = disp_vect[0][:]
vect2 = disp_vect[1][:]

# --------coefficients convergence--------
# base order
n_base = 5

# -- order of polynomial --
n_max = 15

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
    print(orders_array[i])
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

np.savetxt('k_coeffs/x1_1.txt', x1_coeffs1)
np.savetxt('k_coeffs/x2_1.txt', x2_coeffs1)
np.savetxt('k_coeffs/x1_minus_x2_1.txt', x1_minus_x2_coeffs1)

np.savetxt('k_coeffs/x1_2.txt', x1_coeffs2)
np.savetxt('k_coeffs/x2_2.txt', x2_coeffs2)
np.savetxt('k_coeffs/x1_minus_x2_2.txt', x1_minus_x2_coeffs2)

# # load data
# x1_coeffs1 = np.loadtxt('k_coeffs/x1_1.txt')
# x2_coeffs1 = np.loadtxt('k_coeffs/x1_1.txt')
# x1_minus_x2_coeffs1 = np.loadtxt('k_coeffs/x1_minus_x2_1.txt')
#
# x1_coeffs2 = np.loadtxt('k_coeffs/x1_2.txt')
# x2_coeffs2 = np.loadtxt('k_coeffs/x2_2.txt')
# x1_minus_x2_coeffs2 = np.loadtxt('k_coeffs/x1_minus_x2_2.txt')
#
#
# # plot for dof 1
# plt.subplots(figsize=(16, 10))
#
# for i in range(np.shape(x1_coeffs1)[1]):
#     plt.plot(orders_array, x1_coeffs1[:, i])
#     plt.plot(orders_array, x2_coeffs1[:, i])
#     plt.plot(orders_array, x1_minus_x2_coeffs1[:, i])
#
#     plt.scatter(orders_array, x1_coeffs1[:, i])
#     plt.scatter(orders_array, x2_coeffs1[:, i])
#     plt.scatter(orders_array, x1_minus_x2_coeffs1[:, i])
#
# fs_ticks = 16
# ax = plt.gca()
# ax.set_xticklabels(ax.get_xticks(), fontsize=fs_ticks)
# ax.set_yticklabels(ax.get_yticks(), fontsize=fs_ticks)
#
# plt.yscale('log')
#
# fs = 16
# plt.xlabel('Max. order', font='Times New Roman', fontsize=fs)
# plt.ylabel('Coefficients'' values', font='Times New Roman', fontsize=fs)
#
# plt.savefig('stiffness_coeffs1.pdf', dpi=1000, bbox_inches='tight')
# plt.show()
#
# # plot for dof 2
# plt.subplots(figsize=(16, 10))
#
# for i in range(np.shape(x1_coeffs2)[1]):
#     plt.plot(orders_array, x1_coeffs2[:, i])
#     plt.plot(orders_array, x2_coeffs2[:, i])
#     plt.plot(orders_array, x1_minus_x2_coeffs2[:, i])
#
#     plt.scatter(orders_array, x1_coeffs2[:, i])
#     plt.scatter(orders_array, x2_coeffs2[:, i])
#     plt.scatter(orders_array, x1_minus_x2_coeffs2[:, i])
#
# plt.yscale('log')
#
# fs_ticks = 16
# ax = plt.gca()
# ax.set_xticklabels(ax.get_xticks(), fontsize=fs_ticks)
# ax.set_yticklabels(ax.get_yticks(), fontsize=fs_ticks)
#
# fs = 16
# plt.xlabel('Max. order', font='Times New Roman', fontsize=fs)
# plt.ylabel('Coefficients'' values', font='Times New Roman', fontsize=fs)
#
# plt.savefig('stiffness_coeffs2.pdf', dpi=1000, bbox_inches='tight')
# plt.show()

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

            ax.annotate(r'$k_{q_1^2}$', (14, 3.6e5 + 5e5), font='Times New Roman', fontsize=11, color='#9467bd')
            ax.annotate(r'$k_{q_1^3}$', (14, 7.6e6 + 8e6), font='Times New Roman', fontsize=11, color='#828282')
    if dof == 2:
        for i in range(np.shape(x1_coeffs2)[1]):
            ax.plot(orders_array, x1_coeffs2[:, i], linewidth=size_lines)
            ax.plot(orders_array, x2_coeffs2[:, i], linewidth=size_lines)
            ax.plot(orders_array, x1_minus_x2_coeffs2[:, i], linewidth=size_lines)

            ax.scatter(orders_array, x1_coeffs2[:, i], s=size_dots)
            ax.scatter(orders_array, x2_coeffs2[:, i], s=size_dots)
            ax.scatter(orders_array, x1_minus_x2_coeffs2[:, i], s=size_dots)

            ax.annotate(r'$k_{q_2^5}$', (14, 9e6 + 9e6), font='Times New Roman', fontsize=11, color='#d62728')

    ax.grid(True, linewidth=0.5, alpha=0.3)

    fs_ticks = 9
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs_ticks)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs_ticks)

    fs = 12
    ax.set_xlabel(r'Maximum polynomial order $n$ [-]', font='Times New Roman', fontsize=fs)
    ax.set_ylabel(r'Coefficient value $k$' if dof == 1 else '', font='Times New Roman', fontsize=fs)

    ax.set_yscale('log')

plt.subplots_adjust(wspace=0.08)

plt.ylim(1e-10, 1e7 + 8e7)

# plt.yticks([1e-9, 1e-7, 1e-5, 1e-3, 1, 1e1, 1e3, 1e5, 1e7])

plt.savefig('Convergence_coeffs.pdf', format = 'pdf',bbox_inches='tight', dpi=1000)
plt.show()

# # -- order of polynomial --
# n = 10
#
# # stiffness -> use displacements
# vect1 = disp_vect[0][:]
# vect2 = disp_vect[1][:]
#
# # build and solve system
# A = func_matrix(n, vect1, vect2)
# b = measur_matrix(f_ext_vect, acc_vect, vel_vect, disp_vect)
#
# k = np.linalg.pinv(A) @ b
# print(k)
#
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
# '''
# '''
# [[-5.25128087e-07 -1.48366763e-05]
#  [ 3.60000000e+05 -1.06815321e-08]
#  [ 7.60000000e+06 -1.51742790e-06]
#  [-1.30037170e-01  6.06018584e-05]
#  [-4.36039411e+00  4.73728855e-03]
#  [-4.61300240e+01 -2.59344280e-03]
#  [ 3.31615070e-07  1.48370880e-05]
#  [-6.10917914e-06 -2.19326353e-08]
#  [ 4.06553911e-04 -1.06475636e-06]
#  [ 1.67432064e-02 -1.75619498e-05]
#  [ 4.31437856e-02  8.99999999e+06]
#  [-1.83261037e-01  7.12901354e-04]
#  [-7.65447908e-07  1.48317770e-05]
#  [-1.49940719e-04 -2.15381306e-07]
#  [-3.59150533e-03  4.30248183e-05]
#  [ 1.14366382e-01  3.98131832e-03]
#  [ 8.12829685e+00  1.21289015e-01]
#  [ 7.76809082e+01  9.40345764e-01]]
# # maximum order for which all coefficients will be computed
# n_max = 10
#
# # stiffness -> use displacements
# vect1 = disp_vect[0][:]
# vect2 = disp_vect[1][:]
#
# # build and solve system
# A = func_matrix(n_max, vect1, vect2)
# b = measur_matrix(f_ext_vect, acc_vect, vel_vect, disp_vect)
#
# k = np.linalg.pinv(A) @ b
#
# # nonlinear measured forces
# measured_forces = np.transpose(f_ext_vect - M @ acc_vect - C @ vel_vect - K @ disp_vect)
#
# # build plots
# errors_1 = []
# errors_2 = []
#
# for i in range(np.shape(k)[0]):
#     print(i)
#     # choose which k coefficients to include
#     k_ret_1 = np.zeros(np.shape(k)[0])
#     k_ret_2 = np.zeros(np.shape(k)[0])
#
#     # place 1 where you retain k coefficient
#     for j in range(i + 1):
#         k_ret_1[j] = 1
#         k_ret_2[j] = 1
#
#     f_nl_fitted = np.zeros((np.shape(measured_forces)[0], 2))
#     # loop over the Q times measured
#     for l in range(np.shape(f_nl_fitted)[0]):
#
#         row_j = build_row(n_max, vect1, vect2, l)
#         for m in range(np.shape(k)[0]):
#             if k_ret_1[m] == 0:
#                 break
#             f_nl_fitted[l][0] += k[m][0]*row_j[m]*k_ret_1[m]
#             f_nl_fitted[l][1] += k[m][1]*row_j[m]*k_ret_2[m]
#
#     MSE_error_1 = 0
#     MSE_error_2 = 0
#     for o in range(len(f_nl_fitted[:, 0])):
#         MSE_error_1 += (measured_forces[o, 0] - f_nl_fitted[o, 0])**2
#         MSE_error_2 += (measured_forces[o, 1] - f_nl_fitted[o, 1])**2
#
#     MSE_error_1 = MSE_error_1*100/(len(measured_forces[:, 0])*np.var(measured_forces[:, 0]))
#     MSE_error_2 = MSE_error_2*100/(len(measured_forces[:, 1])*np.var(measured_forces[:, 1]))
#
#     errors_1.append(MSE_error_1)
#     errors_2.append(MSE_error_2)
#
# plt.subplots(figsize=(16, 10))
#
# array_plot = np.arange(1, np.shape(k)[0] + 1, 1)
#
# plt.plot(array_plot, errors_1, color='blue', label='$x_1$')
# plt.scatter(array_plot, errors_1, color='blue')
#
# plt.plot(array_plot, errors_2, color='red', label='$x_2$')
# plt.scatter(array_plot, errors_2, color='red')
#
# plt.ylabel('MSEs [-]')
# plt.xlabel('Terms included')
#
# plt.yscale('log')
# plt.legend(loc='best')
#
# plt.annotate('$x_1$', (0, np.min(errors_1)))
# plt.annotate('$x_2$', (n_max, np.min(errors_1)))
# plt.annotate('$(x_1 - x_2)$', (2*n_max - 1, np.min(errors_1)))
#
# plt.vlines(1, ymin=np.min(errors_1), ymax=np.max(errors_1), linestyle='dashed', color='gray')
# plt.vlines(n_max + 1, ymin=np.min(errors_1), ymax=np.max(errors_1), linestyle='dashed', color='gray')
# plt.vlines(2*n_max + 1, ymin=np.min(errors_1), ymax=np.max(errors_1), linestyle='dashed', color='gray')
#
# plt.savefig('MSEs.pdf', dpi=1000, bbox_inches='tight')
# plt.show()
