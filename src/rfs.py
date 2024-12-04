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


M = [[1, 0], [0, 1]]
C = [[3, -1], [-1, 3]]
K = [[2*1e4, -1*1e4], [-1*1e4, 2*1e4]]


def measure_matrix(f_ext_vect, q_dot_dot_vect, q_dot_vect, q_vect):

    # all vectors contain only the selected time interval

    measured_mtrx = np.transpose(f_ext_vect) - np.transpose(q_dot_dot_vect) @ M - np.transpose(q_dot_vect) @ C - np.transpose(q_vect) @ K

    return measured_mtrx


# file_name = 'data_test/group4_test3_1.mat'
file_name = r"C:\Users\guila\unsync\school\master_2\nlVib\group4_test1_1.mat"
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

# -- order of polynomial --
n = 10

# stiffness -> use displacements
vect1 = disp_vect[0][:]
vect2 = disp_vect[1][:]

# damping -> use velocities
vect1 = vel_vect[0][:]
vect2 = vel_vect[1][:]

# build and solve system
A = func_matrix(n, vect1, vect2)
b = measure_matrix(f_ext_vect, acc_vect, vel_vect, disp_vect)

k = np.linalg.pinv(A) @ b
print(k)
'''
[[-5.25128087e-07 -1.48366763e-05]
 [ 3.60000000e+05 -1.06815321e-08]
 [ 7.60000000e+06 -1.51742790e-06]
 [-1.30037170e-01  6.06018584e-05]
 [-4.36039411e+00  4.73728855e-03]
 [-4.61300240e+01 -2.59344280e-03]
 [ 3.31615070e-07  1.48370880e-05]
 [-6.10917914e-06 -2.19326353e-08]
 [ 4.06553911e-04 -1.06475636e-06]
 [ 1.67432064e-02 -1.75619498e-05]
 [ 4.31437856e-02  8.99999999e+06]
 [-1.83261037e-01  7.12901354e-04]
 [-7.65447908e-07  1.48317770e-05]
 [-1.49940719e-04 -2.15381306e-07]
 [-3.59150533e-03  4.30248183e-05]
 [ 1.14366382e-01  3.98131832e-03]
 [ 8.12829685e+00  1.21289015e-01]
 [ 7.76809082e+01  9.40345764e-01]]

# retrieve nonlinear force
idx_max = np.argwhere(k > 1e5)

f_nl = np.zeros((len(vect1), 1))

for time_idx in range(len(vect1)):
    for i in range(np.shape(idx_max)[0]): # for all points satisfying condition
        coeff = k[idx_max[i][0]][idx_max[i][1]]

        f_nl += coeff
'''
