import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import nlsys

ndof = 2

# Linear system matrices
M = np.eye(2)
C = np.array([[3, -1], [-1, 3]])
K = 1E4 * np.array([[2, -1], [-1, 2]])

# Nonlinearities
f_nl = lambda x, x_dot: np.array([3.6e5*x[0]**2 + 7.6e6*x[0]**3, 9e6*x[1]**5])

amplitude = 50
f_ext = lambda w, t: np.array([amplitude*np.sin(w*t), 0])

identity = np.eye(ndof)
null = np.zeros((ndof, ndof))
M_inv = np.linalg.inv(M)
ndof_ss = 2*ndof

L = np.vstack((np.hstack((null, identity)), np.hstack((-M_inv @ K, -M_inv @ C))))
g_nl  = lambda y:    np.concatenate((np.zeros(ndof), M_inv @ f_nl(y[:len(y)//2], y[len(y)//2:])))
g_ext = lambda w, t: np.concatenate((np.zeros(ndof), M_inv @ f_ext(w, t)))

integrand = lambda t, y, w: L @ y - g_nl(y) + g_ext(w, t)

# Choose an excitation frequency
tdiv = nlsys.TimeDivision()
tdiv.f = 18

# Choose time to integrate
integration_time = 100 # [s]

# Initial conditions
y0 = [-0.04, -0.04, 0, 0]

y = solve_ivp(integrand, [0, integration_time], y0, args=(tdiv.w,), dense_output=True).sol
t_sample = np.linspace(0, integration_time, 300)
plt.plot(t_sample, y(t_sample).T[:, :2])

plt.xlabel('time (s)')
plt.ylabel('State vector')

plt.legend(['x1', 'x2', 'x1_dot', 'x2_dot'])

plt.grid()
plt.tight_layout()
plt.show()


def simulate_sol(x1_0, x2_0, f_excitation):
    # Choose an excitation frequency
    tdiv = nlsys.TimeDivision()
    tdiv.f = f_excitation

    # Choose time to integrate
    integration_time = 40 # [s]

    # Initial conditions
    # Inital velocities are always 0
    y0 = [x1_0, x2_0, 0, 0]

    # Integrate until T with specified initial conditions
    y = solve_ivp(integrand, [0, integration_time], y0, args=(tdiv.w,), dense_output=True).sol

    sampled_solution = y(np.linspace(20, 40, 100))
    amplitude = round(np.max(sampled_solution[0]), 4)

    return amplitude


# Set of initial conditions

nb_points_x0 = 110

x1_0_array = np.linspace(-0.15, 0.15, nb_points_x0)
x2_0_array = x1_0_array

# Forcing frequency
forcing_f = 18

contour_values = np.zeros((len(x1_0_array), len(x2_0_array)))

for i in range(len(x1_0_array)):
    for j in range(len(x2_0_array)):
        contour_values[i][j] = abs(simulate_sol(x1_0_array[i], x2_0_array[j], forcing_f))

f_name_txt = 'values_' + str(nb_points_x0) + '.txt'
np.savetxt(f_name_txt, contour_values)

plt.subplots(figsize=(16, 10))

X, Y = np.meshgrid(x1_0_array, x2_0_array)
plt.pcolormesh(X, Y, contour_values, cmap='bwr') # or seismic

fs_ticks = 22

plt.xticks(fontsize=fs_ticks, font='Times New Roman')
plt.yticks(fontsize=fs_ticks, font='Times New Roman')

fs = 16

plt.xlabel(r'Initial displacement $x_{1}(0)$ [m]', fontsize=fs, font='Times New Roman')
plt.ylabel(r'Initial displacement $x_{2}(0)$ [m]', fontsize=fs, font='Times New Roman')

plt.savefig('basins_attraction_' + str(nb_points_x0) + '.pdf', dpi=1000, bbox_inches='tight')
plt.show()
