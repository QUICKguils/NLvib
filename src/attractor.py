"""Computation of the basins of attraction."""

from functools import partial
import pathlib
from threading import Thread
from typing import NamedTuple
from queue import Queue

import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import Pool
from scipy.integrate import solve_ivp

import nlsys


def compute_ss_amplitude(sys: nlsys.NLSystem, y0, f_ext_tdiv: nlsys.TimeDivision, ivp_span, t_eval):
    """Steady-state amplitude of the system response, for the given ICs."""
    y = solve_ivp(sys.integrand, ivp_span, y0, args=(f_ext_tdiv.w,), t_eval=t_eval).y

    max_dof1 = np.max(y[0,:])
    max_dof2 = np.max(y[1,:])
    min_dof1 = np.min(y[0,:])
    min_dof2 = np.min(y[1,:])

    return np.array([max_dof1, max_dof2, min_dof1, min_dof2])


class AttractorSolution(NamedTuple):
    """Solution of a basin af attraction computation."""
    n_grid: int
    ss_ampl: np.ndarray
    dof1: np.ndarray
    dof2: np.ndarray


def simple_attractor(
        sys: nlsys.NLSystem, f_ext_tdiv: nlsys.TimeDivision, bounds, n_grid: int,
        ivp_span, t_eval):
    """Compute the attractor of the system, at the given excitation frequency."""
    dof1 = np.linspace(bounds[0], bounds[1], n_grid)
    dof2 = np.linspace(bounds[2], bounds[3], n_grid)

    # TODO: size of ss_ampl should not be hardcoded to 4
    ss_ampl = np.zeros((4, len(dof1), len(dof2)))

    for id1, d1 in enumerate(dof1):
        for id2, d2 in enumerate(dof2):
            print(f"Attractor: {id1}, {id2}")
            ss_ampl[:, id1, id2] = compute_ss_amplitude(sys, [d1, d2, 0, 0], f_ext_tdiv, ivp_span, t_eval)

    return AttractorSolution(ss_ampl=ss_ampl, dof1=dof1, dof2=dof2, n_grid=n_grid)


# WARN: multiprocessing is wonky.
# Apparently, it will require a deep refactor of the code to make this works
# with the standary python multiprocessing library.
# For example, one issue is that function attributes like f_ext in the
# nlsys.NLsystem class are not allowed, when using the multiprocessing Pool().
# As a remedy, one can use the multiprocessing utils provided by the pathos
# library. Honestly I don't 100% understand what I am doing here.
def mproc_attractor(
        sys: nlsys.NLSystem, f_ext_tdiv: nlsys.TimeDivision, bounds, n_grid: int,
        ivp_span, t_eval):
    """Compute the attractor of the system, at the given excitation frequency using pathos multiprocessing."""
    dof1 = np.linspace(bounds[0], bounds[1], n_grid)
    dof2 = np.linspace(bounds[2], bounds[3], n_grid)

    ss_ampl = np.zeros((4, len(dof1), len(dof2)))

    def compute_point(sys, f_ext_tdiv, ivp_span, t_eval, d1, d2):
        print(f"mproc solver: {d1:.4f}, {d2:.4f}")
        return compute_ss_amplitude(sys, [d1, d2, 0, 0], f_ext_tdiv, ivp_span, t_eval)

    with Pool() as pool:
        partial_compute = partial(compute_point, sys, f_ext_tdiv, ivp_span, t_eval)
        points = [(d1, d2) for d1 in dof1 for d2 in dof2]
        results = pool.starmap(partial_compute, points)

    result_idx = 0
    for id1, d1 in enumerate(dof1):
        for id2, d2 in enumerate(dof2):
            ss_ampl[:, id1, id2] = results[result_idx]
            result_idx += 1

    return AttractorSolution(ss_ampl=ss_ampl, dof1=dof1, dof2=dof2, n_grid=n_grid)


# NOTE: does not really improves performances, unless a disabled GIL version of python is used.
def mthread_attractor(
        sys: nlsys.NLSystem, f_ext_tdiv: nlsys.TimeDivision, bounds, n_grid: int,
        ivp_span, t_eval):
    """Compute the attractor of the system, using threading."""
    dof1 = np.linspace(bounds[0], bounds[1], n_grid)
    dof2 = np.linspace(bounds[2], bounds[3], n_grid)

    # TODO: size of ss_ampl should not be hardcoded to 4
    ss_ampl = np.zeros((4, len(dof1), len(dof2)))

    # Thread worker function
    def worker(task_queue, result_queue):
        while not task_queue.empty():
            try:
                id1, id2, d1, d2 = task_queue.get_nowait()
                print(f"Attractor(multithreading): id1={id1}, id2={id2}")
                result = compute_ss_amplitude(sys, [d1, d2, 0, 0], f_ext_tdiv, ivp_span, t_eval)
                result_queue.put((id1, id2, result))
            except Queue.Empty:
                break

    task_queue = Queue()
    result_queue = Queue()

    # Populate task queue
    for id1, d1 in enumerate(dof1):
        for id2, d2 in enumerate(dof2):
            task_queue.put((id1, id2, d1, d2))

    # Create and start threads
    num_threads = min(32, n_grid * n_grid)  # Cap number of threads for efficiency
    threads = [Thread(target=worker, args=(task_queue, result_queue)) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Collect results
    while not result_queue.empty():
        id1, id2, result = result_queue.get()
        ss_ampl[:, id1, id2] = result

    return AttractorSolution(ss_ampl=ss_ampl, dof1=dof1, dof2=dof2, n_grid=n_grid)


def compute_attractor(
        sys: nlsys.NLSystem, f_ext_tdiv: nlsys.TimeDivision,
        bounds=[-0.2, 0.2, -0.2, 0.2], n_grid=30, mult=200, solve=simple_attractor):
    """Wrapper function to choose the solving method and set the solving parameters."""

    # Assumed steady-state time span.
    # Heuristically chosen to be a certain multiple of the excitation force period.
    ivp_span = mult * f_ext_tdiv.T * np.array([0, 1])
    ss_span  = mult * f_ext_tdiv.T * np.array([2/3, 1])

    # Time sample of the system response.
    # Should be refined enough to accurately spot the maximas.
    t_eval = np.linspace(*ss_span, mult)

    return solve(sys, f_ext_tdiv, bounds, n_grid, ivp_span, t_eval)


def plot_attractor(sol: AttractorSolution, n_dof=0) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.set_aspect('equal', adjustable='box')

    dof1_mat, dof2_mat = np.meshgrid(sol.dof1, sol.dof2)

    ax.pcolormesh(dof1_mat, dof2_mat, sol.ss_ampl[n_dof], cmap='bwr')

    ax.set_xlabel(r'Initial displacement $x_{1}(0)$ [m]')
    ax.set_ylabel(r'Initial displacement $x_{2}(0)$ [m]')

    fig.show()


def _plot_time_response(sys: nlsys.NLSystem, y0, f_ext_tdiv: nlsys.TimeDivision) -> None:
    """Check the time response of the system, for the given ICs and excitation frequency."""

    mult = 300
    t_span = [0, mult*f_ext_tdiv.T]
    t_sample = np.linspace(*t_span, 2*mult+10)

    y = solve_ivp(sys_forced.integrand, t_span, [-0.04, -0.04, 0, 0], args=(f_ext_tdiv.w,), t_eval=t_sample).y

    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.plot(t_sample, y[0, :])
    fig.show()


if __name__ == '__main__':
    # Build the nonlinear forced system
    f_ext_ampl = 50  # Excitation force amplitude (N)
    sys_forced = nlsys.build_damped_forced_system(nlsys.f_nl, f_ext_ampl)

    # Range of DOF1 and DOF2 initial displacements
    bounds = [-0.2, 0.2, -0.2, 0.2] # mode 1
    # bounds = [-0.2, 0.2, -0.2, 0.2] # mode 2

    # Excitation frequency where bifurcation occurs
    f_ext_tdiv = nlsys.TimeDivision()
    f_ext_tdiv.f = 18  # mode 1
    # f_ext_tdiv.f = 28.7  # mode 2

    # Compute the basins of attraction
    sol_attractor = compute_attractor(
        sys_forced, f_ext_tdiv, bounds,
        n_grid=10,
        solve=mproc_attractor
    )
    plot_attractor(sol_attractor, n_dof=0)
    plot_attractor(sol_attractor, n_dof=1)

    # Maybe think to save them
    if True:
        ROOT_DIR = pathlib.Path(__file__).parent.parent
        OUT_DIR = ROOT_DIR / "out"
        fpath_dof1 = OUT_DIR / "ss_max_dof1.txt"
        fpath_dof2 = OUT_DIR / "ss_max_dof2.txt"
        np.savetxt(fpath_dof1, sol_attractor.ss_ampl[0])
        np.savetxt(fpath_dof2, sol_attractor.ss_ampl[1])
