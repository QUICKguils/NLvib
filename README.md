# Nonlinear Vibration Project

This repository holds all the code written for the project carried out as part
of the Nonlinear Vibration course (AERO0035), academic year 2024-2025.

## Compatibility

The code rely on a few well established external python libraries:
- matplotlib
- numpy
- scipy

The computation of the basins of attraction through multiprocessing require the
installation of the [Pathos library](https://github.com/uqfoundation/pathos).

## Code Structure

The project code consist of one flat directory, containing several python files
that can be executed as is.

- `mplrc.py`: conventional matplotlib parameters, to remain consistent in all the plots.
- `nlsys.py`: definition of the nonlinear system under study.
- `asm.py`: identification of nonlinearities with the acceleration surface method (ASM).
- `rfs_stiffness.py`: quantification of stiffness nonlinearities with the restoring force surface (RFS) method.
- `rfs_damping.py`: quantification of damping nonlinearities with the restoring force surface (RFS) method.
- `shooting.py`: collection of nonlinear solvers based on the shooting method.
- `nfrc.py`: computation of the nonlinear frequency response curves (NFRCs).
- `mmn.py`: computation of the nonlinear normal modes (NNMs).
- `attractor.py`: computation of the basins of attraction.
- `part3.py`: derive results and plots needed to answer the third part of the project.
