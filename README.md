Bluest - Python package for Multilevel BLUE estimation
==============================================

Dependencies
------------

- Python >= 3.6, numpy, scipy, matplotlib
- g++ compiler supporting C++11
- Python packages (all installable with pip3 install packagename): pybind11, networkx, mpi4py, cvxopt, cvxpy.
- (Optional, for the examples): fenics >= 2018.1.0 (sudo apt install fenics on Ubuntu 20.04, see https://fenicsproject.org/download/)
- (Optional, not really needed) MOSEK (https://www.mosek.com/), gurobi (https://www.gurobi.com/). These are both commercial softwares, but free academic licenses available.

Installation
-------------

Clone repo, enter the base directory where setup.py is present, then

python3 -m pip install -e .

This installs bluest in development mode, which is better since bluest is in very active development.
