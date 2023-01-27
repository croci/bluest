#  Paper examples

This folder contains the scripts that generate the results from our paper:

M. Croci, K. E. Willcox, S. J. Wright, *Multi-output multilevel best linear unbiased estimators via semidefinite programming*. Preprint (2023). URL https://export.arxiv.org/abs/2301.07831

#### Problem 1 - Steady Navier-Stokes flow past two cylinders ([folder](/examples/paper_examples/navier_stokes))

The main **BLUEST** script here is [bluest_ns.py](/examples/paper_examples/navier_stokes/bluest_NS.py). 

This is a multi-output estimation problem in which models are defined via local and global grid refinement.

#### Problem 2 - Hodgkin-Huxley and FitzHugh-Nagumo PDE and ODE models ([folder](/examples/paper_examples/hodgkin-huxley))

The main **BLUEST** script here is [blue_hodgkin_huxley.py](/examples/paper_examples/hodgkin_huxley/blue_hodgkin_huxley.py). 

This is a multi-output estimation problem in which the high-fidelity model is the Hodgkin-Huxley PDE model,
and the low-fidelity models are: the Hodgkin-Huxley ODE model, the FitzHugh-Nagumo PDE and ODE models, grid and timestep refinements.

#### Problem 3 - Random field diffusion PDE with restricted high-fidelity samples ([folder](/examples/paper_examples/restrictions_matern))

The main **BLUEST** script here is [restrictions_matern.py](/examples/paper_examples/restrictions_matern/restrictions_matern.py).

This is a single-output estimation problem in which the maximum number of samples from the two highest-fidelity models is restricted.
This script also computes the **BLUEST** estimation variance and is perhaps the most complicated.
