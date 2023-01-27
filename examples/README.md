# Examples

In this folder you will find examples of how to use **BLUEST** with PDE models built using [legacy FEniCS](https://fenicsproject.org/).
We recommend that first-time **BLUEST** users start from the [tutorial folder](/tutorials/README.md) instead.

The extension of these examples to FEniCSx models is likely straightforward.
Note that these examples require further dependencies, listed below.

#### Example 1 [[.py](single_output_example.py)]

Here we solve a simple diffusion equation with random coefficients and boundary conditions, and with a single quantity of interest.
The script includes various estimator tests. For more info, see [blue_models.py](/bluest/blue_models.py).

#### Example 2 [[.py](multi_output_example.py)]
This is the same setup as for Example 1, but with multiple quantities of interest.
The script includes various estimator tests. For more info, see [blue_models.py](/bluest/blue_models.py).

#### Paper examples
In the [paper_examples](/examples/paper_examples/README.md) folder you will find the scripts used to generate the results of the paper

M. Croci, K. E. Willcox, S. J. Wright, *Multi-output multilevel best linear unbiased estimators via semidefinite programming*. Preprint (2023). URL https://export.arxiv.org/abs/2301.07831

## Dependencies

On top of the standard **BLUEST** dependencies, these examples require [legacy FEniCS](https://fenicsproject.org/) (>=2018.1.0), [Gmsh](https://gmsh.info/), `pygmsh`, `matplotlib`, `meshio`.
[Ipopt](https://coin-or.github.io/Ipopt/) and `cyipopt` are also needed for some tests, but are otherwise optional.

We remind that a docker file with all pre-installed dependencies is available, see [main README](/README.md).
