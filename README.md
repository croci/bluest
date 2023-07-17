<p align="center">
  <img src="readme/bluest_logo.png" alt="logo" width="250"/>
</p>

<p align="center">
    <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
</p>

**BLUEST**: Python package for single- and multi-output multilevel best linear unbiased estimation

## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
  * [Installing with pip](#installing-with-pip)
  * [Installing via containers](#installing-via-containers)
* [Examples and tutorials](#examples-and-tutorials)
* [How to cite](#how-to-cite)
* [Authors and contributors](#authors-and-contributors)
* [License](#license)

## Description
**BLUEST** is a Python package for multilevel and multifidelity best linear unbiased estimation. It implements single- and multi-output best linear unbiased estimators (MLBLUEs) and the multilevel and multifidelity Monte Carlo methods (MLMC and MFMC respectively). The main features of BLUEST are:

- It automatically optimizes the MLBLUE, MLMC and MFMC estimators with respect to the models available and the number of samples, i.e., BLUEST will find the best model combinations to use in the estimations.
- Easy to wrap any computational model through its simple Python API. Note that wrapping non-Python code into Python is generally easy.
- Supports parallel sampling of models via MPI and supports nested MPI communicators (e.g., if the models themselves use MPI).

Note: All estimators are implemented in "multifidelity mode", i.e., they are unbiased estimators for the user-provided high-fidelity model. The user has to ensure that the high-fidelity model is accurate enough in terms of bias/weak error.

See the [**Examples and Tutorials**](#examples-and-tutorials) section below, the [**tutorials folder**](tutorials/README.md), and [**our paper**](#how-to-cite) to get an idea of how to use this software and its capabilities.

## Dependencies and installation

**BLUEST** requires Python >= 3.7 with the following packages: `numpy`, `scipy`, `pybind11`, `networkx` (>=3.0), `mpi4py`, `cvxopt`, `cvxpy`.

**BLUEST** is developed and tested exclusively on Linux machines. It *may* also work on Windows or Mac.

Some examples use additional packages, but these are **not required** to use **BLUEST**. See [**examples folder**](examples/README.md).

### Installing with pip

Make sure pip and setuptools are up-to-date, then simply call:

```bash
> pip install git+https://github.com/croci/bluest.git
```

To install in developer mode you can instead clone the repo (or a specific branch) and then install **BLUEST** in developer mode:

```bash
> git clone https://github.com/croci/bluest.git
> cd bluest
> pip install -e .
```

### Installing via containers

There is a pre-built docker image containing all BLUEST dependencies (including those for the examples) built upon the legacy FEniCS docker images (https://bitbucket.org/fenics-project/docker/src/master/). This image can be downloaded and run as follows:

```bash
> docker run -ti -v $(pwd):/home/fenics/shared croci/bluest:latest
```

If you prefer using singularity you can also convert and run the docker image by typing:

```bash
> singularity build bluest.img croci/bluest:latest
> singularity exec --cleanenv ./bluest.img /bin/bash -l
```

Once the container is running call:

```bash
> pip install --user git+https://github.com/croci/bluest.git
```

## Examples and tutorials

In order to learn how to use **BLUEST**, we recommend starting from the tutorials in the [**tutorials folder**](tutorials/README.md). These tutorials do not require any additional package. 

If you are curious about how to combine PDE models from legacy FEniCS, please check out the [**examples folder**](examples/README.md).

## How to cite

If you use this package in your publications please cite the package as follows:

M. Croci, K. E. Willcox, S. J. Wright, *Multi-output multilevel best linear unbiased estimators via semidefinite programming*. Computer Methods in Applied Mechanics and Engineering 413, 116130 (2023). Article DOI https://doi.org/10.1016/j.cma.2023.116130, Preprint https://export.arxiv.org/abs/2301.07831

```tex
@misc{croci2023multioutput,
    title={Multi-output multilevel best linear unbiased estimators via semidefinite programming},
    author={M. Croci and K. E. Willcox and S. J. Wright},
    year={2023},
    eprint={2301.07831},
    archivePrefix={arXiv}
}
```

## Authors and contributors

This software is currently mantained and developed by Matteo Croci ([email](mailto:mat.mcroci@gmail.com), [website](https://croci.github.io/)), University of Texas at Austin, TX, USA.

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

Note that some files in the [**examples folder**](examples/README.md) and the files in the [**docker folder**](docker/README.md) may be covered by different licences. See LICENCE.md files in those folders.
