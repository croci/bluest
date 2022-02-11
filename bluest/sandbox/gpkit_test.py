#NOTE: requires GPKIT. This makes CVXOPT run, but CVXOPT is slow for this problem. MOSEK takes the same time as for CVXPY. I would just use CVXPY
from gpkit import Variable, Vectorize, VectorVariable, Model

import numpy as np
from time import time

M = 20
K = 20
No = 10

cost = 2.**(2*np.arange(K))
var_mat = np.outer(100*np.random.rand(No), 2.**(-3*np.arange(K)[::-1]))

# DGP requires Variables to be declared positive via `pos=True`.
m = VectorVariable(K, 'm')

obj = cost@m
constraints = [m >= 1] + [var_mat/m <= 10]
problem = Model(obj, constraints)

tic = time()
sol = problem.solve(solver=['cvxopt', 'mosek_conif'][1], verbosity=2)
toc = time()
print(toc-tic)
