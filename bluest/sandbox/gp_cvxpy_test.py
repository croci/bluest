import cvxpy as cp
import numpy as np
from time import time

M = 20
K = 20
No = 10

option = ['SDP', 'GP'][1]

if option == 'GP':
    cost = cp.Parameter(K, pos=True)
    var_list = [cp.Parameter(K, pos=True) for n in range(No)]
    cost.value = 2.**(2*np.arange(K))
    for var in var_list:
        var.value  = 100*np.random.rand()*2.**(-3*np.arange(K)[::-1])

    # DGP requires Variables to be declared positive via `pos=True`.
    m = cp.Variable(K, pos=True)

    obj = cp.Minimize(cost@m)
    constraints = [m >= 1] + [var/m <= 10 for var in var_list]
    problem = cp.Problem(obj, constraints)

    print(cp.installed_solvers())

    for solver in cp.installed_solvers():
        tic = time()
        try: problem.solve(gp=True, solver=solver, warm_start=False, verbose=False)
        except Exception:
            try: problem.solve(solver=solver, warm_start=False, verbose=False)
            except Exception:
                print(solver, ": FAILED")
                continue
        toc = time()
        print(solver, ": ", toc-tic)

else:

    cost = cp.Parameter(K, pos=True)
    var_list = [cp.Parameter(K, pos=True) for n in range(No)]
    cost.value = 2.**(2*np.arange(K))
    for var in var_list:
        var.value  = 100*np.random.rand()*2.**(-3*np.arange(K)[::-1])

    # DGP requires Variables to be declared positive via `pos=True`.
    m = cp.Variable(K, pos=True)
    t = cp.Variable(No, pos=True)

    obj = cp.Minimize(cost@m)
    constraints = [m >= 1, t <= 100] + [cp.bmat([[cp.diag(m), cp.reshape(cp.sqrt(var),(K,1))],[cp.reshape(cp.sqrt(var),(1,K)), cp.reshape(t[i], (1,1))]]) >= 0 for i,var in enumerate(var_list)]
    problem = cp.Problem(obj, constraints)

    print(cp.installed_solvers())

    for solver in cp.installed_solvers():
        tic = time()
        try: problem.solve(solver=solver, warm_start=False, verbose=True)
        except Exception:
            print(solver, ": FAILED")
            continue
        toc = time()
        print(solver, ": ", toc-tic)

