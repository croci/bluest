from dolfin import *
from bluest import *
from numpy.random import RandomState
import numpy as np
import math
import sys
import os
from io import StringIO
from single_matern_field import MaternField,make_nested_mapping

set_log_level(30)

mpiRank = MPI.rank(MPI.comm_world)
mpiSize = MPI.size(MPI.comm_world)

verbose = mpiRank == 0

RNG = RandomState(mpiRank)

dim = 2 # spatial dimension
buf = 1
n_levels  = 7
No = 1

outer_meshes = [RectangleMesh(MPI.comm_self, Point(0,0), Point(1,1), 2**(l+1), 2**(l+1)) for l in range(buf, n_levels+buf)][::-1]
meshes = [RectangleMesh(MPI.comm_self, Point(0,0), Point(1,1), 2**l, 2**l) for l in range(buf, n_levels+buf)][::-1]

outer_function_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in outer_meshes]
function_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in meshes]

k = 1
parameters = {"lmbda"    : 0.2, # correlation length 
              "avg"      : 1.0, # mean
              "sigma"    : 0.3, # standard dev.
              "lognormal_scaling" : True,
              "nu"       : 2*k-dim/2} # smoothness parameter

materns = [MaternField(outer_V, inner_V, parameters) for outer_V,inner_V in zip(outer_function_spaces,function_spaces)]

nested_mappings = [[0 for j in range(n_levels)] for i in range(n_levels)]
for i in range(n_levels):
    for j in range(i+1,n_levels):
        nested_mappings[i][j] = make_nested_mapping(function_spaces[i], function_spaces[j])


def get_bcs(V, sample):
    _,b,c,d = sample # a = 0

    bc_expr = Expression('2.0*b*x[0]*x[1] + c*x[0]*x[0] + d*x[1]*x[1] + 1.0', b=b, c=c, d=d, degree=2, mpi_comm=MPI.comm_self)

    bc = DirichletBC(V, bc_expr, "on_boundary")

    return [bc]

def compute_solution(V, sample):
    u = TrialFunction(V)
    v = TestFunction(V)

    sol = Function(V)

    D = exp(sample[0])
    f = Expression('exp(-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)))', degree=4, mpi_comm=MPI.comm_self)

    lhs = inner(D*grad(u), grad(v))*dx + u*v*dx
    rhs = f*v*dx

    bcs = get_bcs(V, sample)

    solve(lhs == rhs, sol, bcs)

    return sol

class PoissonProblem(BLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        lmin = min(ls)
        sample_gauss = RNG.randn(3)/4
        sample = materns[lmin].sample()
        samples = []
        for i in range(L):
            if ls[i] == lmin: samples.append([sample] + list(sample_gauss.copy()))
            else:
                ss = Function(function_spaces[ls[i]])
                ss.vector()[:] = sample.vector()[nested_mappings[lmin][ls[i]]]
                samples.append([ss] + list(sample_gauss.copy()))

        return samples

    def evaluate(self, ls, samples, N=1):

        L = len(ls)
        out = [[0 for i in range(L)] for n in range(No)]

        for i in range(L):

            l = ls[i]
            V = function_spaces[l]

            sol = compute_solution(V, samples[i])

            Pf = assemble(sol*sol*dx)

            out[0][i] = Pf

        return out

M = n_levels

def build_test_covariance(string="full"):
    C = np.nan*np.ones((M,M))
    if string == "--O":
        C[0,2:] = np.inf
        C[2:,0] = np.inf
        C[1,3:] = np.inf
        C[3:,1] = np.inf
    elif string == "gaps":
        C[0,2] = np.inf
        C[2,0] = np.inf
        C[2,6] = np.inf
        C[6,2] = np.inf
    elif string == "MLMC":
        for i in range(M):
            C[i,i+2:] = np.inf
            C[i+2:,i] = np.inf
    elif string == "bad":
        C[0,2:] = np.inf
        C[2:,0] = np.inf
        for i in range(M):
            if i > 0 and i + 2 < M:
                C[i,i+1] = np.inf
                C[i+1,i] = np.inf
            if i + 3 < M:
                C[i,i+3] = np.inf
                C[i+3,i] = np.inf
    else: pass
    return C

ndofs = np.array([V.dim() for V in function_spaces])
costs = np.concatenate([ndofs]); costs = costs/min(costs);

#C = [build_test_covariance("--O") for n in range(No)]
C = None
load_model_graph = os.path.exists("./restrictions_matern_model_data.npz")
if load_model_graph:
    problem = PoissonProblem(M, n_outputs=No, costs=costs, datafile="restrictions_matern_model_data.npz")
else:
    problem = PoissonProblem(M, n_outputs=No, C=C, costs=costs, covariance_estimation_samples=1000)
    problem.save_graph_data("restrictions_matern_model_data.npz")

C = problem.get_covariance()
true_C = C.copy()
true_dV = problem.get_mlmc_variance().copy()

d = np.diag(C)
delv = problem.get_mlmc_variance()[0,1:]
covest_ex = (d[:-1] + d[1:] - delv)/2

vals = np.array(problem.evaluate(list(range(M)), problem.sampler(list(range(M))))[0])
valdiff = abs(vals[:-1]-vals[1:])
m = 2*np.polyfit(np.log2(ndofs[2:][:3]), np.log2(valdiff[2:][:3]), 1)[0]

r = (ndofs[2]/ndofs[3])**m
v0 = (r*d[2]-d[3])/(r-1)
v1 = v0
delv_est = 2.**np.polyval(np.polyfit(np.log2(ndofs[3:][:3]), np.log2(delv[:3]), 1), np.log2(ndofs[1:3]))
covest_est = np.array([v0 + v1 - delv_est[0], v1 + d[2] - delv_est[1]])/2

c01,c12 = covest_est

C[0,0] = v0
C[1,1] = v1
C[0,1],C[1,0] = c01,c01
C[1,2],C[2,1] = c12,c12

C[0,2:] = np.inf
C[2:,0] = np.inf
C[1,3:] = np.inf
C[3:,1] = np.inf

C[np.isnan(C)] = np.inf
C_est = C.copy()

dV_est = true_dV.copy()
dV_est[0,1] = delv_est[0]
dV_est[1,2] = delv_est[1]
dV_est[0,2:] = np.nan
dV_est[2:,0] = np.nan
dV_est[1,3:] = np.nan
dV_est[3:,1] = np.nan

eps = 1.e-3*np.sqrt(C[0,0])

# First with no restrictions
out_BLUE = problem.setup_solver(K=4, eps=eps, continuous_relaxation=False)
out_MLMC = problem.setup_mlmc(eps=eps)
out_MFMC = problem.setup_mfmc(eps=eps)
printout = StringIO()
if verbose: print("\n\n\n", "Exact, no restrictions:\n", "BLUE: ", int(out_BLUE[1]["total_cost"]), "\n MLMC: ", int(out_MLMC[1]["total_cost"]), "\n MFMC: ", int(out_MFMC[1]["total_cost"])," ", file=printout)

problem = PoissonProblem(M, C=C_est, mlmc_variances=[dV_est], costs=costs)

# Then with restrictions and estimation
out_BLUE = problem.setup_solver(K=4, eps=eps, continuous_relaxation=False)
out_MLMC = problem.setup_mlmc(eps=eps)
out_MFMC = problem.setup_mfmc(eps=eps)
printout2 = StringIO()
if verbose: print("\n", "Estimated, with restrictions:\n", "BLUE: ", int(out_BLUE[1]["total_cost"]), "\n MLMC: ", int(out_MLMC[1]["total_cost"]), "\n MFMC: ", int(out_MFMC[1]["total_cost"]), " ", file=printout2)
#np.savez("samples.npz", out_BLUE=out_BLUE, out_MLMC=out_MLMC, out_MFMC=out_MFMC, costs=costs)

if verbose: print("\n\n", out_BLUE, "\n\n", out_MLMC, "\n\n", out_MFMC, "\n\n", flush=True)

solver_test = False
if solver_test:
    from time import time
    K = 3; eps = 1.e-2*np.sqrt(C[0,0]); budget = 1e5
    OUT = [[],[]]

    out_cvxpy,out_cvxopt,out_ipopt,out_scipy = None, None, None, None
    for i in range(2):
        for solver in ["cvxopt", "cvxpy", "ipopt", "scipy"]:
            tic = time()
            if i == 0: out = problem.setup_solver(K=K, budget=budget, solver=solver, continuous_relaxation=True, optimization_solver_params={'feastol':1.e-3, 'abstol':1e-5, 'reltol':1e-2})[1]
            else:      out = problem.setup_solver(K=K, eps=eps, solver=solver, continuous_relaxation=True, optimization_solver_params={'feastol':1.e-7})[1]
            toc = time() - tic
            out = np.array([max(out['errors']), out['total_cost'], toc])
            OUT[i].append(out)

        OUT[i] = np.vstack(OUT[i])

    for i in range(2):
        print("\terrors\t   total cost\t   time\n")
        print(OUT[i], "\n")

    import sys; sys.exit(0)


# Then with restrictions and no estimation
C = true_C.copy()
C[0,2:] = np.inf
C[2:,0] = np.inf
C[1,3:] = np.inf
C[3:,1] = np.inf
dV = problem.get_mlmc_variance().copy()
dV[0,2:] = np.nan
dV[2:,0] = np.nan
dV[1,3:] = np.nan
dV[3:,1] = np.nan

problem = PoissonProblem(M, C=C, mlmc_variances=[dV], costs=costs)
out_BLUE = problem.setup_solver(K=4, eps=eps, continuous_relaxation=False)
out_MLMC = problem.setup_mlmc(eps=eps)
out_MFMC = problem.setup_mfmc(eps=eps)

# NOTE: printing things in the right order, be careful
if verbose:
    print(printout.getvalue()[:-2])
    print("\n", "Exact, with restrictions:\n", "BLUE: ", int(out_BLUE[1]["total_cost"]), "\n MLMC: ", int(out_MLMC[1]["total_cost"]), "\n MFMC: ", int(out_MFMC[1]["total_cost"]))
    print(printout2.getvalue()[:-2])

C_hat = problem.get_covariance(); C_hat[np.isnan(C_hat)] = 0
C[~np.isfinite(C)] = 0
rel_err = np.linalg.norm(C-C_hat)/np.linalg.norm(C)
err = np.sqrt((v0-true_C[0,0])**2 + (v1 - true_C[1,1])**2 + (c01 - true_C[0,1])**2 + (c12 - true_C[1,2])**2)

if verbose: print("\n\nCovariance estimation errors. Matrix rel err: ", rel_err, "Entries abs err: ", err)

perform_variance_test = False
if perform_variance_test:
    err_ex, err = problem.variance_test(eps=eps, K=4, N=100)
    if verbose: print(err_ex, err)
