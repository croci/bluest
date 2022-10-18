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

Nmax = 1000
Nrestr = [2,5,10][0]

ndofs = np.array([V.dim() for V in function_spaces])
costs = np.concatenate([ndofs]); costs = costs/min(costs);

C = None
load_model_graph_full = os.path.exists("./restrictions_matern_model_data.npz")
load_model_graph_restricted = os.path.exists("./restrictions_matern_model_data_restricted%d.npz" % Nrestr)
load_model_graph = load_model_graph_full and load_model_graph_restricted
if load_model_graph:
    problem = PoissonProblem(M, n_outputs=No, costs=costs, datafile="restrictions_matern_model_data.npz")
    problem_restr = PoissonProblem(M, n_outputs=No, costs=costs, datafile="restrictions_matern_model_data_restricted%d.npz" % Nrestr, skip_projection=True)
else:
    problem_restr = PoissonProblem(M, n_outputs=No, C=C, costs=costs, skip_projection=True, covariance_estimation_samples=Nrestr)
    problem_restr.save_graph_data("restrictions_matern_model_data_restricted%d.npz" % Nrestr)
    problem = PoissonProblem(M, n_outputs=No, C=C, costs=costs, covariance_estimation_samples=Nmax)
    problem.save_graph_data("restrictions_matern_model_data.npz")

C = problem.get_covariance()
true_C = C.copy()
true_dV = problem.get_mlmc_variance()

Cr = problem_restr.get_covariance()
dVr = problem_restr.get_mlmc_variance()

def estimated():
    newC = true_C.copy()
    newdV = true_dV.copy()
    newC[:2,:] = Cr[:2,:]
    newC[:,:2] = Cr[:,:2]
    newdV[:2,:] = dVr[:2,:]

    return newC, newdV

def extrapolated(ndiags=1):
    assert ndiags < M

    d = np.diag(C).copy()
    vals = np.array(problem.evaluate(list(range(M)), problem.sampler(list(range(M))))[0])
    valdiff = abs(vals[:-1]-vals[1:])
    m = 2*np.polyfit(np.log2(ndofs[2:][:3]), np.log2(valdiff[2:][:3]), 1)[0]
    r = (ndofs[2]/ndofs[3])**m
    v1 = (r*d[2]-d[3])/(r-1)
    r = (ndofs[1]/ndofs[2])**m
    v0 = (r*v1-d[2])/(r-1)

    newC = true_C.copy()
    newdV = true_dV.copy()

    newC[0,0] = v0
    newC[1,1] = v1
    d[0] = v0
    d[1] = v1

    for i in range(1,M):
        delv = np.diag(true_dV,i)[2:][:2]
        if len(delv) > 1:
            delv_est = 2.**np.polyval(np.polyfit(np.log2(ndofs[3:][:2]), np.log2(delv[:2]), 1), np.log2(ndofs[1:3]))
            newdV[0,i],newdV[1,i+1] = delv_est[0],delv_est[1]
            newC[0,i] = (d[0] + d[i] - newdV[0,i])/2
            newC[1,i+1] = (d[1] + d[i+1] - newdV[1,i+1])/2
            newC[i,0] = newC[0,i]
            newC[i+1,1] = newC[1,i+1]
        else:
            delv = true_dV[2:4,i]
            r1 = (ndofs[2]/ndofs[3])**m
            r0 = (ndofs[1]/ndofs[2])**m
            newdV[1,i] = (r1*delv[0] - delv[1])/(r1-1)
            newdV[0,i] = (r0*newdV[1,i] - delv[0])/(r0-1)
            newC[0,i] = (d[0] + d[i] - newdV[0,i])/2
            newC[1,i] = (d[1] + d[i] - newdV[1,i])/2
            newC[i,0] = newC[0,i]
            newC[i,1] = newC[1,i]

    newdV[0,(ndiags+1):] = np.nan
    newdV[1,(ndiags+2):] = np.nan
    newC[0,(ndiags+1):] = np.inf
    newC[(ndiags+1):,0] = np.inf
    newC[1,(ndiags+2):] = np.inf
    newC[(ndiags+2):,1] = np.inf

    return newC,newdV

check_all = True
perform_variance_test = False
K = 5
eps = 1.e-2*np.sqrt(true_C[0,0])
max_model_samples = np.inf*np.ones((M,)); max_model_samples[:2] = Nrestr

# First with no restrictions
out_BLUE = problem.setup_solver(K=K, eps=eps, continuous_relaxation=False, max_model_samples=max_model_samples)
out_MLMC = problem.setup_mlmc(eps=eps)
out_MFMC = problem.setup_mfmc(eps=eps)
printout = StringIO()
if verbose: print("\n\n\n", "Exact full covariance with sample restrictions:\n", "BLUE: ", int(out_BLUE[1]["total_cost"]), "\n MLMC: ", int(out_MLMC[1]["total_cost"]), "\n MFMC: ", int(out_MFMC[1]["total_cost"])," ", file=printout)
if perform_variance_test:
    err_ex, err = problem.variance_test(N=100, K=K, eps=eps, continuous_relaxation=False, max_model_samples=max_model_samples)
    errors = [np.array([err_ex[0], err[0]])]

printouts = [printout]
if check_all:
    for i in range(M):
        if verbose: print("\n\nType: ", i, "\n\n", flush=True)
        # just assign 0 to estimated and i>0 to extrapolated
        if i == 0:
            newC,newdV = estimated()
        else:
            newC,newdV = extrapolated(i)

        problem = PoissonProblem(M, C=newC, mlmc_variances=[newdV], costs=costs)

        # Then with restrictions and estimation
        out_BLUE = problem.setup_solver(K=K, eps=eps, continuous_relaxation=False, max_model_samples=max_model_samples)
        out_MLMC = problem.setup_mlmc(eps=eps)
        printouts.append(StringIO())
        if verbose: print("\n", "Trick of type %d:\n" % i, "BLUE: ", int(out_BLUE[1]["total_cost"]), "\n MLMC: ", int(out_MLMC[1]["total_cost"]), " ", file=printouts[-1])
        if perform_variance_test:
            err_ex, err = problem.variance_test(N=50, K=K, eps=eps, continuous_relaxation=False, max_model_samples=max_model_samples)
            errors.append(np.array([err_ex[0],err[0]]))

if verbose:
    for i in range(len(printouts)):
        print(printouts[i].getvalue()[:-2])

    if perform_variance_test:
        errors_vec = np.vstack(errors)
        print("\nEstimator error, theoretical vs actual:\n\n", errors, flush=True)
        np.savez("estimator_errors%d.npz" % Nrestr, errors=errors_vec)
