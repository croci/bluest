from dolfin import *
from bluest import *
from numpy.random import RandomState
import numpy as np
import math
import sys
import os
from io import StringIO
from single_matern_field import MaternField,make_nested_mapping
from cvxpy.error import SolverError

set_log_level(30)

comm = MPI.comm_world
mpiRank = MPI.rank(comm)
mpiSize = MPI.size(comm)

mpiBlockSize = 5
if mpiSize%mpiBlockSize == 0:
    color = mpiRank%(mpiSize//mpiBlockSize)
    ncolors = mpiSize//mpiBlockSize
    subcomm = comm.Split(color, mpiRank)
    subrank = MPI.rank(subcomm)
    subsize = MPI.size(subcomm)
else:
    color = 0
    ncolors = 1
    subcomm = comm
    subrank = mpiRank
    subsize = mpiSize

intracomm = comm.Split(subrank == 0, mpiRank)
intrarank = MPI.rank(intracomm)
intrasize = MPI.size(intracomm)

if mpiRank == 0: assert intrarank == 0 and subrank == 0

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

materns = [MaternField(outer_V, inner_V, parameters, RNG=RNG) for outer_V,inner_V in zip(outer_function_spaces,function_spaces)]

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

check_all = True
perform_variance_test = True
global_verbose = False

Nmax = 1000
N_variance_test = 50
K = 4
Nrestr_list = [2, 5, 10, 50, 100]; Nrestr_list = [5]
if not perform_variance_test:
    N_variance_test = 1

C = None

ndofs = np.array([V.dim() for V in function_spaces])
costs = np.concatenate([ndofs]); costs = costs/min(costs);

load_model_graph_full = os.path.exists("./restrictions_matern_model_data.npz")
if load_model_graph_full:
    problem = PoissonProblem(M, n_outputs=No, costs=costs, datafile="restrictions_matern_model_data.npz", verbose=global_verbose)
else:
    problem = PoissonProblem(M, n_outputs=No, C=C, costs=costs, covariance_estimation_samples=Nmax, verbose=global_verbose)
    problem.save_graph_data("restrictions_matern_model_data.npz")

C = problem.get_covariance()
true_C = C.copy()
true_dV = problem.get_mlmc_variance()

deterministic_vals = np.array(problem.evaluate(list(range(M)), problem.sampler(list(range(M))))[0])

def estimated(Cr,dVr):
    newC = true_C.copy()
    newdV = true_dV.copy()
    newC[:2,:] = Cr[:2,:]
    newC[:,:2] = Cr[:,:2]
    newdV[:2,:] = dVr[:2,:]
    return newC, newdV

def extrapolated(ndiags=1):
    assert ndiags < M

    d = np.diag(C).copy()
    vals = deterministic_vals.copy()
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


#FIXME: the extrapolated version is deterministic so need to parallelise variance_test
#FIXME: looking at the output something is odd with the output of one of the processors
for Nrestr in Nrestr_list:

    assert Nrestr%mpiBlockSize == 0

    out_eps = {"c_list" : [[] for i in range(M+2)], "v_list" : [[] for i in range(M+2)]}
    out_bud = {"c_list" : [[] for i in range(M+2)], "v_list" : [[] for i in range(M+2)]}
    outputs = {"eps" : out_eps, "budget" : out_bud}

    EPS = 5.e-3*np.sqrt(true_C[0,0]); BUDGET = 4*Nrestr*sum(costs[:2])
    max_model_samples = np.inf*np.ones((M,)); max_model_samples[:2] = Nrestr;

    #FIXME: do it by hand in parallel
    Ntest = int(np.ceil(N_variance_test/ncolors))

    Cr_list  = [0 for nn in range(Ntest)]
    dVr_list = [0 for nn in range(Ntest)]
    if verbose: print("\n\nQUICK RUN!\n\n", flush=True)
    for nn in range(Ntest):
        check = True
        while check:
            try:
                problem_restr = PoissonProblem(M, n_outputs=No, C=None, costs=costs, skip_projection=True, covariance_estimation_samples=Nrestr, comm=subcomm, verbose=global_verbose)

                Cr_list[nn]  = problem_restr.get_covariance().copy()
                dVr_list[nn] = problem_restr.get_mlmc_variance().copy()

                Cr = Cr_list[nn].copy()
                dVr = dVr_list[nn].copy()

                for mode in ["eps", "budget"]:

                    if   mode == "eps":
                        eps    = EPS*1; budget = None
                    elif mode == "budget":
                        budget = BUDGET*1; eps = None

                    if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: start", "\n\n", flush=True)

                    # First with exact quantities
                    if budget is None:
                        out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt")
                    else:
                        out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params={'feastol':1.0e-4})

                    if check_all:
                        for i in range(M):
                            if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: ", i, "\n\n", flush=True)
                            # just assign 0 to estimated and i>0 to extrapolated
                            if i == 0:
                                #newC,newdV = estimated(Cr,dVr)
                                newC,newdV = Cr,dVr
                            else:
                                newC,newdV = extrapolated(i)

                            problem = PoissonProblem(M, C=newC, mlmc_variances=[newdV], costs=costs, comm=subcomm, verbose=global_verbose)

                            # Then with restrictions and estimation
                            if budget is None:
                                out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt")
                            else:
                                out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params={'feastol':1.0e-4})

                check = False

            except Exception:
                print("\n\nERROR!\n\n", flush=True)

    comm.barrier()

    if verbose: print("\n\nSLOW RUN!\n\n", flush=True)
    for nn in range(Ntest):
        Cr = Cr_list[nn].copy()
        dVr = dVr_list[nn].copy()

        for mode in ["eps", "budget"]:

            if   mode == "eps":
                eps    = EPS*1; budget = None
            elif mode == "budget":
                budget = BUDGET*1; eps = None

            if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: start", "\n\n", flush=True)

            # First with exact quantities
            out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples)
            outputs[mode]['c_list'][0].append(out_BLUE[1]["total_cost"])

            #printout = StringIO()
            #if verbose: print("\n\n\n", "Exact full covariance with sample restrictions:\n", "BLUE: ", int(out_BLUE[1]["total_cost"]), " ", file=printout)
            if perform_variance_test:
                _, err = problem.variance_test(N=N_variance_test, K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples)
                outputs[mode]['v_list'][0].append(err[0])

            #printouts = [printout]
            if check_all:
                for i in range(M):
                    if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: ", i, "\n\n", flush=True)
                    # just assign 0 to estimated and i>0 to extrapolated
                    if i == 0:
                        newC,newdV = estimated(Cr,dVr)
                    else:
                        newC,newdV = extrapolated(i)

                    problem = PoissonProblem(M, C=newC, mlmc_variances=[newdV], costs=costs, comm=subcomm, verbose=global_verbose)

                    # Then with restrictions and estimation
                    out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples)
                    outputs[mode]['c_list'][i+1].append(out_BLUE[1]["total_cost"])

                    #printouts.append(StringIO())
                    #if verbose: print("\n", "Trick of type %d:\n" % i, "BLUE: ", int(out_BLUE[1]["total_cost"]), " ", file=printouts[-1])
                    if perform_variance_test:
                        _, err = problem.variance_test(N=N_variance_test, K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples)
                        outputs[mode]['v_list'][i+1].append(err[0])

            #if verbose:
            #    for i in range(len(printouts)):
            #        print(printouts[i].getvalue()[:-2])

    # NOTE each subcomm now sends their results to mpiRank 0
    if subrank == 0:
        coloroutputs = intracomm.gather(outputs, root=0)

    if mpiRank == 0:
        for ii in range(1,ncolors):
            for mode in ["eps", "budget"]:
                for i in range(M+1):
                    outputs[mode]['c_list'][i] += coloroutputs[ii][mode]['c_list'][i]
                    outputs[mode]['v_list'][i] += coloroutputs[ii][mode]['v_list'][i]

        # NOTE: can use this later to compute expectation and std_dev of the cost and estimator variance.
        np.savez("estimator_sample_data%d.npz" % Nrestr, **outputs)
