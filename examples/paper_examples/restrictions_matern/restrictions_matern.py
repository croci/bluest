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
from mpi4py.MPI import PROD as MPIPROD

set_log_level(30)

comm = MPI.comm_world
mpiRank = MPI.rank(comm)
mpiSize = MPI.size(comm)

mpiBlockSize = 1
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
    def sampler(self, ls, N=1, mean=False):
        L = len(ls)
        lmin = min(ls)
        sample_gauss = RNG.randn(3)/4
        sample = materns[lmin].sample()
        if mean:
            sample_gauss *= 0
            sample.vector()[:] = 1.0

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

slow_run = False
check_all = True
perform_variance_test = True
global_verbose = False
solver_test = False

Nmax = 1000
N_variance_test = 50
maxdiag = 3
K = 4
Nrestr_list = [2, 4, 8, 16]
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

if solver_test:
    from time import time
    #EPS = 0.0018621360085025829 # roughly 5e-3*np.sqrt(true_C[0,0])
    #BUDGET = 4*32*sum(costs)
    EPS = 1e-3*np.sqrt(problem.get_covariance()[0,0])
    BUDGET = 1e4*max(costs)
    Nrestr = 32
    max_model_samples = np.inf*np.ones((M,)); max_model_samples[:2] = Nrestr;
    for K in [3,5,7]:
        for solver in ["cvxopt", "ipopt"]:
            for i in range(2):
                eps = EPS if i==0 else None
                budget = BUDGET if i==1 else None
                print("\n\n", solver, " K = ", K, [" EPS"," BUDGET"][i])
                tic = time()
                out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=True, max_model_samples=max_model_samples, solver=solver)
                toc = time()-tic
                print(toc)
                print(out_BLUE["errors"], out_BLUE["total_cost"])

    import sys; sys.exit(0)

C = problem.get_covariance()
true_C = C.copy()
true_dV = problem.get_mlmc_variance()

deterministic_vals = np.array(problem.evaluate(list(range(M)), problem.sampler(list(range(M)), mean=True))[0])

def estimated(Nrestr, Nmax, Cex, dVex, Cr, dVr):
    newC = Cr.copy()
    newdV = dVr.copy()
    newC[2:,2:] = (newC[2:,2:]*Nrestr + Nmax*Cex[2:,2:])/(Nrestr + Nmax)
    newdV[2:,:] = (newdV[2:,:]*Nrestr + Nmax*dVex[2:,:])/(Nrestr + Nmax)
    return newC, newdV

def extrapolated(Cex,dVex,ndiags=1):
    assert ndiags < M

    d = np.diag(Cex).copy()
    vals = deterministic_vals.copy()
    valdiff = abs(vals[:-1]-vals[1:])
    m = 2*np.polyfit(np.log2(ndofs[2:][:3]), np.log2(valdiff[2:][:3]), 1)[0]
    r = (ndofs[2]/ndofs[3])**m
    v1 = (r*d[2]-d[3])/(r-1)
    r = (ndofs[1]/ndofs[2])**m
    v0 = (r*v1-d[2])/(r-1)

    newC = Cex.copy()
    newdV = dVex.copy()

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
            delv = dVex[2:4,i]
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

EPS = 0.0018621360085025829 # roughly 5e-3*np.sqrt(true_C[0,0])
BUDGET = 4*32*sum(costs)

problem_check = PoissonProblem(M, n_outputs=No, C=None, costs=costs, comm=comm, covariance_estimation_samples=1000, verbose=True)
out_BLUE = problem.setup_solver(K=K, eps=EPS, solver="cvxopt")
cost_ex = out_BLUE['total_cost'] 
out_BLUE = problem.setup_solver(K=K, budget=BUDGET, solver="cvxopt")
eps_ex = out_BLUE['errors']

if mpiRank == 0: print("\n\nBest cost: ", cost_ex, " Best error: ", eps_ex, "\n", flush=True)

cost_ratio1 = sum(costs)/sum(costs[2:])
cost_ratio2 = sum(costs[:2])/sum(costs[2:])

#FIXME: the extrapolated version is deterministic so need to parallelise variance_test
#FIXME: looking at the output something is odd with the output of one of the processors
for Nrestr in Nrestr_list:

    if verbose: print("\n\nNrestr: ", Nrestr, "\n\n", flush=True)

    comm.barrier()

    #Nmax = int(np.ceil(cost_ratio1*Nrestr - cost_ratio2))
    Nmax = 16

    assert Nrestr%mpiBlockSize == 0

    out_eps = {"c_list" : [[] for i in range(maxdiag+2)], "v_list" : [[] for i in range(maxdiag+2)]}
    out_bud = {"c_list" : [[] for i in range(maxdiag+2)], "v_list" : [[] for i in range(maxdiag+2)]}
    outputs = {"eps" : out_eps, "budget" : out_bud}

    #BUDGET = 2*Nrestr*sum(costs)
    if verbose: print("\n\n EPS: ", EPS, " BUDGET: ", BUDGET, "\n\n", flush=True)
    max_model_samples = np.inf*np.ones((M,)); max_model_samples[:2] = Nrestr;

    #FIXME: do it by hand in parallel
    Ntest = int(np.ceil(N_variance_test/ncolors))

    Cex_list  = [0 for nn in range(Ntest)]
    dVex_list = [0 for nn in range(Ntest)]
    Cr_list   = [0 for nn in range(Ntest)]
    dVr_list  = [0 for nn in range(Ntest)]
    if verbose: print("\n\nQUICK RUN!\n\n", flush=True)
    for nn in range(Ntest):
        check = True
        while check:
            try:
                problem = PoissonProblem(M, n_outputs=No, C=None, costs=costs, comm=subcomm, covariance_estimation_samples=Nmax, verbose=global_verbose)
                Cex_list[nn] = problem.get_covariance().copy()
                dVex_list[nn] = problem.get_mlmc_variance().copy()

                problem_restr = PoissonProblem(M, n_outputs=No, C=None, costs=costs, skip_projection=True, covariance_estimation_samples=Nrestr, comm=subcomm, verbose=global_verbose)

                Cr_list[nn]  = problem_restr.get_covariance().copy()
                dVr_list[nn] = problem_restr.get_mlmc_variance().copy()

                Cr = Cr_list[nn].copy()
                dVr = dVr_list[nn].copy()

                Cex = Cex_list[nn].copy()
                dVex = dVex_list[nn].copy()

                for mode in ["eps", "budget"]:

                    if   mode == "eps":
                        eps    = EPS*1; budget = None
                        optimization_solver_params = None
                    elif mode == "budget":
                        budget = BUDGET*1; eps = None
                        optimization_solver_params = {'feastol':1.0e-4}

                    if verbose: print("\n\nRank: %d, Mode: %s. Sample: %d/%d " % (mpiRank, mode, nn+1, Ntest), "Type: start", "\n\n", flush=True)

                    # First with exact quantities
                    problem = PoissonProblem(M, n_outputs=No, C=Cex, mlmc_variances=[dVex], costs=costs, comm=subcomm, verbose=global_verbose)
                    out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)

                    if check_all:
                        for i in range(maxdiag + 1):
                            if verbose: print("\n\nRank: %d, Mode: %s. Sample: %d/%d " % (mpiRank, mode, nn+1, Ntest), "Type: ", i, "\n\n", flush=True)
                            # just assign 0 to estimated and i>0 to extrapolated
                            if i == 0:
                                #newC,newdV = estimated(Nrestr, Nmax, Cex, dVex, Cr, dVr)
                                newC,newdV = Cr,dVr
                            else:
                                newC,newdV = extrapolated(Cex, dVex, i)

                            if Nrestr < M: spd_eps = 1.e-12
                            else:          spd_eps = 5.e-14
                            problem_i = PoissonProblem(M, C=newC, mlmc_variances=[newdV], costs=costs, comm=subcomm, verbose=global_verbose, spg_params={"spd_threshold":spd_eps})

                            # Then with restrictions and estimation
                            out_BLUE = problem_i.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)

                check = False
                #check = not bool(subcomm.allreduce(not check, op=MPIPROD))

            except BLUESTError:
                if subrank == 0: print("\n\nRank: %d, ERROR!\n\n" % mpiRank, flush=True)
                #check = True
                #check = not bool(subcomm.allreduce(not check, op=MPIPROD))

    comm.barrier()

    if slow_run:
        if verbose: print("\n\nSLOW RUN!\n\n", flush=True)
        for nn in range(Ntest):
            Cr = Cr_list[nn].copy()
            dVr = dVr_list[nn].copy()

            Cex = Cex_list[nn].copy()
            dVex = dVex_list[nn].copy()

            for mode in ["eps", "budget"]:

                if   mode == "eps":
                    eps    = EPS*1; budget = None
                    optimization_solver_params = None
                elif mode == "budget":
                    budget = BUDGET*1; eps = None
                    optimization_solver_params = {'feastol':1.0e-4}

                if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: start", "\n\n", flush=True)

                # First with exact quantities
                problem = PoissonProblem(M, n_outputs=No, C=Cex, mlmc_variances=[dVex], costs=costs, comm=subcomm, verbose=global_verbose)
                out_BLUE = problem.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)

                outputs[mode]['c_list'][0].append(out_BLUE["total_cost"])

                #printout = StringIO()
                #if verbose: print("\n\n\n", "Exact full covariance with sample restrictions:\n", "BLUE: ", int(out_BLUE["total_cost"]), " ", file=printout)
                if perform_variance_test:
                    _, err = problem.variance_test(N=N_variance_test, K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)
                    outputs[mode]['v_list'][0].append(err[0])

                #printouts = [printout]
                if check_all:
                    for i in range(maxdiag+1):
                        if verbose: print("\n\nMode: %s. Sample: %d/%d " % (mode, nn+1, Ntest), "Type: ", i, "\n\n", flush=True)
                        # just assign 0 to estimated and i>0 to extrapolated
                        if i == 0:
                            #newC,newdV = estimated(Nrestr, Nmax, Cex, dVex, Cr, dVr)
                            newC,newdV = Cr,dVr
                        else:
                            newC,newdV = extrapolated(Cex, dVex, i)

                        if Nrestr < M: spd_eps = 1.e-12
                        else:          spd_eps = 5.e-14
                        problem_i = PoissonProblem(M, C=newC, mlmc_variances=[newdV], costs=costs, comm=subcomm, verbose=global_verbose, spg_params={"spd_threshold":spd_eps})

                        # Then with restrictions and estimation
                        out_BLUE = problem_i.setup_solver(K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)
                        outputs[mode]['c_list'][i+1].append(out_BLUE["total_cost"])

                        #printouts.append(StringIO())
                        #if verbose: print("\n", "Trick of type %d:\n" % i, "BLUE: ", int(out_BLUE["total_cost"]), " ", file=printouts[-1])
                        if perform_variance_test:
                            _, err = problem_i.variance_test(N=N_variance_test, K=K, eps=eps, budget=budget, continuous_relaxation=False, max_model_samples=max_model_samples, solver="cvxopt", optimization_solver_params=optimization_solver_params)
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
                    for i in range(maxdiag+2):
                        outputs[mode]['c_list'][i] += coloroutputs[ii][mode]['c_list'][i]
                        outputs[mode]['v_list'][i] += coloroutputs[ii][mode]['v_list'][i]

            # NOTE: can use this later to compute expectation and std_dev of the cost and estimator variance.
            np.savez("estimator_sample_data%d.npz" % Nrestr, **outputs)
