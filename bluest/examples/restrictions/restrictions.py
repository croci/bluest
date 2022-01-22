from dolfin import *
from bluest import *
from numpy.random import RandomState
import numpy as np
import math
import sys
import os

set_log_level(30)

mpiRank = MPI.rank(MPI.comm_world)
mpiSize = MPI.size(MPI.comm_world)

verbose = mpiRank == 0

RNG = RandomState(mpiRank)

dim = 2 # spatial dimension
buf = 1
n_levels  = 7

meshes = [RectangleMesh(MPI.comm_self, Point(0,0), Point(1,1), 2**l, 2**l) for l in range(buf, n_levels+buf)][::-1]

function_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in meshes]

left   = CompiledSubDomain("near(x[0], 0) && on_boundary")
right  = CompiledSubDomain("near(x[0], 1) && on_boundary")
bottom = CompiledSubDomain("near(x[1], 0) && on_boundary")
top    = CompiledSubDomain("near(x[1], 1) && on_boundary")

def get_bcs(V, sample):
    _,b,c,d = sample # a = 0

    b = math.exp(b); c = c**2; d = math.sqrt(math.fabs(d))
    bottom_bexpr = Expression("b*sin(4*DOLFIN_PI*x[0])", b=b, degree=3, mpi_comm=MPI.comm_self)
    left_bexpr   = Expression("c*sin(2*DOLFIN_PI*x[1])",  c=c, degree=3, mpi_comm=MPI.comm_self)
    top_bexpr    = Expression("c + (d-c)*x[0]", c=c, d=d, degree=1, mpi_comm=MPI.comm_self)
    right_bexpr  = Expression("b + (d-b)*x[1]", b=b, d=d, degree=1, mpi_comm=MPI.comm_self)

    left_bc   = DirichletBC(V, left_bexpr, left)
    right_bc  = DirichletBC(V, right_bexpr, right)
    top_bc    = DirichletBC(V, top_bexpr, top)
    bottom_bc = DirichletBC(V, bottom_bexpr, bottom)

    return [left_bc, right_bc, top_bc, bottom_bc]

class PoissonProblem(BLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        sample = RNG.randn(4)/4
        return [sample.copy() for i in range(L)]

    def evaluate(self, ls, samples, N=1):

        L = len(ls)
        out = [0 for i in range(L)]

        for i in range(L):
            if ls[i] > n_levels-1:
                out[i] = sum(samples[i]**2)
                continue

            V = function_spaces[ls[i]]

            u = TrialFunction(V)
            v = TestFunction(V)

            sol = Function(V)

            D = Constant(exp(samples[i][0]))
            f = Constant(1.0) #Expression("e*sin(exp(x[0]*x[1])) + (1-e)*exp(3*cos(x[1]+x[0]))", degree=3, e=samples[i][-1]**2, mpi_comm=MPI.comm_self)

            lhs = inner(D*grad(u), grad(v))*dx + u*v*dx
            rhs = f*v*dx

            bcs = get_bcs(V, samples[i])

            solve(lhs == rhs, sol, bcs)

            out[i] = assemble(sol*sol*dx)
            #out[i] = assemble(exp(sin(sol))*dx)

        return [out]

M = n_levels + 1

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
costs = np.concatenate([ndofs, [1]])

C = build_test_covariance("--O")
load_model_graph = os.path.exists("./restrictions_model_data.npz")
if load_model_graph:
    problem = PoissonProblem(M, costs=costs, datafile="restrictions_model_data.npz")
else:
    problem = PoissonProblem(M, C=C, costs=costs, covariance_estimation_samples=10000)
    problem.save_graph_data("restrictions_model_data.npz")

C = problem.get_covariance()
true_C = C.copy()

C0 = C[2:-1,2:-1].copy()

d = np.diag(C0)
d1 = np.diag(C0,1)
diffv = abs(d[:-1] - d[1:])
vdiff = d[:-1] + d[1:] - 2*d1

dvp = np.polyfit(np.log2(ndofs[3:][:3]), np.log2(diffv[:3]), 1)
vdp = np.polyfit(np.log2(ndofs[3:][:3]), np.log2(vdiff[:3]), 1)

v1 = d[0] + 2.**(np.log2(ndofs[2])*dvp[0] + dvp[1])
v1t = np.median(d[:3]) - 2.**(np.log2(ndofs[2])*dvp[0] + dvp[1])
v0 = d[0] + 2.**(np.log2(ndofs[1])*dvp[0] + dvp[1])
v0t = np.median(d[:3]) - 2.**(np.log2(ndofs[1])*dvp[0] + dvp[1])

delv12 = 2.**(np.log2(ndofs[2])*vdp[0] + vdp[1])
delv01 = 2.**(np.log2(ndofs[1])*vdp[0] + vdp[1])
c12 = 0.5*(v1t + np.median(d[:3]) - delv12)
c01 = 0.5*(v0t + v1t - delv01)

C[0,0] = v0
C[1,1] = v1
C[0,1],C[1,0] = c01,c01
C[1,2],C[2,1] = c12,c12

#C[0,0] = true_C[0,0]
#C[1,1] = true_C[1,1]
#C[0,1],C[1,0] = true_C[0,1],true_C[1,0]
#C[1,2],C[2,1] = true_C[1,2],true_C[2,1]

C[np.isnan(C)] = np.inf

eps = 1.e-3

problem.setup_solver(K=3, eps=eps)

problem = PoissonProblem(M, C=C, costs=costs, skip_projection=True)

problem.setup_solver(K=3, eps=eps)

sys.exit(0)

####################################################################################

complexity_test = False
standard_MC_test = False
comparison_test = False
variance_test = False

if complexity_test:
    eps = 2**np.arange(3,8)
    tot_cost, rate = problem.complexity_test(eps, K=3)
    sys.exit(0)

if standard_MC_test:
    eps = 0.1
    problem.setup_solver(K=3, eps=eps, solver="gurobi")
    out = problem.solve()
    out_MC = problem.solve_mc(eps=eps)
    if verbose: print("BLUE (mu, err, cost):", out)
    if verbose: print("MC   (mu, err, cost):", out_MC)
    sys.exit(0)

if comparison_test:
    budget = 10.;  eps    = None

    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)
    out      = problem.setup_solver(K=M, budget=budget, eps=eps, solver="cvxpy")

    if verbose: print("\nMLMC. Errors: %s. Total cost: %f." % (out_MLMC[1]["errors"], out_MLMC[1]["total_cost"]))
    if verbose: print("MFMC. Errors: %s. Total cost: %f." % (out_MFMC[1]["errors"], out_MFMC[1]["total_cost"]))
    if verbose: print("BLUE. Errors: %s. Total cost: %f.\n\n" % (out[1]["errors"],      out[1]["total_cost"]))

    eps    = 0.25;  budget = None

    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)
    out      = problem.setup_solver(K=M, budget=budget, eps=eps, solver="cvxpy")

    if verbose: print("\nMLMC. Errors: %s. Total cost: %f." % (out_MLMC[1]["errors"], out_MLMC[1]["total_cost"]))
    if verbose: print("MFMC. Errors: %s. Total cost: %f." % (out_MFMC[1]["errors"], out_MFMC[1]["total_cost"]))
    if verbose: print("BLUE. Errors: %s. Total cost: %f." % (out[1]["errors"],      out[1]["total_cost"]))

    sys.exit(0)

if variance_test:
    budget = 1.; eps = None
    err_ex, err = problem.variance_test(budget=budget, eps=eps, K=3, N=100)
    sys.exit(0)

#problem.setup_solver(K=3, budget=10., solver="cvxpy")
problem.setup_solver(K=3, eps=0.1, solver="cvxpy")

out = problem.solve()
if verbose: print(out)
