from dolfin import *
from bluest import *
from numpy.random import RandomState
import numpy as np
import math
import sys

set_log_level(30)

mpiRank = MPI.rank(MPI.comm_world)

RNG = RandomState(mpiRank)

dim = 2 # spatial dimension
buf = 1
n_levels  = 6

meshes = [RectangleMesh(MPI.comm_self, Point(0,0), Point(1,1), 2**l, 2**l) for l in range(buf, n_levels+buf)][::-1]

function_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in meshes]

left   = CompiledSubDomain("near(x[0], 0) && on_boundary")
right  = CompiledSubDomain("near(x[0], 1) && on_boundary")
bottom = CompiledSubDomain("near(x[1], 0) && on_boundary")
top    = CompiledSubDomain("near(x[1], 1) && on_boundary")

def get_bcs(V, sample):
    _,b,c,d,_ = sample # a = 0

    b = math.exp(b); c = c**2; d = math.sqrt(math.fabs(d))
    bottom_bexpr = Expression("b*sin(10*DOLFIN_PI*x[0])", b=b, degree=3)
    left_bexpr   = Expression("c*sin(6*DOLFIN_PI*x[1])",  c=c, degree=3)
    top_bexpr    = Expression("c + (d-c)*x[0]", c=c, d=d, degree=1)
    right_bexpr  = Expression("b + (d-b)*x[1]", b=b, d=d, degree=1)

    left_bc   = DirichletBC(V, left_bexpr, left)
    right_bc  = DirichletBC(V, right_bexpr, right)
    top_bc    = DirichletBC(V, top_bexpr, top)
    bottom_bc = DirichletBC(V, bottom_bexpr, bottom)

    return [left_bc, right_bc, top_bc, bottom_bc]

No = 2

class PoissonProblem(MultiBLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        sample = RNG.randn(5)/5
        return [sample.copy() for i in range(L)]

    def evaluate(self, ls, samples, N=1):

        L = len(ls)
        out = [[0 for i in range(L)]  for n in range(No)]

        for i in range(L):
            if ls[i] > n_levels-1:
                for n in range(No):
                    out[n][i] = sum(samples[i]**2)
                continue

            V = function_spaces[ls[i]]

            u = TrialFunction(V)
            v = TestFunction(V)

            sol = Function(V)

            D = Constant(exp(samples[i][0]))
            f = Expression("e*sin(exp(x[0]*x[1])) + (1-e)*exp(3*cos(x[1]+x[0]))", degree=3, e=samples[i][-1]**2)

            lhs = inner(D*grad(u), grad(v))*dx + u*v*dx
            rhs = f*v*dx

            bcs = get_bcs(V, samples[i])

            solve(lhs == rhs, sol, bcs)

            out[0][i] = assemble(inner(grad(sol),grad(sol))*dx)
            out[1][i] = assemble(exp(sin(sol))*dx)

        return out

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

C = [build_test_covariance("full") for n in range(No)]
problem = PoissonProblem(M, C=C, n_outputs=No, covariance_estimation_samples=50, spg_params={"maxit":10000, "maxfc":10**6, "verbose":False})
print(problem.get_correlation(), "\n")

complexity_test = False
standard_MC_test = False
comparison_test = False

if complexity_test:
    eps = 2**np.arange(3,8)
    tot_cost, rate = problem.complexity_test(eps, K=3)
    sys.exit(0)

if standard_MC_test:
    eps = 0.1
    problem.setup_solver(K=3, eps=eps, solver="gurobi")
    out = problem.solve()
    out_MC = problem.solve_mc(eps=eps)
    print("BLUE (mu, err, cost):", out)
    print("MC   (mu, err, cost):", out_MC)
    sys.exit(0)

if comparison_test:
    budget = 10.;  eps    = None

    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)
    out      = problem.setup_solver(K=M, budget=budget, eps=eps, solver="cvxpy")

    print("\nMLMC. Error: %f. Total cost: %f." % (out_MLMC[1]["error"], out_MLMC[1]["total_cost"]))
    print("MFMC. Error: %f. Total cost: %f." % (out_MFMC[1]["error"], out_MFMC[1]["total_cost"]))
    print("BLUE. Error: %f. Total cost: %f.\n\n" % (out["error"],      out["total_cost"]))

    eps    = 0.25;  budget = None

    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)
    out      = problem.setup_solver(K=M, budget=budget, eps=eps, solver="cvxpy")

    print("\nMLMC. Error: %f. Total cost: %f." % (out_MLMC[1]["error"], out_MLMC[1]["total_cost"]))
    print("MFMC. Error: %f. Total cost: %f." % (out_MFMC[1]["error"], out_MFMC[1]["total_cost"]))
    print("BLUE. Error: %f. Total cost: %f." % (out["error"],      out["total_cost"]))

    sys.exit(0)

problem.setup_solver(K=3, budget=10., solver="cvxpy")
problem.setup_solver(K=3, eps=10, solver="cvxpy")

out = problem.solve()

#TODO: 1- cost comparison with MFMC
#      2- compare actual result with standard MC
