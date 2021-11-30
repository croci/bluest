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
n_levels  = 6

meshes = [RectangleMesh(MPI.comm_self, Point(0,0), Point(1,1), 2**l, 2**l) for l in range(1, n_levels+1)][::-1]

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

class PoissonProblem(BLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        sample = RNG.randn(5)
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
            f = Expression("e*sin(exp(x[0]*x[1])) + (1-e)*exp(3*cos(x[1]+x[0]))", degree=3, e=samples[i][-1]**2)

            lhs = inner(D*grad(u), grad(v))*dx + u*v*dx
            rhs = f*v*dx

            bcs = get_bcs(V, samples[i])

            solve(lhs == rhs, sol, bcs)

            out[i] = assemble(inner(grad(sol),grad(sol))*dx)

        return out

problem = PoissonProblem(n_levels+1, covariance_estimation_samples=50)
print(problem.get_correlation())

problem.setup_solver(K=3, budget=1.,solver="gurobi")
#problem.setup_solver(K=3, eps=10,solver="gurobi")

out = problem.solve()

#TODO: 1- cost comparison with MFMC
#      2- introduce some NaNs in the covariance

# NOTE: if you want to estimate the complexity of BLUE you should
#       play with the tolerances, e.g.
#Eps = [5.0e-4/2**i for i in range(7)]
