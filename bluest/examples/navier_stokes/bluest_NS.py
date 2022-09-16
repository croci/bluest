from bluest import *
import numpy as np
from dolfin import set_log_level, MPI, LogLevel
from NS import build_space, solve_stokes, solve_navier_stokes, postprocess
import sys

set_log_level(30)
set_log_level(LogLevel.ERROR)

mpiRank = MPI.rank(MPI.comm_world)
mpiSize = MPI.size(MPI.comm_world)

verbose = mpiRank == 0

RNG = np.random.RandomState(mpiRank)

mode = ["full", "part1", "part2"][0]

No = 6 - 3*int(mode != "full")

if verbose: print("Loading models...")

# 0:  (64, f, f). cost = 22.60032949
# 1:  (64, f, c). cost = 15.75815486
# 2:  (64, c, f). cost = 17.81515651
# 3:  (64, c, c). cost = 10.95222405
# 4:  (32, f, f). cost = 6.09950577
# 5:  (32, f, c). cost = 4.2777594
# 6:  (32, c, f). cost = 4.75914333
# 7:  (32, c, c). cost = 2.94629325
# 8:  (16, f, f). cost = 1.70378913
# 9:  (16, f, c). cost = 1.29555189
# 10: (16, c, f). cost = 1.40823723
# 11: (16, c, c). cost = 1

model_data = []
for N_bulk in [64, 32, 16]:
    for N_circle1 in [4*N_bulk, max(16,N_bulk//2)]:
        for N_circle2 in [4*N_bulk, max(16,N_bulk//2)]:

            # Prepare function space, BCs and measure on circle
            W, bndry, ds_circle1, ds_circle2 = build_space(N_circle1, N_circle2, N_bulk, comm=MPI.comm_self)
            model_data.append({"W" : W, "bndry" : bndry, "ds_circle1" : ds_circle1, "ds_circle2" : ds_circle2})
            if verbose: print("Model %d loaded." % len(model_data))

M = len(model_data) # should be 12

if verbose: print("Models loaded.")

class NavierStokesProblem(BLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        nu = 0.0005 + 0.001*RNG.rand()
        U  = 0.1 + 0.4*RNG.rand()
        sample = np.array([U,nu])
        return [sample.copy() for i in range(L)]

    def evaluate(self, ls, samples, N=1):

        L = len(ls)
        out = [[0 for i in range(L)]  for n in range(No)]

        for i in range(L):

            l = ls[i]

            U,nu = samples[i][0], samples[i][1]
            W, bndry, ds_circle1, ds_circle2 = (model_data[l][item] for item in ["W", "bndry", "ds_circle1", "ds_circle2"])

            w = solve_navier_stokes(W, nu, U, bndry, comm=MPI.comm_self)

            C_D1, C_L1, p_diff1 = postprocess(w, nu, U, ds_circle1, 1)
            C_D2, C_L2, p_diff2 = postprocess(w, nu, U, ds_circle2, 2)

            if mode in ["full", "part1"]:
                out[0][i] = C_D1
                out[1][i] = C_L1
                out[2][i] = p_diff1
                if mode == "full":
                    out[3][i] = C_D2
                    out[4][i] = C_L2
                    out[5][i] = p_diff2
            else:
                out[1][i] = C_D2
                out[2][i] = C_L2
                out[3][i] = p_diff2

        return out

if __name__ == "__main__":

    mus = np.array([5.5720, 0.0110, 0.117488, 10.0786, 0.0595, -0.018147]) # the estimated avg of the outputs

    if mode == "part1":   mus = mus[:3]
    elif mode == "part2": mus = mus[3:]
    costs = np.array([model["W"].dim() for model in model_data]); costs = costs/min(costs)
    if verbose: print("Model pseudo-costs: ", costs)

    #problem = NavierStokesProblem(M, n_outputs=No, costs=costs, covariance_estimation_samples=max(mpiSize*50,50))
    #problem.save_graph_data("NS_model_data_full.npz")

    problem = NavierStokesProblem(M, costs=costs, n_outputs=No, datafile="NS_model_data_%s.npz" % mode)

    C = problem.get_covariances()

    if verbose: print("\nRanks of estimated model covariances:", [np.linalg.matrix_rank(c, tol=1.0e-12) for c in C], "\n")
    if verbose: print("Output functional variances:", [c[0,0] for c in C])

    solver = ["BLUE", "MLMC", "MFMC"]

    eps = 0.01*abs(mus); budget=None

    solver_test = True
    if solver_test:
        from time import time
        #FIXME: this eps is not correct isn't it? It should be on the variance
        K = 4; eps = 0.01*abs(mus)/100; budget = max(costs)*100
        OUT = [[],[]]

        out_cvxpy,out_cvxopt,out_ipopt,out_scipy = None, None, None, None
        for i in range(2):
            for solver in ["cvxpy", "ipopt"]:
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

        sys.exit(0)

    out_BLUE = problem.setup_solver(K=3, budget=budget, eps=eps)
    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)

    print("\n\n\n", out_BLUE, "\n\n", out_MLMC, "\n\n", out_MFMC)

    #out = problem.solve()
    #if verbose: print(out)
