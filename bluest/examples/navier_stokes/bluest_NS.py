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

No = 6

if verbose: print("Loading models...")

model_data = []
for N_bulk in [64, 32, 16]:
    for N_circle1 in [4*N_bulk, max(16,N_bulk//2)]:
        for N_circle2 in [4*N_bulk, max(16,N_bulk//2)]:

            # Prepare function space, BCs and measure on circle
            W, bndry, ds_circle1, ds_circle2 = build_space(N_circle1, N_circle2, N_bulk, comm=MPI.comm_self)
            model_data.append({"W" : W, "bndry" : bndry, "ds_circle1" : ds_circle1, "ds_circle2" : ds_circle2})
            if verbose: print("Model %d loaded." % len(model_data))

M = len(model_data) # should be 24

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

            out[0][i] = C_D1
            out[1][i] = C_L1
            out[2][i] = p_diff1

            out[3][i] = C_D2
            out[4][i] = C_L2
            out[5][i] = p_diff2

        return out

if __name__ == "__main__":

    mus = np.array([5.5720, 0.0110, 0.117488, 10.0786, 0.0595, -0.018147])

    #costs = np.load("NS_costs.npz")["times"]
    #problem = NavierStokesProblem(M, n_outputs=No, costs=costs, covariance_estimation_samples=max(mpiSize*50,50))
    #problem.save_graph_data("bluest-NS.npz")

    problem = NavierStokesProblem(M, n_outputs=No, datafile="bluest-NS.npz")

    C = problem.get_covariances()

    if verbose: print("\nRanks of estimated model covariances:", [np.linalg.matrix_rank(c, tol=1.0e-12) for c in C], "\n")
    if verbose: print("Output functional variances:", [c[0,0] for c in C])

    #import sys; sys.exit(0)

    solver = ["BLUE", "MLMC", "MFMC"]

    #eps = None; budget = 10
    eps = 0.5*abs(mus); budget=None

    out_BLUE = problem.setup_solver(K=3, budget=budget, eps=eps)
    if verbose: print("\nBLUE. Errors: %s. Total cost: %f." % (out_BLUE[1]["errors"], out_BLUE[1]["total_cost"]))
    out_MLMC = problem.setup_mlmc(budget=budget, eps=eps)
    if verbose: print("\nMLMC. Errors: %s. Total cost: %f." % (out_MLMC[1]["errors"], out_MLMC[1]["total_cost"]))
    out_MFMC = problem.setup_mfmc(budget=budget, eps=eps)
    if verbose: print("\nMFMC. Errors: %s. Total cost: %f." % (out_MFMC[1]["errors"], out_MFMC[1]["total_cost"]))

    out = problem.solve()
    if verbose: print(out)
    MPI.comm_world.barrier()