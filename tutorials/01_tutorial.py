from bluest import BLUEProblem
import numpy as np
from scipy.special import gamma

# approximate E[e^Z] for Z being a std Gaussian random variable
# model i defined by truncating the exponential series after n_models - i terms
# high-fidelity model defined exactly as exp(Z).
# lowest fidelity model defined as log(|Z|) for fun

n_models = 5

def exponential_series(x,i):
    ii = np.arange(i+1)
    return np.sum(x**ii/gamma(ii+1)) # Euler Gamma function (recall \Gamma(n+1) = n!

class MyProblem(BLUEProblem):
    def sampler(self, ls):
        L = len(ls)
        Z = np.random.randn()
        samples = [float(Z) for i in range(L)]
        return samples

    def evaluate(self, ls, samples):
        L = len(ls)
        out = [0 for i in range(L)]

        for i in range(L):
            if ls[i] == 0:
                out[i] = np.exp(samples[i])
            elif ls[i] < n_models-1:
                out[i] = exponential_series(samples[i], n_models-ls[i])
            else:
                out[i] = np.log(abs(samples[i]))

        return [out]

# define costs somewhat arbitrarily. If not provided, they will
# be estimated via CPU time, which for this problem makes little sense.
costs = np.array([2**(n_models-i) for i in range(n_models)])

# Default verbose option is True for debugging, you can see everything that goes on
# under the hood if you set it to True. Advised if something breaks!
# 32 (or even 20) samples are enough for application runs. For debugging and Maths papers, set it to 1000.
# These samples won't be re-used. Sample re-use introduces bias and is not implemented here yet.
problem = MyProblem(n_models, costs=costs, covariance_estimation_samples=32, verbose=False)

################################ PART 1 - BASIC USAGE ########################################

# Get covariance and correlation matrix
print("Covariance matrix:\n")
print(problem.get_covariance())
print("\nCorrelation matrix:\n")
print(problem.get_correlation())

# get cost vector
print("\nCost vector:\n")
print(problem.get_costs())

# define statistical error tolerance
eps = 0.01*np.sqrt(problem.get_covariance()[0,0])

# solve with standard MC
sol_MC = problem.solve_mc(eps=eps)
print("\n\nStd MC\n")
print("Std MC solution: ", sol_MC[0], "\nTotal cost: ", sol_MC[2])

# Solve with MLMC
MLMC_data = problem.setup_mlmc(eps=eps)
sol_MLMC = problem.solve_mlmc(eps=eps)

print("\n\nMLMC\n")
print("MLMC data:\n")
for key, item in MLMC_data.items(): print(key, ": ", item)
print("MLMC solution: ", sol_MLMC[0])

# Solve with MFMC
MFMC_data = problem.setup_mfmc(eps=eps)
sol_MFMC = problem.solve_mfmc(eps=eps)

print("\n\nMFMC\n")
print("MFMC data:\n")
for key, item in MFMC_data.items(): print(key, ": ", item)
print("MFMC solution: ", sol_MFMC[0])

# Solve with MLBLUE. K denotes the maximum group size allowed.
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
sol_MLBLUE = problem.solve(K=n_models, eps=eps)

print("\n\nMLBLUE\n")
print("MLBLUE data:\n")
for key, item in MLBLUE_data.items(): print(key, ": ", item)
print("MLBLUE solution: ", sol_MLBLUE[0])

# Alternatively, can specify which groups to use:
groups = [[0], [1], [0,3], [4,5], [0,1,2,3,4]]
MLBLUE_data = problem.setup_solver(groups=groups, eps=eps)

print("\n\nMLBLUE\n")
print("MLBLUE data:\n")
for key, item in MLBLUE_data.items(): print(key, ": ", item)

# MLBLUE is more sensitive than the other methods to integer projection,
# always good to check all methods. This does not require any sampling.
MLMC_data   = problem.setup_mlmc(eps=eps)
MFMC_data   = problem.setup_mfmc(eps=eps)
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
print("\nCost comparison. MLMC: %f, MFMC: %f, MLBLUE: %f" % (MLMC_data["total_cost"], MFMC_data["total_cost"], MLBLUE_data["total_cost"]))

# You can also set a budget rather than a tolerance, e.g.
budget = 100*max(costs) # budget corresponding to 100 std MC samples
# same syntax for MLMC and MFMC. Never provide both eps and budget at the same time
MLBLUE_data = problem.setup_solver(K=n_models, budget=budget)

#NOTE: no need for calling setup_solver, can call solve directly, although I do not recommend it
#      as it is always better to check first.

# IMPORTANT: OPTIMIZATION SOLVER
# Most of the time this is not needed, but it may happen that the SDP solver fails
# due to a lack of feature/bug in CVXPY/CVXOPT for which the optimizer does not
# recognise it has found a solution, and it keeps iterating until failure.
# If this happens, you can tweak the optimization solver parameters as follows:
cvxopt_params = {
        "abstol" : 1.e-7,
        "reltol" : 1.e-4,
        "maxiters" : 1000, # called max_iters for "cvxpy"
        "feastol" : 1.0e-6,
}
MLBLUE_data = problem.setup_solver(K=n_models, budget=budget, solver="cvxopt", optimization_solver_params=cvxopt_params)
# Changing feastol is typically enough, sometimes you can increase the other tolerances or reduce maxiters

################################ PART 2 - PARALLELIZATION #######################################

# Everything runs in parallel with MPI. Simply call the script with e.g.:
#
# mpiexec -n NPROCS python3 minimal.py 
#
# And all the sampling will be split across NPROCS workers and occur in parallel (N.B. number of pilot samples will be
# rounded up so that it is a multiple of NPROCS, but the number of online samples will not be increased).
#
# On computing nodes it is often best to set the OMP_NUM_THREADS flag to avoid unwanted multithreading
#
# OMP_NUM_THREADS=1 mpiexec -n NPROCS python3 minimal.py 
#
# NOTE: BLUEST uses mpi4py for MPI parallelization.
# NOTE: Always make sure that the random number generator you use is thread-safe and that you are using independent
#       generators for each worker by using skipahead functionalities (if implemented) or different random seeds.
#       e.g. in Python:
#       
from mpi4py import MPI
from numpy.random import RandomState

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

RNG = RandomState(mpiRank) # sets a different seed for each worker's random number generator
RNG.randn()
#
#       BLUEST will handle the rest of the parallelization for you so that you do not have to worry about it apart from the RNG.
#
# NOTE: If your models also use MPI, then you need to be careful. Simplest option is to set
#       the MPI communicator of your models to MPI_COMM_SELF so that each worker loads the full model
#       and deadlocks are avoided. However, BLUEST does support nested MPI communicators: ask me if you need this.

################################ PART 3 - ADVANCED USAGE ########################################

# Nothing BLUEST-specific in the next two lines: just creating/cleaning up a temporary directory for this script
import shutil; shutil.rmtree("/tmp/mlblue/", ignore_errors=True) # cleaning up from previous runs
import os; os.makedirs("/tmp/mlblue", exist_ok=True) # creating temporary directory for the script

# Can save offline estimation data and load it later.
problem.save_graph_data("/tmp/mlblue/minimal_data.npz")
problem = MyProblem(n_models, datafile="/tmp/mlblue/minimal_data.npz")

# Can overload cost vector when loading.
problem = MyProblem(n_models, costs=costs, datafile="/tmp/mlblue/minimal_data.npz")

# Can avoid offline sampling if covariance and costs are known.
C = np.random.randn(n_models, n_models); C = C.T@C;
problem = MyProblem(n_models, C = C, costs=costs, verbose=False)

# ONLY IF YOU WANT TO USE MLMC:
# For MLMC, however, you also want to provide the variance of model differences V[P_i-P_j]:
# Only upper triangular block is needed with dV[i,j] = V[P_i - P_j]. The rest must be set to NaN
dV = np.nan*np.ones_like(C)
for i in range(n_models):
    for j in range(i+1,n_models):
        # V[P_i - P_j] = V[P_i] + V[P_j] - 2*C(P_i, P_j)
        dV[i,j] = C[i,i] + C[j,j] - 2*C[i,j]

problem = MyProblem(n_models, C = C, mlmc_variances=dV, costs=costs, verbose=False)
# When estimating the covariance with pilot samples, the MLMC variances will also be estimated
# and they will also be saved to file with save_graph_data and loaded automatically afterwards.
# you can access them with
problem.get_mlmc_variance()

# Can ask to re-estimate some entries by setting entries of C to NaN. Note that
# all models might need to be sampled so you might as well re-estimate everything
C[0,0] = np.nan
problem = MyProblem(n_models, C = C, costs=costs, covariance_estimation_samples=32, verbose=False)

# Can exclude model groups also by setting covariance entries to inf, e.g.
C = np.nan*C # setting all entries of C to NaN, they will be re-estimated
# The covariance between Model 0 and Model 1 won't be estimated
# and the two models will never be sampled together
# Model set might be pruned after this in case some models become useless
C[0,1] = np.inf; C[1,0] = np.inf
problem = MyProblem(n_models, C = C, costs=costs, covariance_estimation_samples=32)

# If a covariance matrix is given, this will be projected to be spd.
# You can skip this projection at your own risk by setting skip_projection=True
problem = MyProblem(n_models, C = C, costs=costs, covariance_estimation_samples=32, skip_projection=True)

# IMPORTANT: if C is singular, it will be projected to be spd by default.
# In this case the best course of action is to remove a model.
# Alternatively you can use skip_projection=True and hope for the best.
# This is one of the cases in which you do want to check whether MLMC of MFMC
# do better than MLBLUE.

# NOT SO IMPORTANT:
# You can tweak the parameters for the spd projection, but it is almost never needed
spg_params = {"maxit" : 10000,
              "maxfc" : 10000**2,
              "verbose" : False,
              "spd_threshold" : 5.0e-14,  # minimum eigenvalue
              "eps"     : 1.0e-10,        # optimization solver tolerance
              "lmbda_min" : 10.**-30,
              "lmbda_max" : 10.**30,
              "linesearch_history_length" : 10,
             }
# In some very rare cases with near-singular covariance matrices spd_threshold and eps might require some tweaking. 
# Best to ask me if you get here.
problem = MyProblem(n_models, C = C, costs=costs, covariance_estimation_samples=32, spg_params=spg_params)

# You can ask BLUEST to save all sample outputs (e.g. snapshots), and their input parameters
C = np.random.randn(n_models, n_models); C = C.T@C;
problem = MyProblem(n_models, C = C, costs=costs, samplefile="/tmp/mlblue/snapshots.npz", verbose=False)

# all samples will be saved in different npz files with naming convention snapshots$MODELS.npz
# where $MODELS corresponds to which models are sample together. New samples will always be appended.
sol_MLBLUE = problem.solve(K=n_models, eps=eps)

# You can avoid saving the pilot samples by setting a samplefile later
problem = MyProblem(n_models, costs=costs, covariance_estimation_samples=32, verbose=False)
problem.params["samplefile"] = "/tmp/mlblue/snapshots.npz"

# You can change samplefile name as you go
problem.params["samplefile"] = "/tmp/mlblue/snapshots_MLMC.npz" # set specific filename for MLMC
sol_MLMC = problem.solve_mlmc(eps=eps)
problem.params["samplefile"] = "/tmp/mlblue/snapshots.npz" # reset to default

#NOTE: batch sampling (taking multiple samples in one go) is supported, but untested,
#      it is sufficient to define:
#      def sampler(self, ls, Nbatch=1):
#      def evaluate(self, ls, samples, Nbatch=1):
#          return [out]
#      where out[i][n] is the n_th sample in the batch for model i
#      let me know if you are interested in this functionality and if you find bugs

################################ PART 4 - MULTIPLE OUTPUTS ########################################

# Now consider two outputs: e^Z and (e^Z)^2
n_outputs = 2

class MyMultiOutputProblem(BLUEProblem):
    def sampler(self, ls):
        L = len(ls)
        Z = RNG.randn()
        samples = [float(Z) for i in range(L)]
        return samples

    def evaluate(self, ls, samples):
        L = len(ls)
        #NOTE the different definition of the output. This is consisistent with the single-output case if you check
        out = [[0 for i in range(L)] for n in range(n_outputs)]

        pw = [1,2]
        for i in range(L):
            for n in range(n_outputs):
                #NOTE the indexing of out
                if ls[i] == 0:
                    out[n][i] = np.exp(samples[i])**pw[n]
                elif ls[i] < n_models-1:
                    out[n][i] = exponential_series(samples[i], n_models-ls[i])**pw[n]
                else:
                    # let's reuse the same quantity for both outputs, why not?
                    out[n][i] = np.log(abs(samples[i]))

        # note that now we do not need to wrap out into a list as in the single output case
        return out

# NOTE: costs are the same as before, i.e. we assume that each model evaluation costs the same for all QoI that it outputs.
#       different costs vectors currently not supported. Ask me if needed.
costs = np.array([2**(n_models-i) for i in range(n_models)])

# Now need to specify how many outputs
problem = MyMultiOutputProblem(n_models, n_outputs=n_outputs, costs=costs, covariance_estimation_samples=32, verbose=True)

# same as before with prescribed budget
budget = 1000*max(costs) # budget corresponding to 1000 std MC samples
MLBLUE_data = problem.setup_solver(K=n_models, budget=budget)

# can prescribe a single statistical error tolerance for all outputs:
eps = 0.01*np.sqrt(problem.get_covariance(0)[0,0])
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
# or a different tolerance for different outputs
eps = [0.01*np.sqrt(problem.get_covariance(n)[0,0]) for n in range(n_outputs)]
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
#MLBLUE_sol = problem.solve(K=n_models, eps=eps)

# Can prescribe a single group for all
groups = [[0], [1], [0,3], [4,5], [0,1,2,3,4]]
MLBLUE_data = problem.setup_solver(groups=groups, eps=eps)

# Or different groups for each
groups_0 = [[0], [1], [0,3], [4,5], [0,1,2,3,4]]
groups_1 = [[0], [1], [1,3], [3,5], [0,1,3,4]]
multi_groups = [groups_0, groups_1]
#NOTE: untested, let me know if it breaks
MLBLUE_data = problem.setup_solver(multi_groups=multi_groups, eps=eps)

# now need to prescribe a model covariance (and mlmc_variances) for each QoI:
# (can use nan and inf as before)
C0 = np.random.randn(n_models, n_models); C0 = C0.T@C0;
C1 = np.random.randn(n_models, n_models); C1 = C1.T@C1;
C = [C0, C1]
dV = [np.nan*np.ones_like(c) for c in C]
for n in range(n_outputs):
    for i in range(n_models):
        for j in range(i+1,n_models):
            dV[n][i,j] = C[n][i,i] + C[n][j,j] - 2*C[n][i,j]

# each covariance matrix will be projected to be spd
problem = MyMultiOutputProblem(n_models, n_outputs=n_outputs, C = C, mlmc_variances=dV, costs=costs, verbose=False)

# can get them all
C = problem.get_covariances()
R = problem.get_correlations()
dV = problem.get_mlmc_variances()
# or one at a time
C0 = problem.get_covariance(0)
R0 = problem.get_correlation(0)
dV0 = problem.get_mlmc_variance(0)
C1 = problem.get_covariance(1)
R1 = problem.get_correlation(1)
dV1 = problem.get_mlmc_variance(1)

# Can still store all the samples for all the outputs to file
problem = MyMultiOutputProblem(n_models, n_outputs=n_outputs, C = C, costs=costs, samplefile="/tmp/mlblue/snapshots_multi.npz", verbose=False)
# or only some of the outputs. Just put them in a list.
outputs_to_save = [1]
problem = MyMultiOutputProblem(n_models, n_outputs=n_outputs, C = C, costs=costs, outputs_to_save = outputs_to_save, samplefile="/tmp/mlblue/snapshots_multi_again.npz", verbose=False)

# NOTE: everything else is the same as for the single-output case. Any questions please ask!
