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
        sample = np.random.randn()
        return [float(sample) for i in range(L)]

    def evaluate(self, ls, samples):
        L = len(ls)
        out = [0 for i in range(L)]

        for i in range(L):
            if i == 0:
                out[i] = np.exp(samples[i])
            elif i < n_models-1:
                out[i] = exponential_series(samples[i], n_models-i)
            else:
                out[i] = np.log(abs(samples[i]))

        return [out]


# define costs somewhat arbitrarily. If not provided, they will
# be estimated via CPU time, which for this problem makes little sense.
costs = np.array([2**(n_models-i) for i in range(n_models)])
problem = MyProblem(n_models, costs=costs, covariance_estimation_samples=32, verbose=False)

print("Covariance matrix:\n")
print(problem.get_covariance())
print("\nCorrelation matrix:\n")
print(problem.get_correlation())

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

# Solve with MLBLUE
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
sol_MLBLUE = problem.solve(K=n_models, eps=eps)

print("\n\nMLBLUE\n")
print("MLBLUE data:\n")
for key, item in MLBLUE_data.items(): print(key, ": ", item)
print("MLBLUE solution: ", sol_MLBLUE[0])

