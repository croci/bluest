# blue function for coupled levels

from numpy import random, zeros, array
from numpy import sum as npsum
from time import time

def blue_fn(ls, N, problem, sampler=None, N1 = 1, verbose=True):
    """
    Inputs:
        ls: tuple with the indices of the coupled models to sample from
        N: number of paths
        problem: problem class.
            If problem class:
              problem.evaluate(ls, samples) returns coupled list of
              samples Ps[i] corresponding to model ls[i].
              Optionally, user-defined problem.cost
        sampler: sampling function, by default standard Normal.
            input: N, ls
            output: samples list so that samples[i] is the model
            ls[i] sample.
         N1: number of paths to generate concurrently.
         verbose: boolean flag that turns progress bar on (True) or off (False)

    Outputs:
        (sumse, sumsc, cost) where sumse, sumsc are
        arrays of outputs:
        sumse[i]   = sum(Ps[i])
        sumsc[i,j] = sum(Ps[i]*Ps[j])
        cost = user-defined computational cost. By default, time.
    """

    L = len(ls)
    cpu_cost = 0.0
    sumse = zeros((L,))
    sumsc = zeros((L,L))

    if sampler is None:
        def sampler(ls, N):
            sample = random.randn(N)
            return [sample for i in range(L)]

    if verbose: print("\rSampling models %s [%-50s] %d%%" % (ls, '', 0), end="\r")

    for i in range(1, N+1, N1):
        N2 = min(N1, N - i + 1)

        samples = sampler(ls, N2)

        start = time()
        Ps = problem.evaluate(ls, samples) 
            
        end = time()
        cpu_cost += end - start # cost defined as total computational time
        sumse += array([npsum(Ps[i]) for i in range(L)])
        sumsc += array([[npsum(Ps[i]*Ps[j]) for i in range(L)] for j in range(L)])

        if verbose: print("\rSampling models %s [%-50s] %d%%" % (ls, '='*int(round(50*(i+1)/N)), int(round(100*(i+1)/N))), end='\r')

    if verbose: print('\rSampling of models %s completed. %s' % (ls, ' '*100))

    if hasattr(problem, 'cost'): cost = N*problem.cost
    else:                         cost = cpu_cost

    return (sumse, sumsc, cost)
