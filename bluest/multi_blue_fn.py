# blue function for coupled levels

from numpy import random, zeros, array
from numpy import sum as npsum
from time import time

def blue_fn(ls, N, problem, sampler=None, inners = None, N1 = 1, No = 1, verbose=True):
    """
    Inputs:
        ls: tuple with the indices of the coupled models to sample from
        N: number of paths
        problem: problem class.
              problem.evaluate(ls, samples) returns coupled list of
              samples Ps[n][i] corresponding to output n and model ls[i].
              Optionally, user-defined problem.cost
        sampler: sampling function, by default standard Normal.
            input: N, ls
            output: samples list so that samples[i] is the model
            ls[i] sample.
         N1: number of paths to generate concurrently.
         No: number of outputs.
         verbose: boolean flag that turns progress bar on (True) or off (False)

    Outputs:
        (sumse, sumsc, cost) where sumse, sumsc are
        arrays of outputs:
        sumse[n][i]   = sum(Ps[n][i])
        sumsc[n][i,j] = sum(Ps[n][i]*Ps[n][j])
        cost = user-defined computational cost. By default, time.
    """

    L = len(ls)

    cpu_cost = 0.0
    sumse = [[0 for i in range(L)] for n in range(No)]
    sumsc = [zeros((L,L)) for n in range(No)]

    if inners is None:
        inners = [lambda a,b : a*b for n in range(No)]

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
        for n in range(No):
            if N1 == 1:
                for i in range(L):
                    sumse[n][i] += Ps[n][i]
                sumsc[n] += array([[inners[n](Ps[n][i],Ps[n][j]) for i in range(L)] for j in range(L)])
            else:
                for i in range(L):
                    sumse[n][i] += sum(Ps[n][i])
                sumsc[n] += array([[sum(inners[n](Ps[n][i][n2],Ps[n][j][n2]) for n2 in range(N2)) for i in range(L)] for j in range(L)])

        if verbose: print("\rSampling models %s [%-50s] %d%%" % (ls, '='*int(round(50*(i+1)/N)), int(round(100*(i+1)/N))), end='\r')

    if verbose: print('\rSampling of models %s completed. %s' % (ls, ' '*100))

    if hasattr(problem, 'cost'): cost = N*problem.cost
    else:                        cost = cpu_cost

    return (sumse, sumsc, cost)
