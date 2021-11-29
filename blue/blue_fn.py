# blue function for coupled levels

from numpy import random, zeros, array
from numpy import sum as npsum
from time import time

def blue_fn(ls, N, problems, sampler=None, N1 = 1):
    """
    Inputs:
        ls: tuple with the indices of the coupled models to sample from
        N: number of paths
        problems: list of problems or problems class.
            If list of problems:
              problems[l]: application-specific model-l problem problem
              Problems must have an evaluate method such that
              problems[l].evaluate(sample) returns output P_l.
              Optionally, user-defined problems[l].cost
            If problems class:
              problems.evaluate(ls, samples) returns coupled list of
              samples Ps[i] corresponding to model ls[i].
              Optionally, user-defined problems.cost
        sampler: sampling function, by default standard Normal.
            input: N, ls
            output: samples list so that samples[i] is the model
            ls[i] sample.
         N1: number of paths to generate concurrently.

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

    coupled_problem = not isinstance(problems, list)

    if sampler is None:
        def sampler(N, ls):
            sample = random.randn(N)
            return [sample for i in range(L)]

    for i in range(1, N+1, N1):
        N2 = min(N1, N - i + 1)

        samples = sampler(N2, ls)

        start = time()
        if coupled_problem:
            Ps = problems.evaluate(ls, samples) 
        else:
            Ps = [problems[l].evaluate(samples[i]) for i,l in enumerate(ls)]
            
        end = time()
        cpu_cost += end - start # cost defined as total computational time
        sumse += array([npsum(Ps[i]) for i in range(L)])
        sumsc += array([[npsum(Ps[i]*Ps[j]) for i in range(L)] for j in range(L)])

    if coupled_problem and hasattr(problems, 'cost'):
        cost = N*problems.cost
    elif all(hasattr(problems[i], 'cost') for i in range(L)):
        cost = N*npsum([problems[i].cost for i in range(L)])
    else:
        cost = cpu_cost

    return (sumse, sumsc, cost)
