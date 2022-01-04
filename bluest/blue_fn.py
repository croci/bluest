# blue function for coupled levels

from numpy import zeros, array
from numpy.random import RandomState
from numpy import sum as npsum
from time import time
from shutil import get_terminal_size
from mpi4py.MPI import COMM_WORLD, SUM

cols = get_terminal_size()[0]

def blue_fn(ls, N, problem, sampler=None, inners = None, N1 = 1, No = 1, verbose=True):
    """
    Inputs:
        ls: tuple with the indices of the coupled models to sample from
        N: number of paths
        problem: problem class.
              problem.evaluate(ls, samples) returns coupled list of
              samples Ps[n][i] corresponding to output n and model ls[i].
              Optionally, user-defined problem.cost
              Optionally, user-defined problem.get_comm() that returns
              the MPI communicator between the sampling groups
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

    verbose = verbose and COMM_WORLD.Get_rank() == 0

    try: comm = problem.get_comm()
    except AttributeError: comm = COMM_WORLD

    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()

    if inners is None:
        inners = [lambda a,b : a*b for n in range(No)]

    if sampler is None:
        RNG = RandomState(1+mpiRank)
        def sampler(ls, N):
            sample = RNG.randn(N)
            return [sample for i in range(L)]

    if verbose:
        l = len('Sampling models %s [] 100%%' % ls)
        part = max(round(0.9*(cols-l)),0)
        fmtstr = '{0: <%d}' % part
        print("\rSampling models %s [%s] %d%%" % (ls, fmtstr.format(''), 0), end="\r", flush=True)


    nprocs  = min(mpiSize,max(N,1))
    NN      = [N//nprocs]*nprocs 
    NN[0]  += N%nprocs
    NN     += [0 for i in range(mpiSize - nprocs)]

    for it in range(1, NN[mpiRank]+1, N1):
        N2 = min(N1, NN[mpiRank] - it + 1)

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

        if verbose:
            print("\rSampling models %s [%s] %d%%" % (ls, fmtstr.format('='*round(part*it/NN[mpiRank])), (100*it)//NN[mpiRank]), end='\r', flush=True)

    if verbose:
        l = len('Sampling of models %s completed.' % ls)
        print('\rSampling of models %s completed.%s' % (ls, ' '*max(cols-l,0)), flush=True)

    if hasattr(problem, 'cost'): cost = N*problem.cost
    else:                        cost = comm.allreduce(cpu_cost, op = SUM)

    for n in range(No):
        sumsc[n] = comm.allreduce(sumsc[n], op = SUM)
        for i in range(L):
            #NOTE: if sumse[n][i] is a scalar or a numpy array the following works
            sumse[n][i] = comm.allreduce(sumse[n][i], op = SUM)

    return (sumse, sumsc, cost)
