# blue function for coupled levels

from numpy import zeros, array, isfinite, ndarray, savez_compressed, load
from numpy.random import RandomState
from numpy import sum as npsum
from time import time
from inspect import signature
from shutil import get_terminal_size
from mpi4py.MPI import COMM_WORLD, SUM
import os

cols = get_terminal_size()[0]
if cols == 0: cols = 80 # if ran with MPI 0 cols is returned due to a bug

def is_output_finite(Ps):
    No = len(Ps)
    L  = len(Ps[0])
    for i in range(L):
        for n in range(No):
            check = isfinite(Ps[n][i])
            if isinstance(check, ndarray):
                check = check.all()
            else:
                try: check = all(check)
                except TypeError: pass
            if not check:
                return False,i,n

    return True,None,None

def flatten_nested_list(X):
    if isinstance(X, ndarray): return X.flatten()
    elif isinstance(X, (tuple, list)): return [flatten_nested_list(item) for item in X]
    return X

def blue_fn(ls, N, problem, sampler=None, inners = None, comm = None, N1 = 1, No = 1, verbose=True, compute_mlmc_differences=False, filename=None, outputs_to_save=None):
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
        comm: (optional) the MPI communicator between the sampling groups
        inners: (optional) list of length No of Python functions so that
                inners[n] implements an inner product suitable for output n.
        verbose: boolean flag that turns progress bar on (True) or off (False)
        filename: (optional) string containing a filename where to store the
                  computed samples.

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
    if compute_mlmc_differences:
        sumsd1 = [[[0 for j in range(L)] for i in range(L)] for n in range(No)]
        sumsd2 = [[[0 for j in range(L)] for i in range(L)] for n in range(No)]

    verbose = verbose and COMM_WORLD.Get_rank() == 0

    if comm is None: comm = COMM_WORLD

    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()

    if inners is None:
        inners = [lambda a,b : a*b for n in range(No)]

    if sampler is None:
        RNG = RandomState(1+mpiRank)
        def sampler(ls, N=1):
            sample = RNG.randn(N)
            return [sample for i in range(L)]

    if verbose:
        l = len('Sampling models %s [] 100%%' % ls)
        part = max(int(round(0.9*(cols-l))),0)
        fmtstr = '{0: <%d}' % part
        print("\rSampling models %s [%s] %d%%" % (ls, fmtstr.format(''), 0), end="\r", flush=True)

    if filename is not None:
        ext = "." + filename.split(".")[-1]
        basename = '.'.join(filename.split(".")[:-1]) + ''.join(str(l) for l in ls)
        filename = basename + ext
        outfilename = basename + "_%d" % mpiRank + ext
        outdict = {"values_%d_%d" % (n,i) : [] for n in range(No) for i in range(L)}
        outdict.update({"inputs_%d" % i : [] for i in range(L)})
        if outputs_to_save is None: outputs_to_save = list(range(No))

    NN = [N//mpiSize]*mpiSize
    for i in range(N%mpiSize):
        NN[i] += 1

    assert sum(NN) == N

    nobatch = len(signature(sampler).parameters) == 1
    if nobatch: N1 = 1

    for it in range(1, NN[mpiRank]+1, N1):
        N2 = min(N1, NN[mpiRank] - it + 1)

        isfinite = False
        while not isfinite:
            if nobatch: samples = sampler(ls)
            else:       samples = sampler(ls, N2)

            start = time()
            Ps = problem.evaluate(ls, samples) 
            end = time()

            isfinite,model_n,output_n = is_output_finite(Ps)
            if not isfinite:
                print("Warning! Problem evaluation returned inf or NaN value for model %d and output %d. Resampling..." % (model_n,output_n), flush=True)

        cpu_cost += end - start # cost defined as total computational time

        if filename is not None:
            Ps_flat = flatten_nested_list(Ps)
            samples_flat = flatten_nested_list(samples)
            for n in range(No):
                if n in outputs_to_save:
                    for i in range(L):
                        if N1 == 1:
                            outdict["values_%d_%d" % (n,i)].append(Ps[n][i])
                            outdict["inputs_%d" % i].append(samples[i])
                        else:
                            for n2 in range(N2):
                                outdict["values_%d_%d" % (n,i)].append(Ps[n][i][n2])
                                outdict["inputs_%d" % i].append(samples[i][n2])

        if compute_mlmc_differences:
            for n in range(No):
                for i in range(L):
                    for j in range(i+1, L):
                        if N1 == 1:
                            sumsd1[n][i][j] += Ps[n][i] - Ps[n][j]
                            sumsd2[n][i][j] += inners[n](Ps[n][i] - Ps[n][j], Ps[n][i] - Ps[n][j])
                        else:
                            for n2 in range(N2):
                                sumsd1[n][i][j] += Ps[n][i][n2] - Ps[n][j][n2]
                                sumsd2[n][i][j] += inners[n](Ps[n][i][n2] - Ps[n][j][n2], Ps[n][i][n2] - Ps[n][j][n2])

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
            print("\rSampling models %s [%s] %d%%" % (ls, fmtstr.format('='*int(round(part*it/NN[mpiRank]))), (100*it)//NN[mpiRank]), end='\r', flush=True)

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
            if compute_mlmc_differences:
                for j in range(i+1, L):
                    sumsd1[n][i][j] = comm.allreduce(sumsd1[n][i][j], op = SUM)
                    sumsd2[n][i][j] = comm.allreduce(sumsd2[n][i][j], op = SUM)

    if filename is not None:
        if mpiRank != 0:
            savez_compressed(outfilename, **outdict)
        comm.barrier()
        if mpiRank == 0:
            for rank in range(1,mpiSize):
                outfilename = basename + "_%d" % rank + ext
                nextdict = load(outfilename, allow_pickle=True)
                for key in outdict.keys():
                    outdict[key] += [item for item in nextdict[key]]
                os.remove(outfilename)

            outdict["models"] = array([ls])
            outdict["n_samples"] = array([N])
            outdict["n_outputs"] = array([No])

            if os.path.isfile(filename):
                old_dict = dict(load(filename, allow_pickle=True))
                for key in old_dict.keys():
                    old_dict[key] = [item for item in old_dict[key]]
                assert old_dict["models"] == ls
                assert old_dict["n_outputs"] == outdict["n_outputs"]
                for key in old_dict.keys():
                    if "values" in key or "inputs" in key:
                        old_dict[key] += outdict[key]
                if isinstance(old_dict["n_samples"], list):
                    old_dict["n_samples"][0] += N
                else:
                    old_dict["n_samples"] += N
                outdict = old_dict
                
            savez_compressed(filename, **outdict)
        
        comm.barrier()

    if compute_mlmc_differences:
        return (sumse, sumsc, cost, sumsd1, sumsd2)
    else:
        return (sumse, sumsc, cost)
