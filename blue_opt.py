import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from itertools import combinations
from time import time

N = 100
K = 3

C = np.random.randn(N,N); C = C.T@C

def construct_matrix_map(groups):
    idx = np.concatenate(groups)
    l = len(idx)
    return sp.csc_matrix((np.ones((l,),dtype=np.int16), (np.arange(l), idx)),dtype=np.int16)

groups = []
maps = []
invcovs = []
for k in range(1, K+1):
    group = [comb for comb in combinations(range(N), k)] 
    M = construct_matrix_map(group)

    iCs = np.hstack([np.linalg.inv(C[(np.array([item]).T, np.array([item]))]) for item in group]) 

    groups.append(group)
    maps.append(M)
    invcovs.append(iCs)

#indices = [(np.array([item]).T, np.array([item])) for item in groups]
#covs = [C[idx] for idx in indices]
#invcovs = [np.linalg.inv(cov) for cov in covs]
#M = construct_matrix_map(groups)
#L = len(groups)

delta = 0.0
sizes = [0] + [item.shape[1] for item in invcovs]
cumsizes = np.cumsum(sizes)
L = cumsizes[-1]
m_idx = np.vstack([cumsizes[:-1],cumsizes[1:]]).T

m0 = 1 + np.random.randint(100, size=L)

def objective(m=m0):
    PHI = np.zeros((N,N))
    for k in range(1, K+1):
        fac = np.repeat(m[m_idx[k-1,0]:m_idx[k-1,1]],k)*invcovs[k-1]) #NOTE: this is correct
        #FIXME
        PHI += maps[k-1].T@(np.repeat(m[m_idx[k-1,0]:m_idx[k-1,1]],k)*invcovs[k-1])@maps[k-1]

    #for i in range(L):
    #    PHI[indices[i]] += m[i]*invcovs[i]

    PHI[np.diag_indices(N)] += delta

    out = np.linalg.solve(PHI,np.eye(N,1).flatten())[0]
    return out

objective(m0)
