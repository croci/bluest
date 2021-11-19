import numpy as np
from numba import njit
from itertools import combinations
import sys

N = 6
K = 2

C = np.random.randn(N,N); C = C.T@C

groups,indices,invcovs,invcovs_sq = [[[] for k in range(K)] for i in range(4)]
sizes = [0]
for k in range(1, K+1):
    for comb in combinations(range(N), k):
        groups[k-1].append(comb)
        idx = np.array([comb])
        indices[k-1].append((idx.T, idx))
        invcovs[k-1].append(np.linalg.inv(C[indices[k-1][-1]]))
        invcovs_sq[k-1].append(invcovs[k-1][-1])

    sizes += [len(groups[k-1])]
    groups[k-1] = np.array(groups[k-1])
    invcovs[k-1] = np.vstack(invcovs[k-1]).flatten()


delta = 0.0
cumsizes = np.cumsum(sizes)
L = cumsizes[-1]

m0 = 1 + np.random.randint(100, size=L)

def objective(m=m0):
    PHI = delta*np.eye(N).flatten()
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    for k in range(1, K+1):
        PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

    PHI = PHI.reshape((N,N))

    out = np.linalg.solve(PHI,np.eye(N,1).flatten())[0]

    return out


@njit
def objectiveK(k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                PHI[N*group[j]+group[l]] += mk[i]*invcovsk[k*k*i + k*j + l]

    return PHI

out1 = objective(m0)

def objective_slow(m=m0):
    PHI = delta*np.eye(N)
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    for k in range(1, K+1):
        for i in range(sizes[k]):
            PHI[indices[k-1][i]] += m[k-1][i]*invcovs_sq[k-1][i]

    out = np.linalg.solve(PHI,np.eye(N,1).flatten())[0]
    return out

out2 = objective_slow(m0)

assert abs(out1-out2) <= 1.0e-10

########################################################

from scipy.optimize import minimize,LinearConstraint, Bounds

print("Optimizing...")

w = 1. + 5*np.arange(L)[::-1]
budget = 10*sum(w)
constraint1 = Bounds(np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
constraint2 = LinearConstraint(w, budget, budget)
constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}

res = minimize(objective, np.ones((L,)), bounds = constraint1, constraints=constraint2, method="SLSQP", options={"ftol":1.0e-6,"disp":True}, tol = 1.0e-6)
