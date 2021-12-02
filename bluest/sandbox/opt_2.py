import numpy as np
from numba import njit
from itertools import combinations, combinations_with_replacement
import sys

N = 5
K = 3

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

print("Problem size: ", L)

w = 1. + 5*np.arange(L)[::-1]
budget = 10*sum(w)

########################################################

def objective(m, delta=delta):
    PHI = delta*np.eye(N).flatten()
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    for k in range(1, K+1):
        PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

    PHI = PHI.reshape((N,N))

    try: out = np.linalg.solve(PHI,np.eye(N,1).flatten())[0]
    except np.linalg.LinAlgError:
        out = np.linalg.pinv(PHI)[0,0]

    return out

def objective_with_grad(m):
    PHI = delta*np.eye(N).flatten()
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    for k in range(1, K+1):
        PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

    PHI = PHI.reshape((N,N))
    invPHI = np.linalg.inv(PHI)

    obj = invPHI[0,0]

    grad = np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

    return obj,grad

@njit
def gradK(k, Lk,groupsk,invcovsk,invPHI):
    grad = np.zeros((Lk,))
    for i in range(Lk):
        temp = invPHI[groupsk[i],0] # PHI is symmetric
        for j in range(k):
            for l in range(k):
                grad[i] += temp[j]*invcovsk[k*k*i + k*j + l]*temp[l]

    return grad

@njit
def objectiveK(k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                PHI[N*group[j]+group[l]] += mk[i]*invcovsk[k*k*i + k*j + l]

    return PHI

########################################################

#maybe try with the beta directly?

lmbda,V = np.linalg.eigh(C)

b = V[:,0]

def get_m(A):
    A = A.reshape((L,N))
    # A is L-by-N
    #c = np.sum(lmbda*(A**2),1)
    c = np.zeros((L,))
    for i in range(L):
        c[i] = A[i,:]@(C@A[i,:])
    return np.sqrt(c/w)*budget/sum(np.sqrt(c*w))

def objective_scipy(A):
    A = A.reshape((L,N))
    c = np.zeros((L,))
    for i in range(L):
        c[i] = A[i,:]@(C@A[i,:])
    # A is L-by-N
    #c = np.sum(lmbda*(A**2),1)
    return sum(np.sqrt(c*w))**2

def constraint_scipy(A):
    return np.sum(A.reshape((L,N)),0) - np.eye(N,1).flatten()

def scipy_solve():
    from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search
    import scipy.sparse as sp

    print("Optimizing...")

    constraint = {"type":"eq", "fun" : constraint_scipy}

    x0 = np.random.randn(L*N)
    #res = minimize(lambda A : objective(get_m(A)), x0, constraints=constraint, options={"disp" : True, "maxiter":1000})
    res = minimize(objective_scipy, x0, constraints=constraint, options={"disp" : True, "maxiter":1000})
    #res = minimize(objective_scipy, x0, constraints=constraint, method="SLSQP", options={"ftol" : 1.0e-10, "disp" : True, "maxiter":1000}, tol = 1.0e-10)

    A = res.x
    #beta = res.x.reshape((L,N))@V.T

    m = get_m(A)

    return m

def find_integer_opt(sol):
    lb = np.maximum(np.floor(sol), np.zeros((L,)))
    ub = np.ceil(sol)
    bnds = np.vstack([lb,ub])
    r = np.arange(L)
    best_fval = np.inf
    for item in combinations_with_replacement([0,1], L):
        val = bnds[item, r]
        bval = w.dot(val)
        if bval <= budget:
            fval = objective(val)
        if fval < best_fval:
            best_fval = fval
            best_val = val

    return best_val, best_fval

if __name__ == '__main__':
    sol  = scipy_solve()
    int_sol = find_integer_opt(sol)[0]
