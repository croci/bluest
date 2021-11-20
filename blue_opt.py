import numpy as np
from numba import njit
from itertools import combinations, combinations_with_replacement
import sys

N = 10
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


delta = 0.05
cumsizes = np.cumsum(sizes)
L = cumsizes[-1]

print("Problem size: ", L)

m0 = 1 + np.random.randint(100, size=L)

def objective(m=m0):
    PHI = delta*np.eye(N).flatten()
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    for k in range(1, K+1):
        PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

    PHI = PHI.reshape((N,N))

    out = np.linalg.solve(PHI,np.eye(N,1).flatten())[0]

    return out

def objective_with_grad(m=m0):
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

out3,grad = objective_with_grad(m0)

########################################################

from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search
import scipy.sparse as sp

print("Optimizing...")

w = 1. + 5*np.arange(L)[::-1]
budget = 10*sum(w)
constraint1 = Bounds(int(abs(delta) < 1.e-10)*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
#constraint2 = LinearConstraint(w, budget, budget)
constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}

res = minimize(objective_with_grad, np.ones((L,)), jac=True, bounds = constraint1, constraints=constraint2, method="SLSQP", options={"ftol" : 1.0e-8, "disp" : True}, tol = 1.0e-8)

sol = res.x

print("\nFinding integer solution via informed brute force...\n")

def find_integer_opt(sol):
    lb = np.maximum(np.floor(sol), int(abs(delta) < 1.e-10)*np.ones((L,)))
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

    return best_val, best_fval, w.dot(best_val)

val, fval, bval = find_integer_opt(sol)
print("Objective value: ", fval, "\n\nOptimality gap: ", fval/res.fun)
assert bval <= budget
assert min(val) >= 0


#def linesearch(xk, fk, gk):
#    pk = -gk
#    pk -= (w.dot(pk)/w.dot(w))*w
#    alpha = 1/abs(pk).max()
#    fnew = objective(xk + alpha*pk)
#    m = -gk.dot(pk)
#    it = 0
#    while fk-fnew > alpha*m and it < 100:
#        it += 1
#        alpha /= 2
#        fnew = objective(np.maximum(xk + alpha*pk, 0))
#        assert False
#        print(fk-fnew, alpha*m)
#
#    if it == 100: print("Linesearch failed")
#    return np.maximum(xk + alpha*pk, 0)
#
#def proj_grad_desc(xk):
#    fk,gk = objective_with_grad(xk)
#    fkold = np.inf
#    it = 0
#    while abs(fk-fkold) > 1.0e-8 and it < 100:
#        it += 1
#        xk = linesearch(xk,fk,gk)
#        fkold = fk + 0
#        fk,gk = objective_with_grad(xk)
#
#    return xk
#
#x0 = 10*np.ones((L,))
#res2 = proj_grad_desc(x0)

#eps = res.fun
#constraint1 = LinearConstraint(np.eye(L), int(abs(delta) < 1.e-10)*np.ones((L,)), np.ones((L,))*np.inf, keep_feasible=True)
#constraint2 = NonlinearConstraint(objective, 0, eps, jac = lambda x:objective_with_grad(x)[1])
#res2 = minimize(lambda x: (w.dot(x),w), 10*np.random.rand(L), jac=True, hessp=lambda x,*args : np.zeros((L,)), constraints=[constraint1,constraint2], method="trust-constr", options={"disp" : True, 'initial_constr_penalty': 10.0, 'initial_tr_radius': 3.0, 'initial_barrier_parameter': 0.1, 'initial_barrier_tolerance': 0.1}, tol = 1.0e-8)

