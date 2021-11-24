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
        out = np.linalg.pinvh(PHI)[0,0]

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

def gurobi_solve():
    import gurobipy as gp
    from gurobipy import GRB

    def objective_gurobi(m,t):
        PHI = delta*np.eye(N).flatten()
        E = np.zeros((N*N,))
        for k in range(1, K+1):
            Lk = sizes[k]
            mk = m[cumsizes[k-1]:cumsizes[k]]
            groupsk = groups[k-1]
            invcovsk = invcovs[k-1]
            for i in range(Lk):
                group = groupsk[i]
                for j in range(k):
                    for l in range(k):
                        E[N*group[j] + group[l]] = 1.
                        PHI = PHI + E*(mk[i]*invcovsk[k*k*i + k*j + l])
                        E[N*group[j] + group[l]] = 0

        out = np.zeros((N,))
        e = np.zeros((N,))
        for i in range(N):
            for j in range(N):
                e[i] = 1
                out = out + e*(PHI[N*i + j]*t[j])
                e[i] = 0
        return out

    M = gp.Model("BLUE")
    M.params.NonConvex = 2
    m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
    t = M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t")
    M.setObjective(t[0], GRB.MINIMIZE)
    M.addConstr(m@w == budget, name="budget")
    M.addConstr(m >= 0, name="positivity")
    ob = objective_gurobi(m,t)
    M.addConstr(ob[0] == 1)
    M.addConstrs((ob[i] == 0 for i in range(1,N)))
    M.optimize()

    return np.array(M.getAttr("X")[:L])

def cvxpy_solve(delta=0.01):
    import cvxpy as cp

    def objective_cvxpy(m):
        PHI = delta*np.eye(N).flatten()
        E = np.zeros((N*N,))
        for k in range(1, K+1):
            Lk = sizes[k]
            mk = m[cumsizes[k-1]:cumsizes[k]]
            groupsk = groups[k-1]
            invcovsk = invcovs[k-1]
            for i in range(Lk):
                group = groupsk[i]
                for j in range(k):
                    for l in range(k):
                        E[N*group[j] + group[l]] = 1.
                        PHI = PHI + E*(mk[i]*invcovsk[k*k*i + k*j + l])
                        E[N*group[j] + group[l]] = 0

        PHI = cp.reshape(PHI, (N,N))
        e = np.zeros((N,)); e[0] = 1
        out = cp.matrix_frac(e, PHI)
        return out

    m = cp.Variable(L)
    obj = cp.Minimize(objective_cvxpy(m))
    constraints = [m >= 0.1*np.ones((L,)), w@m == budget]
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=True, solver="SCS", eps=1.0e-4)

    return m.value

def scipy_solve():
    from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search
    import scipy.sparse as sp

    print("Optimizing...")

    constraint1 = Bounds(0.1*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
    constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}

    x0 = np.random.rand(L); x0 = x0/(x0@w)*budget
    res = minimize(objective_with_grad, x0, jac=True, bounds = constraint1, constraints=constraint2, method="SLSQP", options={"ftol" : 1.0e-10, "disp" : True, "maxiter":1000}, tol = 1.0e-10)

    return res.x

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
    gurobi_sol = gurobi_solve()
    cvxpy_sol  = cvxpy_solve()
    scipy_sol  = scipy_solve()

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [objective(sol) for sol in sols]

    int_sols = [find_integer_opt(sol)[0] for sol in sols]
    int_fvals = [objective(sol) for sol in int_sols]

    print(fvals)
    print(int_fvals)

