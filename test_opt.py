import numpy as np
from numba import njit
from itertools import combinations, combinations_with_replacement
import sys

N = 10
K = 3

C = np.random.randn(N,N); C = C.T@C
C2 = np.random.randn(N,N); C2 = C2.T@C2

groups,indices,invcovs,invcovs2,invcovs_sq = [[[] for k in range(K)] for i in range(5)]
sizes = [0]
for k in range(1, K+1):
    for comb in combinations(range(N), k):
        groups[k-1].append(comb)
        idx = np.array([comb])
        indices[k-1].append((idx.T, idx))
        invcovs[k-1].append(np.linalg.inv(C[indices[k-1][-1]]))
        invcovs2[k-1].append(np.linalg.inv(C2[indices[k-1][-1]]))
        invcovs_sq[k-1].append(invcovs[k-1][-1])

    sizes += [len(groups[k-1])]
    groups[k-1] = np.array(groups[k-1])
    invcovs[k-1] = np.vstack(invcovs[k-1]).flatten()
    invcovs2[k-1] = np.vstack(invcovs2[k-1]).flatten()


delta = 0
cumsizes = np.cumsum(sizes)
L = cumsizes[-1]

print("Problem size: ", L)

m0 = 1 + np.random.randint(100, size=L)

w = 1. + 5*np.arange(L)[::-1]
budget = 10*sum(w)

def objective(m, invcovs=invcovs):
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

out1 = objective(m0)

out3,grad = objective_with_grad(m0)

########################################################

import gurobipy as gp
from gurobipy import GRB

#def gurobi_solve():
#
#    def objective_gurobi(m,t):
#        PHI = delta*np.eye(N).flatten()
#        E = np.zeros((N*N,))
#        for k in range(1, K+1):
#            Lk = sizes[k]
#            mk = m[cumsizes[k-1]:cumsizes[k]]
#            groupsk = groups[k-1]
#            invcovsk = invcovs[k-1]
#            for i in range(Lk):
#                group = groupsk[i]
#                for j in range(k):
#                    for l in range(k):
#                        E[N*group[j] + group[l]] = 1.
#                        PHI = PHI + E*(mk[i]*invcovsk[k*k*i + k*j + l])
#                        E[N*group[j] + group[l]] = 0
#
#        out = np.zeros((N,))
#        e = np.zeros((N,))
#        for i in range(N):
#            for j in range(N):
#                e[i] = 1
#                out = out + e*(PHI[N*i + j]*t[j])
#                e[i] = 0
#        return out
#
#    M = gp.Model("BLUE")
#    M.params.NonConvex = 2
#    m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
#    t = M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t")
#    M.setObjective(t[0], GRB.MINIMIZE)
#    M.addConstr(m@w == budget, name="budget")
#    M.addConstr(m >= 0, name="positivity")
#    ob = objective_gurobi(m,t)
#    M.addConstr(ob[0] == 1)
#    M.addConstrs((ob[i] == 0 for i in range(1,N)))
#    M.optimize()
#
#    gurobi_sol = np.array(M.getAttr("X")[:L])
#    return gurobi_sol

def gurobi_solve(eps=1.0e-1):

    def objective_gurobi(m,t,invcovs=invcovs):
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
    t2 = M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t2")
    M.setObjective(m@w, GRB.MINIMIZE)
    M.addConstr(t[0] <= eps**2, name="variance")
    M.addConstr(t2[0] <= eps**2, name="variance2")
    ob = objective_gurobi(m,t, invcovs)
    ob2 = objective_gurobi(m,t2,invcovs2)
    M.addConstr(ob[0] == 1)
    M.addConstrs((ob[i] == 0 for i in range(1,N)))
    M.addConstr(ob2[0] == 1)
    M.addConstrs((ob2[i] == 0 for i in range(1,N)))
    M.optimize()

    gurobi_sol = np.array(M.getAttr("X")[:L])
    return gurobi_sol

gurobi_sol = gurobi_solve()

sys.exit(0)

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
lmbda = 0.0
obj = cp.Minimize(objective_cvxpy(m) + lmbda*cp.norm(m,1))
constraints = [m >= int(abs(delta) < 1.e-10)*np.ones((L,)), w@m == budget]
prob = cp.Problem(obj, constraints)

prob.solve(verbose=True, solver="SCS", eps=1.0e-4)

cvxpy_sol = m.value

########################################################

from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search
import scipy.sparse as sp

print("Optimizing...")

constraint1 = Bounds(int(abs(delta) < 1.e-10)*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}

x0 = np.random.rand(L); x0 = x0/(x0@w)*budget
res = minimize(objective_with_grad, x0, jac=True, bounds = constraint1, constraints=constraint2, method="SLSQP", options={"ftol" : 1.0e-10, "disp" : True, "maxiter":1000}, tol = 1.0e-10)

sol = res.x
scipy_sol = sol

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

    return best_val, best_fval

sols = [gurobi_sol, cvxpy_sol, scipy_sol]
int_sols = [find_integer_opt(sol)[0] for sol in sols]
fvals = [objective(sol) for sol in sols]
int_fvals = [objective(sol) for sol in int_sols]

print(fvals)
print(int_fvals)

#def linesearch(xk, fk, gk):
#    pk = -gk
#    pk -= ((pk.dot(w) - budget + xk.dot(w))/(w.dot(w)))*w
#    alpha = 1/np.linalg.norm(gk)**2
#    fnew = objective(np.maximum(xk + alpha*pk,0))
#    m = -gk.dot(pk)
#    it = 0
#    while fk-fnew > alpha*m and it < 100:
#        it += 1
#        alpha /= 2
#        fnew = objective(np.maximum(xk + alpha*pk, 0))
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
#
#eps = res.fun
#constraint1 = LinearConstraint(np.eye(L), int(abs(delta) < 1.e-10)*np.ones((L,)), np.ones((L,))*np.inf, keep_feasible=True)
#constraint2 = NonlinearConstraint(objective, 0, eps, jac = lambda x:objective_with_grad(x)[1])
#res2 = minimize(lambda x: (w.dot(x),w), 10*np.random.rand(L), jac=True, hessp=lambda x,*args : np.zeros((L,)), constraints=[constraint1,constraint2], method="trust-constr", options={"disp" : True, 'initial_constr_penalty': 10.0, 'initial_tr_radius': 3.0, 'initial_barrier_parameter': 0.1, 'initial_barrier_tolerance': 0.1}, tol = 1.0e-8)

