import numpy as np
import cvxpy as cp

N = 8
L = 10

def get_spd(N):
    A = np.random.randn(8,8); A = A.T@A
    return A

def get_constraint(As, m, t):
    X = sum(As[i]*m[i] for i in range(L))
    e = np.zeros((N,1)); e[0] = 1
    return cp.bmat([[X, e], [e.T, cp.reshape(t,(1,1))]])
    

As = [get_spd(N) for i in range(L)]
Bs = [get_spd(N) for i in range(L)]
w = 1. + 5*np.arange(L)[::-1]

m = cp.Variable(L, nonneg=True)
t = cp.Variable(nonneg=True)

obj = cp.Minimize(t)
constraints = [w@m <= 1, m[0] >= 0.01, get_constraint(As, m, t) >> 0, get_constraint(Bs, m, t) >> 0] 
prob = cp.Problem(obj, constraints)

prob.solve(verbose=True, solver="CVXOPT", abstol=1.0e-12, reltol=1.e-5, max_iters=1000, feastol=1.0e-5, kttsolver='chol',refinement=2)

print(m.value)

