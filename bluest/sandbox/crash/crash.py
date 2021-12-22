import cvxpy as cp
import numpy as np

data = np.load("./crash.npz")

N      = int(data["N"])        # scalar integer
L      = int(data["L"])        # scalar integer
budget = float(data["budget"]) # scalar float

psi    = data["psi"]    # N^2-by-L sparse matrix (loaded as dense)
w      = data["w"]      # length-L vector of floats
e      = data["e"]      # length-L 0-1 vector

m = cp.Variable(L, nonneg=True)
t = cp.Variable(nonneg=True)

# with this scale it converges
scale = 1/(abs(psi).sum(axis=0).mean())
# with scale set to 1 it crashes without any specific error
scale = 1

# SDP constraint. Equivalent to imposing ee^T@pinv(PSI)@ee <= t
PSI = cp.reshape((scale*psi)@m, (N,N))
ee = np.zeros((N,1)); ee[0] = np.sqrt(scale)
sdp_constr = cp.bmat([[PSI,ee],[ee.T,cp.reshape(t,(1,1))]]) >> 0

obj = cp.Minimize(t)
constraints = [w@m <= budget, m@e >= 1, sdp_constr]
problem = cp.Problem(obj, constraints)

problem.solve(verbose=True, solver="CVXOPT", abstol=1.0e-13, reltol=1.e-6, max_iters=200, feastol=1.0e-6, kttsolver='chol',refinement=2)

sol = m.value

print("SOLUTION: ", sol)
print("VALUE: ", np.linalg.pinv((data["psi"]@sol).reshape((N,N)))[0,0])
