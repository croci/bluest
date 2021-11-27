import numpy as np
from numba import njit
from itertools import combinations, combinations_with_replacement
import sys

########################################################

def get_nnz_rows_cols(m,groups):
    K = len(m)
    out = np.unique(np.concatenate([groups[k][abs(m[k]) > 1.0e-6].flatten() for k in range(K)]))
    return out.reshape((len(out),1)), out.reshape((1,len(out)))

def best_closest_integer_solution(sol, obj, constr):
    L = len(sol)
    lb = np.maximum(np.floor(sol), np.zeros((L,)))
    ub = np.ceil(sol)
    bnds = np.vstack([lb,ub])
    r = np.arange(L)
    best_val  = sol
    best_fval = np.inf
    for item in combinations_with_replacement([0,1], L):
        val = bnds[item, r]
        constraint_satisfied = constr(val)
        if constraint_satisfied:
            fval = obj(val)
            if fval < best_fval:
                best_fval = fval
                best_val = val

    return best_val, best_fval

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

class BLUESampleAllocationProblem(object):
    def __init__(self, C, K, groups, costs):

        self.C = C
        self.K = K
        self.costs = costs
        self.samples = None

        flattened_groups = []
        invcovs = [[] for k in range(K)]
        sizes = [0] + [len(groupsk) for groupsk in groups]
        for k in range(1, K+1):
            groupsk = groups[k-1]
            for i in range(len(groupsk)):
                idx = np.array([groupsk[i]])
                index = (idx.T, idx)
                invcovs[k-1].append(np.linalg.inv(C[index]))
                flattened_groups.append(groupsk[i])

            groups[k-1] = np.array(groups[k-1])
            invcovs[k-1] = np.vstack(invcovs[k-1]).flatten()

        self.sizes            = sizes
        self.groups           = groups
        self.flattened_groups = flattened_groups
        self.invcovs          = invcovs
        self.cumsizes         = np.cumsum(sizes)
        self.L                = self.cumsizes[-1]

        self.variance, self.variance_with_grad = self.get_variance_functions()

    def compute_BLUE_estimator(self, sums):
        C = self.C
        K = self.K
        L = self.L
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        y = np.zeros((L,))
        sums = [sums[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
        for k in range(1, K+1):
            for i in range(sizes[k]):
                for j in range(k):
                    y[groups[k][i][j]] += invcovs[k-1][i]@sums[k][i]

        def PHIinvY0(m, y, delta=0.0):
            if abs(m).max() < 0.05: return np.inf
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

            PHI = PHI.reshape((N,N))
            idx = get_nnz_rows_cols(m,groups)
            PHI = PHI[idx]
            y   = y[idx[0].flatten()]

            assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

            try:
                mu = np.linalg.solve(PHI,y)[0]
                var = np.linalg.solve(PHI, np.eye(len(y), 1).flatten())[0]
            except np.linalg.LinAlgError:
                assert False # after the above fix we should never get here
                pinvPHI = np.linalg.pinv(PHI)
                mu  = (pinvPHI@y)[0]
                var = pinvPHI[0,0] 

            return mu, var

        return PHIinvY0(self.samples, y)

    def get_variance_functions(self):
        C = self.C
        K = self.K
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        def variance(m, delta=0.0):
            if abs(m).max() < 0.05: return np.inf
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

            PHI = PHI.reshape((N,N))
            idx = get_nnz_rows_cols(m,groups)
            PHI = PHI[idx]

            assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

            try: out = np.linalg.solve(PHI,np.eye(len(idx[0]),1).flatten())[0]
            except np.linalg.LinAlgError:
                assert False # after the above fix we should never get here
                out = np.linalg.pinv(PHI)[0,0]

            return out

        def variance_with_grad(m, delta=0.0):
            if abs(m).max() < 0.05: return np.inf, np.inf*np.ones((L,))
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

            PHI = PHI.reshape((N,N))
            invPHI = np.linalg.pinv(PHI)

            idx = get_nnz_rows_cols(m,groups)
            var = np.linalg.inv(PHI[idx])[0,0]
            #var = invPHI[0,0]

            grad = np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

            return var,grad

        return variance,variance_with_grad

    def solve(self, budget=None, eps=None, solver="gurobi"):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["gurobi", "scipy", "cvxpy"]:
            raise ValueError("Optimization solvers available: 'gurobi', 'scipy' or 'cvxpy'")

        if eps is None:
            print("Minimizing statistical error for fixed cost...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(budget=budget)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(budget)
            elif solver == "scipy":  samples = self.scipy_solve(budget)

            constraint = lambda m : m@self.costs <= budget
            objective  = self.variance
            
            samples,fval = best_closest_integer_solution(samples, objective, constraint)
            if np.isinf(fval):
                print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                samples = self.gurobi_solve(budget=budget, integer=True)

        else:
            print("Minimizing cost given statistical error tolerance...\n")
            samples = self.gurobi_solve(eps=eps)

            objective   = lambda m : m@self.costs
            constraint  = lambda m : self.variance(m) <= eps**2

            samples,fval = best_closest_integer_solution(samples, objective, constraint)

            if np.isinf(fval):
                print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                samples = self.gurobi_solve(eps=eps, integer=True)


        self.samples = samples

        return samples

    def gurobi_solve(self, budget=None, eps=None, integer=False):
        from gurobipy import Model,GRB

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        
        min_cost = eps is not None

        K        = self.K
        L        = self.L
        w        = self.costs
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        delta = 0

        def gurobi_constraint(m,t):
            ''' ensuring that t = PHI^{-1}[:,0] '''
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

        M = Model("BLUE")
        M.params.NonConvex = 2

        m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
        if integer: m.vType = GRB.INTEGER

        t = M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t")

        if min_cost:
            M.setObjective(m@w, GRB.MINIMIZE)
            M.addConstr(t[0] <= eps**2, name="variance")
            e = np.array([int(0 in group) for groupsk in groups for group in groupsk])
            M.addConstr(m@e >= 1, name="minimum_samples")
        else:
            M.setObjective(t[0], GRB.MINIMIZE)
            M.addConstr(m@w <= budget, name="budget")

        # enforcing the constraint that PHI^{-1}[:,0] = t
        constr = gurobi_constraint(m,t)
        M.addConstr(constr[0] == 1)
        M.addConstrs((constr[i] == 0 for i in range(1,N)))

        M.optimize()

        return np.array(m.X)

    def cvxpy_solve(self, budget, delta=0.01):
        import cvxpy as cp

        K        = self.K
        L        = self.L
        w        = self.costs
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

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

    def scipy_solve(self, budget):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search

        L        = self.L
        w        = self.costs

        print("Optimizing using scipy...")

        constraint1 = Bounds(0.1*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}

        x0 = np.random.rand(L); x0 = x0/(x0@w)*budget
        res = minimize(self.variance_with_grad, x0, jac=True, bounds = constraint1, constraints=constraint2, method="SLSQP", options={"ftol" : 1.0e-10, "disp" : True, "maxiter":1000}, tol = 1.0e-10)

        return res.x

if __name__ == '__main__':
    from math import comb

    N = 10
    K = 3

    C = np.random.randn(N,N); C = C.T@C

    groups = [[comb for comb in combinations(range(N), k)] for k in range(1, K+1)]
    L = sum(len(groups[k-1]) for k in range(1,K+1))
    costs = 1. + 5*np.arange(L)[::-1]
    budget = 10*sum(costs)
    eps = np.sqrt(C[0,0])/100

    print("Problem size: ", L)

    problem = BLUESampleAllocationProblem(C, K, groups, costs)

    gurobi_sol = problem.solve(budget=budget, solver="gurobi")
    cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
    scipy_sol  = problem.solve(budget=budget, solver="scipy")
    #gurobi_eps_sol = problem.solve(eps=eps, solver="gurobi")

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [problem.variance(sol) for sol in sols]

    print(fvals)
