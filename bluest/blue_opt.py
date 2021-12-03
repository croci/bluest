import numpy as np
from numba import njit
from itertools import combinations, combinations_with_replacement

########################################################

def attempt_mlmc_setup(v, w, budget=None, eps=None):
    if budget is None and eps is None:
        raise ValueError("Need to specify either budget or RMSE tolerance")
    elif budget is not None and eps is not None:
        eps = None

    if not all(np.isfinite(v)): return False,None

    q = sum(np.sqrt(v*w))
    if budget is not None: mu = budget/q
    else:                  mu = q/eps**2
    m = mu*np.sqrt(v/w)

    variance = lambda m : sum(v/m)
    if budget is not None: constraint = lambda m : m@w <= budget and all(m >= 1)
    else:                  constraint = lambda m : variance(m) <= eps**2 and all(m >= 1)

    m,var = best_closest_integer_solution(m, variance, constraint)
    if np.isinf(var): return False,None

    err = np.sqrt(var)
    tot_cost = m@w

    mlmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost}

    return True,mlmc_data

def attempt_mfmc_setup(sigmas, rhos, costs, budget=None, eps=None):
    if budget is None and eps is None:
        raise ValueError("Need to specify either budget or RMSE tolerance")
    elif budget is not None and eps is not None:
        eps = None

    if not all(np.isfinite(sigmas)): return False,None

    idx = np.argsort(abs(rhos))[::-1]
    assert idx[0] == 0

    s = sigmas[idx]
    rho = np.concatenate([rhos[idx], [0]])
    w = costs[idx]
    cost_ratio = w[:-1]/w[1:]
    rho_ratio = (rho[:-2]**2 - rho[1:-1]**2)/(rho[1:-1]**2 - rho[2:]**2)

    feasible = all(cost_ratio > rho_ratio)
    if not feasible: return feasible,None

    alphas = rho[1:-1]*s[0]/s[1:]

    r = np.sqrt(w[0]/w*(rho[:-1]**2 - rho[1:]**2)/(1-rho[1]**2))
    if budget is not None: m1 = budget/(w@r)
    else:                  m1 = eps**-2*(w@r)*(s[0]**2/w[0])*(1-rho[1]**2)
    m = np.concatenate([[m1], m1*r[1:]])

    variance = lambda m : s[0]**2/m[0] + sum((1/m[:-1]-1/m[1:])*(alphas**2*s[1:]**2 - 2*alphas*rho[1:-1]*s[0]*s[1:]))
    if budget is not None: constraint = lambda m : m@w <= budget and m[0] >= 1 and all(m[:-1] <= m[1:])
    else:                  constraint = lambda m : variance(m) <= eps**2 and m[0] >= 1 and all(m[:-1] <= m[1:])

    m,var = best_closest_integer_solution(m, variance, constraint)
    if np.isinf(var): return False,None

    err = np.sqrt(var)
    tot_cost = m@w

    mfmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost, "alphas" : alphas}

    return feasible,mfmc_data

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

    return best_val.astype(int), best_fval

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
def objectiveK(N, k,Lk,mk,groupsk,invcovsk):
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
        self.N = C.shape[0]
        self.K = K
        self.costs = costs
        self.samples = None
        self.budget = None
        self.eps = None
        self.tot_cost = None

        flattened_groups = []
        flattened_sq_invcovs = []
        invcovs = [[] for k in range(K)]
        sizes = [0] + [len(groupsk) for groupsk in groups]
        for k in range(1, K+1):
            groupsk = groups[k-1]
            for i in range(len(groupsk)):
                idx = np.array([groupsk[i]])
                index = (idx.T, idx)
                invcovs[k-1].append(np.linalg.inv(C[index]))
                flattened_sq_invcovs.append(invcovs[k-1][-1])
                flattened_groups.append(groupsk[i])

            groups[k-1] = np.array(groups[k-1])
            invcovs[k-1] = np.vstack(invcovs[k-1]).flatten()

        self.sizes            = sizes
        self.groups           = groups
        self.flattened_groups = flattened_groups
        self.flattened_sq_invcovs = flattened_sq_invcovs
        self.invcovs          = invcovs
        self.cumsizes         = np.cumsum(sizes)
        self.L                = self.cumsizes[-1]

        self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])

        self.variance, self.variance_with_grad_and_hess = self.get_variance_functions()

    def compute_BLUE_estimator(self, sums):
        C = self.C
        K = self.K
        L = self.L
        N = self.N
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        y = np.zeros((L,))
        sums = [sums[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
        for k in range(1, K+1):
            for i in range(sizes[k]):
                for j in range(k):
                    for s in range(k):
                        y[groups[k-1][i][j]] += invcovs[k-1][k*k*i + k*j + s]*sums[k-1][i][s]

        def PHIinvY0(m, y, delta=0.0):
            if abs(m).max() < 0.05: return np.inf
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(N, k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

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
        L = self.L
        K = self.K
        N = self.N
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        def variance(m, delta=0.0):
            if abs(m).max() < 0.05: return np.inf
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(N, k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

            PHI = PHI.reshape((N,N))
            idx = get_nnz_rows_cols(m,groups)
            PHI = PHI[idx]

            assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

            try: out = np.linalg.solve(PHI,np.eye(len(idx[0]),1).flatten())[0]
            except np.linalg.LinAlgError:
                assert False # after the above fix we should never get here
                out = np.linalg.pinv(PHI)[0,0]

            return out

        def variance_with_grad_and_hess(m, delta=0.0, nohess=False):
            if abs(m).max() < 0.05: return np.inf, np.inf*np.ones((L,))
            PHI = delta*np.eye(N).flatten()
            m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
            for k in range(1, K+1):
                PHI += objectiveK(N, k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

            PHI = PHI.reshape((N,N))
            invPHI = np.linalg.pinv(PHI)

            idx = get_nnz_rows_cols(m,groups)
            var = np.linalg.inv(PHI[idx])[0,0]
            #var = invPHI[0,0]

            grad = -np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

            def arr(x):
                if isinstance(x,float):
                    return np.array([x])
                else: return x

            if nohess:
                return var,grad,None

            hess = np.zeros((L,L))
            ip = invPHI[:,0]
            for k in range(L):
                for s in range(k,L):
                    hess[k,s] = arr(ip[np.array(self.flattened_groups[k])])@(self.flattened_sq_invcovs[k]@invPHI[np.ix_(self.flattened_groups[k],self.flattened_groups[s])]@self.flattened_sq_invcovs[s])@arr(ip[np.array(self.flattened_groups[s])])

            hess += np.triu(hess,1).T

            return var,grad,hess

        return variance,variance_with_grad_and_hess

    def solve(self, budget=None, eps=None, solver="gurobi", integer=False):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["gurobi", "scipy", "cvxpy"]:
            raise ValueError("Optimization solvers available: 'gurobi', 'scipy' or 'cvxpy'")
        if integer: solver="gurobi"

        if eps is None:
            print("Minimizing statistical error for fixed cost...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(budget=budget, integer=integer)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(budget)
            elif solver == "scipy":  samples = self.scipy_solve(budget)

            if not integer:
                constraint = lambda m : m@self.costs <= budget and m@self.e >= 1
                objective  = self.variance
                
                samples,fval = best_closest_integer_solution(samples, objective, constraint)
                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(budget=budget, integer=True)

        else:
            print("Minimizing cost given statistical error tolerance...\n")
            samples = self.gurobi_solve(eps=eps, integer=integer)

            if not integer:
                objective   = lambda m : m@self.costs
                constraint  = lambda m : m@self.e >= 1 and self.variance(m) <= eps**2

                samples,fval = best_closest_integer_solution(samples, objective, constraint)

                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(eps=eps, integer=True)


        samples = samples.astype(int)

        self.samples = samples
        self.budget = budget
        self.eps = eps
        self.tot_cost = samples@self.costs

        return samples

    def gurobi_solve(self, budget=None, eps=None, integer=False):
        from gurobipy import Model,GRB

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        
        min_cost = eps is not None

        K        = self.K
        L        = self.L
        N        = self.N
        w        = self.costs
        e        = self.e
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
            M.addConstr(m@e >= 1, name="minimum_samples")
        else:
            M.setObjective(t[0], GRB.MINIMIZE)
            M.addConstr(m@w <= budget, name="budget")
            M.addConstr(m@e >= 1, name="minimum_samples")

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
        e        = self.e
        N        = self.N
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
        constraints = [m >= 0.1*np.ones((L,)), w@m == budget, m@e >= 1]
        prob = cp.Problem(obj, constraints)

        prob.solve(verbose=True, solver="SCS", eps=1.0e-4)

        return m.value

    def scipy_solve(self, budget):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,line_search

        L = self.L
        w = self.costs
        e = self.e

        print("Optimizing using scipy...")

        constraint1 = Bounds(0.1*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint2 = {"type":"eq", "fun" : lambda x : w.dot(x) - budget}
        constraint3 = {"type":"ineq", "fun" : lambda x : e.dot(x) - 1}

        #constraint1 = LinearConstraint(np.eye(L), 0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint2 = LinearConstraint(w, -np.inf, budget)
        constraint3 = LinearConstraint(e, 1, np.inf, keep_feasible=True)

        x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
        res = minimize(lambda x : self.variance_with_grad_and_hess(x,nohess=True,delta=0)[:-1], x0, jac=True, hess=lambda x : self.variance_with_grad_and_hess(x,delta=0)[-1], bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : "SVDFactorization", "disp" : True, "maxiter":200}, tol = 1.0e-8)
        #res = minimize(lambda x : self.variance_with_grad_and_hess(x)[:1], x0, jac=True, bounds = constraint1, constraints=[constraint2,constraint3], method="SLSQP", options={"ftol" : 1.0e-8, "disp" : True, "maxiter":100000}, tol = 1.0e-8)

        return res.x

if __name__ == '__main__':

    N = 10
    KK = 3 # if this is set to K it messes up the class

    C = np.random.randn(N,N); C = C.T@C

    groups = [[comb for comb in combinations(range(N), k)] for k in range(1, KK+1)]
    L = sum(len(groups[k-1]) for k in range(1,KK+1))
    costs = 1. + 5*np.arange(L)[::-1]
    budget = 10*sum(costs)
    eps = np.sqrt(C[0,0])/100

    print("Problem size: ", L)

    problem = BLUESampleAllocationProblem(C, KK, groups, costs)

    scipy_sol,cvxpy_sol,gurobi_sol = None,None,None
    scipy_sol  = problem.solve(budget=budget, solver="scipy")
    #cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
    gurobi_sol = problem.solve(budget=budget, solver="gurobi")
    #gurobi_eps_sol = problem.solve(eps=eps, solver="gurobi")

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [problem.variance(sol) for sol in sols if sol is not None]

    print(fvals)
