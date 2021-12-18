import numpy as np
from numba import njit,jit
from itertools import combinations, product

import cvxpy as cp
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

########################################################

# MOSEK is suboptimal for some reason
# SCS does not converge
mosek_params = {
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
    'MSK_DPAR_INTPNT_CO_TOL_MU_RED' : 1.0e-7,
    #'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL':1,
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 100,
}

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

    m,var = best_closest_integer_solution(m, variance, constraint, len(v))
    if np.isinf(var): return False,None

    err = np.sqrt(var)
    tot_cost = m@w

    mlmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost, "variance" : variance}

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

    m,var = best_closest_integer_solution(m, variance, constraint, len(sigmas))
    if np.isinf(var): return False,None

    err = np.sqrt(var)
    tot_cost = m@w

    mfmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost, "alphas" : alphas, "variance" : variance}

    return feasible,mfmc_data

def get_nnz_rows_cols(m,groups,cumsizes):
    K = len(cumsizes)-1
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    out = np.unique(np.concatenate([groups[k][abs(m[k]) > 1.0e-6].flatten() for k in range(K)]))
    return out.reshape((len(out),1)), out.reshape((1,len(out)))

def best_closest_integer_solution(sol, obj, constr, N, e=None):
    L = len(sol)
    val = np.sort(sol)[-int(1.5*N):][0]
    ss = np.round(sol).astype(int)
    idx = np.argwhere(sol >= val).flatten()
    if e is not None:
        idx2 = np.argwhere(e > 0).flatten()
        temp = np.argsort(sol[e>0])[::-1]
        idx2 = idx2[temp[:N]]
        idx = np.unique(np.concatenate([idx, idx2]))

    LL = len(idx)

    lb = np.zeros((L,)); ub = np.zeros((L,))
    lb[idx] = np.floor(sol).astype(int)[idx]
    ub[idx] = np.ceil(sol).astype(int)[idx]
    bnds = np.vstack([lb,ub])

    best_val  = sol.copy()
    best_fval = np.inf
    for item in product([0,1], repeat=LL):
        val = ss.copy()
        val[idx] = bnds[item, idx]
        constraint_satisfied = constr(val)
        if constraint_satisfied:
            fval = obj(val)
            if fval < best_fval:
                best_fval = fval
                best_val = val.copy()
                if LL > 15:
                    return best_val.astype(int), best_fval

    return best_val.astype(int), best_fval

@njit(fastmath=True)
def hessKQ(k, q, Lk, Lq, groupsk, groupsq, invcovsk, invcovsq, invPHI):
    hess = np.zeros((Lk,Lq))
    ip = invPHI[:,0]
    ksq = k*k; qsq = q*q
    for ik in range(Lk):
        ipk = ip[groupsk[ik]]
        for iq in range(Lq):
            ipq = ip[groupsq[iq]]
            iP = invPHI[groupsk[ik],:][:,groupsq[iq]]
            #ipk@icovk@invPHI@icovq@ipq + the simmetric
            #ipk_m Ck_ms*(sum_j iP_sj*(sumi_l Cq_jl*ipq_l)_j)_s
            for lk in range(k):
                for jk in range(k):
                    for jq in range(q):
                        for lq in range(q):
                            hess[ik,iq] += ipk[lk]*invcovsk[ksq*ik + k*lk + jk]*iP[jk,jq]*invcovsq[qsq*iq + q*jq + lq]*ipq[lq]

    return hess

@njit(fastmath=True)
def gradK(k, Lk,groupsk,invcovsk,invPHI):
    grad = np.zeros((Lk,))
    for i in range(Lk):
        temp = invPHI[groupsk[i],0] # PHI is symmetric
        for j in range(k):
            for l in range(k):
                grad[i] += temp[j]*invcovsk[k*k*i + k*j + l]*temp[l]

    return grad

@njit(fastmath=True)
def objectiveK(N, k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                PHI[N*group[j]+group[l]] += mk[i]*invcovsk[k*k*i + k*j + l]

    return PHI

@njit(fastmath=True)
def fastobj(N,k,Lk,groupsk,invcovsk):
    psi = np.zeros((N*N,Lk))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                psi[N*group[j]+group[l], i] += invcovsk[k*k*i + k*j + l]
    return psi

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

        self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])

        self.get_variance_functions()

    def compute_BLUE_estimator(self, sums, samples=None):
        C = self.C
        K = self.K
        L = self.L
        N = self.N
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        if samples is None: samples = self.samples

        y = np.zeros((L,))
        sums = [sums[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
        for k in range(1, K+1):
            for i in range(sizes[k]):
                for j in range(k):
                    for s in range(k):
                        y[groups[k-1][i][j]] += invcovs[k-1][k*k*i + k*j + s]*sums[k-1][i][s]

        def PHIinvY0(m, y, delta=0.0):
            if abs(m).max() < 0.05: return np.inf

            PHI = self.get_phi(m,delta=delta)

            idx = get_nnz_rows_cols(m,groups,cumsizes)
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

        return PHIinvY0(samples, y)

    def get_variance_functions(self):
        C = self.C
        L = self.L
        K = self.K
        N = self.N
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        self.psi = np.hstack([fastobj(N,k,sizes[k],groups[k-1],invcovs[k-1]) for k in range(1,K+1)])

        def get_phi(m, delta=0.0, option=2):
            if option == 1:
                PHI = delta*np.eye(N).flatten()
                m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
                for k in range(1, K+1):
                    PHI += objectiveK(N, k,sizes[k],m[k-1],groups[k-1],invcovs[k-1])

                PHI = PHI.reshape((N,N))
            else:
                PHI = delta*np.eye(N) + (self.psi@m).reshape((N,N))

            return PHI

        def variance(m, delta=0.0):
            if abs(m).max() < 0.05: return np.inf
            PHI = get_phi(m,delta=delta)

            idx = get_nnz_rows_cols(m,groups,cumsizes)
            PHI = PHI[idx]

            assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

            try: out = np.linalg.solve(PHI,np.eye(len(idx[0]),1).flatten())[0]
            except np.linalg.LinAlgError:
                assert False # after the above fix we should never get here
                out = np.linalg.pinv(PHI)[0,0]

            return out

        def variance_with_grad_and_hess(m, delta=0.0, nohess=False):
            if abs(m).max() < 0.05: return np.inf, np.inf*np.ones((L,))
            PHI = get_phi(m,delta=delta)

            invPHI = np.linalg.pinv(PHI)

            idx = get_nnz_rows_cols(m,groups,cumsizes)
            var = np.linalg.inv(PHI[idx])[0,0]
            #var = invPHI[0,0]

            grad = -np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

            if nohess: return var,grad,None

            hess = np.zeros((L,L))

            for k in range(1,K+1):
                for q in range(1,K+1):
                    hess[cumsizes[k-1]:cumsizes[k],:][:,cumsizes[q-1]:cumsizes[q]] = hessKQ(k, q, sizes[k], sizes[q], groups[k-1], groups[q-1], invcovs[k-1], invcovs[q-1], invPHI)

            hess += hess.T

            return var,grad,hess

        self.get_phi = get_phi
        self.variance = variance
        self.variance_with_grad_and_hess = variance_with_grad_and_hess

    def solve(self, budget=None, eps=None, solver="cvxpy", integer=False, x0=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["gurobi", "scipy", "cvxpy"]:
            raise ValueError("Optimization solvers available: 'gurobi', 'scipy' or 'cvxpy'")
        if integer: solver="gurobi"

        if eps is None:
            print("Minimizing statistical error for fixed cost...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(budget=budget, integer=integer)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(budget=budget)
            elif solver == "scipy":  samples = self.scipy_solve(budget=budget, x0=x0)

            if not integer:
                constraint = lambda m : m@self.costs <= 1.0001*budget and m@self.e >= 1
                objective  = self.variance
                
                ss = samples.copy()
                samples,fval = best_closest_integer_solution(samples, objective, constraint, self.N, self.e)
                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(budget=budget, integer=True)

        else:
            print("Minimizing cost given statistical error tolerance...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(eps=eps, integer=integer)
            elif solver == "scipy":  samples = self.scipy_solve(eps=eps, x0=x0)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(eps=eps)

            if not integer:
                objective  = lambda m : m@self.costs
                constraint = lambda m : m@self.e >= 1 and self.variance(m*eps**2) <= 1.0001

                samples,fval = best_closest_integer_solution(samples, objective, constraint, self.N, self.e)

                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(eps=eps, integer=True)


        samples = samples.astype(int)

        self.samples = samples
        self.budget = budget
        self.eps = eps
        self.tot_cost = samples@self.costs

        return samples

    def gurobi_constraint(self, m,t, delta=0):
        ''' ensuring that t = PHI^{-1}[:,0] '''

        K        = self.K
        N        = self.N
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

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

    def gurobi_solve(self, budget=None, eps=None, integer=False):
        from gurobipy import Model,GRB

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        
        L        = self.L
        N        = self.N
        w        = self.costs
        e        = self.e

        M = Model("BLUE")
        M.params.NonConvex = 2

        m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
        if integer: m.vType = GRB.INTEGER

        t = M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t")

        if eps is not None:
            M.setObjective(m@w, GRB.MINIMIZE)
            M.addConstr(t[0] <= eps**2, name="variance")
            M.addConstr(m@e >= 1, name="minimum_samples")
        else:
            M.setObjective(t[0], GRB.MINIMIZE)
            M.addConstr(m@w <= budget, name="budget")
            M.addConstr(m@e >= 1, name="minimum_samples")

        # enforcing the constraint that PHI^{-1}[:,0] = t
        constr = self.gurobi_constraint(m,t)
        M.addConstr(constr[0] == 1)
        M.addConstrs((constr[i] == 0 for i in range(1,N)))

        M.optimize()

        return np.array(m.X)

    def cvxpy_fun(self, m, t, delta=0):
        N = self.N
        PHI = cp.reshape(self.psi@m + delta*np.eye(N).flatten(), (N,N))
        ee = np.zeros((N,1)); ee[0] = 1
        return cp.bmat([[PHI,ee],[ee.T,cp.reshape(t,(1,1))]])

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        L        = self.L
        w        = self.costs
        e        = self.e

        m = cp.Variable(L, nonneg=True)
        t = cp.Variable(nonneg=True)
        #NOTE: the m@e >= 1 constraint is only needed to avoid arbitrarily small
        #      positive values of m. Note that it does not affect the BLUE since
        #      it is a constraint satisfied automatically by the integer formulation 
        if budget is not None:
            obj = cp.Minimize(t)
            #constraints = [w@m <= budget, m@e >= 1, self.cvxpy_fun(m,t,delta=0) >> 0]
            constraints = [w@m <= 1, m@e >= 1/budget, self.cvxpy_fun(m,t,delta=0) >> 0]
        else:
            obj = cp.Minimize((w/np.linalg.norm(w))@m)
            #constraints = [m@e >= 1, t <= eps**2, self.cvxpy_fun(m,t,delta=0) >> 0]
            constraints = [t <= 1, m@e >= eps**2, self.cvxpy_fun(m,t,delta=0) >> 0]
        prob = cp.Problem(obj, constraints)
        
        #prob.solve(verbose=True, solver="MOSEK", mosek_params=mosek_params)
        prob.solve(verbose=True, solver="CVXOPT", abstol=1.0e-10, reltol=1.e-6, max_iters=1000, feastol=1.0e-5, kttsolver='chol',refinement=2)

        if eps is not None: m.value *= eps**-2
        else:               m.value *= budget

        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        L = self.L
        w = self.costs
        e = self.e

        print("Optimizing using scipy...")

        constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint3 = LinearConstraint(e, 1, np.inf, keep_feasible=True)
        if budget is not None:
            constraint2 = LinearConstraint(w, -np.inf, budget)

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize(lambda x : self.variance_with_grad_and_hess(x,nohess=True,delta=0)[:-1], x0, jac=True, hess=lambda x : self.variance_with_grad_and_hess(x,delta=0)[-1], bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3}, tol = 1.0e-8)

        else:
            epsq = eps**2
            constraint2 = NonlinearConstraint(lambda x : self.variance(x,delta=0), epsq, epsq, jac = lambda x : self.variance_with_grad_and_hess(x,nohess=True,delta=0)[1], hess=lambda x,p : self.variance_with_grad_and_hess(x,delta=0)[2]*p)
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":10000, 'verbose':3}, tol = 1.0e-10)

        return res.x

if __name__ == '__main__':

    N = 10
    KK = 3 # if this is set to K it messes up the class

    C = np.random.randn(N,N); C = C.T@C

    groups = [[comb for comb in combinations(range(N), k)] for k in range(1, KK+1)]
    L = sum(len(groups[k-1]) for k in range(1,KK+1))
    costs = 1. + 5*np.arange(L)[::-1]
    budget = 100*sum(costs)
    eps = np.sqrt(C[0,0])/100

    print("Problem size: ", L)

    problem = BLUESampleAllocationProblem(C, KK, groups, costs)

    scipy_sol,cvxpy_sol,gurobi_sol = None,None,None
    if False:
        cvxpy_sol  = problem.solve(eps=eps, solver="cvxpy")
        scipy_sol  = problem.solve(eps=eps, solver="scipy")
        #gurobi_sol = problem.solve(eps=eps, solver="gurobi")
        print("MSE tolerance: ", eps**2)
    else:
        cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
        scipy_sol  = problem.solve(budget=budget, solver="scipy")
        gurobi_sol = problem.solve(budget=budget, solver="gurobi")
        print("Budget: ", budget)

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [(costs@sol, problem.variance(sol)) for sol in sols if sol is not None]

    print(fvals)
