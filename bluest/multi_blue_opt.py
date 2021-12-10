import numpy as np
from itertools import combinations, product
from .blue_opt import BLUESampleAllocationProblem,get_nnz_rows_cols,best_closest_integer_solution, hessKQ, gradK, objectiveK, fastobj, mosek_params

class BLUEMultiObjectiveSampleAllocationProblem(BLUESampleAllocationProblem):
    def __init__(self, C, K, Ks, groups, multi_groups, costs, multi_costs):

        self.n_outputs = len(C)
        self.C = C
        self.N = C[0].shape[0]
        self.K = K
        self.Ks = Ks
        self.costs = costs
        self.multi_groups = multi_groups
        self.multi_costs = multi_costs
        flattened_groups = []
        for k in range(K):
            flattened_groups += groups[k]
            groups[k] = np.array(groups[k])

        self.flattened_groups = flattened_groups
        self.groups = groups

        self.SAPS = [BLUESampleAllocationProblem(C[n], Ks[n], multi_groups[n], multi_costs[n]) for n in range(self.n_outputs)]

        self.samples = None
        self.budget = None
        self.eps = None
        self.tot_cost = None

        #NOTE#FIXME: Can essentially just call the functions in the SAPS as long as we slice m correctly. We just need to get the right indices for each groups in multi_groups.

        self.sizes = [0] + [len(groupsk) for groupsk in groups]
        self.cumsizes = np.cumsum(self.sizes)
        #FIXME: do we need local versions of these?
        self.L = self.cumsizes[-1]
        self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])

        positions = [[[] for k in range(Ks[n])] for n in range(self.n_outputs)]
        invcovs   = [[[] for k in range(Ks[n])] for n in range(self.n_outputs)]
        local_sizes = [[0] + [len(groupsk) for groupsk in groups] for groups in multi_groups]
        for n in range(self.n_outputs):
            for k in range(1, Ks[n]+1):
                groupsk = multi_groups[n][k-1]
                for i in range(len(groupsk)):
                    idx = np.array([groupsk[i]])
                    index = (idx.T, idx)
                    invcovs[n][k-1].append(np.linalg.inv(C[n][index]))
                    pos = [j for j,item in enumerate(self.groups[k-1]) if groupsk[i] == item]
                    assert len(pos) == 1
                    positions[n][k-1].append(pos[0])

                invcovs[n][k-1]   = np.vstack(invcovs[n][k-1]).flatten()
                positions[n][k-1] = np.array(positions[n][k-1])

        self.local_sizes      = local_sizes
        self.invcovs          = invcovs
        self.positions        = positions
        self.local_cumsizes   = [np.cumsum(item) for item in local_sizes]
        self.local_L          = [item[-1] for item in self.local_cumsizes]
        self.local_e          = [np.array([int(0 in group) for groupsk in multi_groups[n] for group in groupsk]) for n in range(self.n_outputs)]

        self.get_variance_functions()

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

            self.get_phi(m,delta=delta)

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
        
        self.get_phi = get_phi

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
                constraint = lambda m : m@self.costs <= budget and m@self.e >= 1
                objective  = self.variance
                
                ss = samples.copy()
                samples,fval = best_closest_integer_solution(samples, objective, constraint)
                assert not np.isinf(fval)
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
                constraint = lambda m : m@self.e >= 1 and self.variance(m) <= eps**2

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

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0):
        import cvxpy as cp

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        K        = self.K
        L        = self.L
        w        = self.costs
        e        = self.e
        N        = self.N
        psi      = self.psi
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        NN = N+1

        def cvxpy_fun(m,t,delta=0):
            PHI = cp.reshape(psi@m + delta*np.eye(N).flatten(), (N,N))
            ee = np.zeros((N,1)); ee[0] = 1
            return cp.bmat([[PHI,ee],[ee.T,cp.reshape(t,(1,1))]])

        m = cp.Variable(L)
        t = cp.Variable()
        if budget is not None:
            obj = cp.Minimize(t)
            constraints = [m >= 0.0*np.ones((L,)), w@m <= budget, m@e >= 1, cvxpy_fun(m,t,delta=0) >> 0]
        else:
            obj = cp.Minimize(w@m)
            constraints = [m >= 0.0*np.ones((L,)), m@e >= 1, t <= eps**2, cvxpy_fun(m,t,delta=0) >> 0]
        prob = cp.Problem(obj, constraints)
        
        #prob.solve(verbose=True, solver="MOSEK", mosek_params=mosek_params)
        prob.solve(verbose=True, solver="CVXOPT", abstol=1.0e-8, reltol=1.e-6, max_iters=1000, feastol=1.0e-4, kttsolver='chol',refinement=2)

        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

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
            res = minimize(lambda x : [w@x,w], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":10000, 'verbose':3}, tol = 1.0e-6)


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
    cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
    scipy_sol  = problem.solve(budget=budget, solver="scipy")
    gurobi_sol = problem.solve(budget=budget, solver="gurobi")
    #gurobi_eps_sol = problem.solve(eps=eps, solver="gurobi")

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [problem.variance(sol) for sol in sols if sol is not None]

    print(fvals)
