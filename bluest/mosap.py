import numpy as np
from itertools import combinations, product
from .sap import SAP,mosek_params,cvxpy_default_params
from .misc import best_closest_integer_solution_BLUE_multi

import cvxpy as cp

class MOSAP(object):
    ''' MOSAP, MultiObjectiveSampleAllocationProblem '''
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

        self.SAPS = [SAP(C[n], Ks[n], multi_groups[n], multi_costs[n]) for n in range(self.n_outputs)]

        self.sizes = [0] + [len(groupsk) for groupsk in groups]
        self.cumsizes = np.cumsum(self.sizes)
        self.L = self.cumsizes[-1]
        self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])

        mappings = [[[] for k in range(Ks[n])] for n in range(self.n_outputs)]
        for n in range(self.n_outputs):
            for k in range(1, Ks[n]+1):
                groupsk = multi_groups[n][k-1]
                for i in range(len(groupsk)):
                    pos = [self.cumsizes[k-1] + j for j,item in enumerate(self.groups[k-1]) if all(groupsk[i] == item)]
                    assert len(pos) == 1
                    mappings[n][k-1].append(pos[0])

                mappings[n][k-1] = np.array(mappings[n][k-1])

            mappings[n] = np.concatenate(mappings[n])

        self.mappings = mappings # m <-> groups, and m[mappings[n]] = m_n <-> multi_groups[n]

        self.samples = None
        self.budget = None
        self.eps = None
        self.tot_cost = None

    def check_input(self, budget, eps):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if eps is not None:
            try:
                if len(eps) != self.n_outputs:
                    raise ValueError("eps must be a scalar or an array of tolerances")
                eps = np.array(eps)
            except TypeError:
                eps = np.array([eps for n in range(self.n_outputs)])
        return budget, eps
    
    def variances(self, m, delta=0):
        No       = self.n_outputs
        mappings = self.mappings
        return [self.SAPS[n].variance(m[mappings[n]],delta=delta) for n in range(No)]

    def variance_GH(self, m, nohess=False,delta=0):
        No       = self.n_outputs
        mappings = self.mappings

        out = [self.SAPS[n].variance_GH(m[mappings[n]],nohess=nohess,delta=delta) for n in range(No)]
        variances = [item[0] for item in out]
        gradients = [item[1] for item in out]
        hessians  = [item[2] for item in out]

        return variances,gradients,hessians

    def compute_BLUE_estimators(self, sums, samples):
        out = []
        for n in range(self.n_outputs):
            sums_n = [sums[n][item] for item in self.mappings[n]]
            out.append(self.SAPS[n].compute_BLUE_estimator(sums_n, samples=samples[self.mappings[n]]))

        #sums_list = [[sums[item] for item in self.mappings[n]] for n in range(self.n_outputs)]
        #out  = [self.SAPS[n].compute_BLUE_estimator(sums[self.mappings[n]]) for n in range(self.n_outputs)]
        mus  = [item[0] for item in out]
        Vars = np.array([item[1] for item in out])
        return mus,Vars

    def solve(self, budget=None, eps=None, solver="cvxpy", integer=False, x0=None, solver_params=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["gurobi", "scipy", "cvxpy", "ipopt"]:
            raise ValueError("Optimization solvers available: 'gurobi', 'scipy', 'ipopt' or 'cvxpy'")
        if integer: solver="gurobi"

        if eps is None:
            print("Minimizing statistical error for fixed cost...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(budget=budget, integer=integer)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(budget=budget, cvxpy_params=solver_params)
            elif solver == "ipopt":  samples = self.ipopt_solve(budget=budget, x0=x0)
            elif solver == "scipy":  samples = self.scipy_solve(budget=budget, x0=x0)

            if not integer:
                constraint = lambda m : m@self.costs <= 1.001*budget and all(m[self.mappings[n]]@self.e[self.mappings[n]] >= 1 for n in range(self.n_outputs))
                objective  = lambda m : max(self.variances(m))

                samples,fval = best_closest_integer_solution_BLUE_multi(samples, [self.SAPS[n].psi for n in range(self.n_outputs)], self.costs, self.e, self.mappings, budget=budget)
                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(budget=budget, integer=True)

        else:
            print("Minimizing cost given statistical error tolerance...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(eps=eps, integer=integer)
            elif solver == "scipy":  samples = self.scipy_solve(eps=eps, x0=x0)
            elif solver == "ipopt":  samples = self.ipopt_solve(eps=eps, x0=x0)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(eps=eps, cvxpy_params=solver_params)

            if not integer:
                objective  = lambda m : m@self.costs
                constraint = lambda m : all(m[self.mappings[n]]@self.e[self.mappings[n]] >= 1 for n in range(self.n_outputs)) and all(np.array(self.variances(m)) <= 1.001*np.array(eps)**2)

                samples,fval = best_closest_integer_solution_BLUE_multi(samples, [self.SAPS[n].psi for n in range(self.n_outputs)], self.costs, self.e, self.mappings, eps=eps)

                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(eps=eps, integer=True)


        samples = samples.astype(int)

        self.samples = samples
        self.budget = budget
        self.eps = eps
        self.tot_cost = samples@self.costs
        for n in range(self.n_outputs):
            self.SAPS[n].samples = samples[self.mappings[n]]

        return samples

    def gurobi_solve(self, budget=None, eps=None, integer=False):
        budget, eps = self.check_input(budget, eps)

        from gurobipy import Model,GRB

        L        = self.L
        N        = self.N
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        es = []
        for n in range(No):
            ee = np.zeros((L,))
            ee[mappings[n]] = e.copy()[mappings[n]]
            es.append(ee)

        M = Model("BLUE")
        M.params.NonConvex = 2

        m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
        if integer: m.vType = GRB.INTEGER

        t = [M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t%d" % n) for n in range(No)]

        if eps is not None:
            M.setObjective(m@w, GRB.MINIMIZE)
            M.addConstrs((t[n][0] <= eps[n]**2 for n in range(No)), name="variance")
            M.addConstrs((m@es[n] >= 1 for n in range(No)), name="minimum_samples")
        else:
            tt = M.addVar(vtype=GRB.CONTINUOUS, name="tmax")
            M.setObjective(tt, GRB.MINIMIZE)
            M.addConstr(m@w <= budget, name="budget")
            M.addConstrs((m@es[n] >= 1 for n in range(No)), name="minimum_samples")
            M.addConstrs((tt >= t[n][0] for n in range(No)), name="max_constraint")

        # enforcing the constraint that PHI^{-1}[:,0] = t
        for n in range(No):
            constr = self.SAPS[n].gurobi_constraint(m,t[n])
            M.addConstr(constr[0] == 1)
            M.addConstrs((constr[i] == 0 for i in range(1,N)))

        M.optimize()

        return np.array(m.X)

    def cvxpy_get_multi_constraints(self, m, t=None, eps=None):
        mappings = self.mappings
        No       = self.n_outputs

        if eps is None and t is None: raise ValueError("Need to provide either t or eps")
        if eps is None: eps = np.ones((No,))
        if t is None: t = 1

        scales = np.array([1/abs(self.SAPS[n].psi).sum(axis=0).mean() for n in range(No)])

        PHIs = []
        bmats = []
        for n in range(No):
            assert self.SAPS[n].psi.shape[1] == len(mappings[n])
            Nn = self.SAPS[n].N
            PHIs.append(cp.reshape((self.SAPS[n].psi*scales[n])@m[mappings[n]], (Nn,Nn)))
            e = np.zeros((Nn,1)); e[0] = np.sqrt(scales[n])/eps[n]
            bmats.append(cp.bmat([[PHIs[-1], e], [e.T, cp.reshape(t,(1,1))]]))

        out = [bmat >> 0 for bmat in bmats]
        return out

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0, cvxpy_params=None):
        budget, eps = self.check_input(budget, eps)

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        cvxpy_solver_params = cvxpy_default_params.copy()
        if cvxpy_params is not None:
            cvxpy_solver_params.update(cvxpy_params)

        scales = np.array([1/abs(self.SAPS[n].psi).sum(axis=0).mean() for n in range(No)])

        m = cp.Variable(L, nonneg=True)
        if budget is not None:
            t = cp.Variable(nonneg=True)
            obj = cp.Minimize(t)
            constraints = [w@m <= 1, *self.cvxpy_get_multi_constraints(m, t=t, eps=None)]
            constraints += [m[mappings[n]]@e[mappings[n]] >= 1./budget for n in range(self.n_outputs)]
        else:
            meps = max(eps)
            eps = eps/meps
            obj = cp.Minimize((w/np.linalg.norm(w))@m)
            constraints = [*self.cvxpy_get_multi_constraints(m, t=None, eps=eps)]
            constraints += [m[mappings[n]]@e[mappings[n]] >= 1*meps**2 for n in range(self.n_outputs)]

        prob = cp.Problem(obj, constraints)

        #prob.solve(verbose=True, solver="SCS")#, acceleration_lookback=0, acceleration_interval=0)
        #prob.solve(verbose=True, solver="MOSEK", mosek_params=mosek_params)
        prob.solve(verbose=True, solver="CVXOPT", **cvxpy_solver_params)

        if budget is not None: m.value *= budget
        elif eps  is not None: m.value *= meps**-2

        print(m.value.round())
        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

        budget, eps = self.check_input(budget, eps)

        delta = 1.0e-6

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        def max_variance(m,delta=0):
            return max(self.variances(m,delta=delta))

        print("Optimizing using scipy...")

        eee = np.zeros((L+1,)); eee[0] = 1
        es = []
        for n in range(No):
            ee = np.zeros((L,))
            ee[mappings[n]] = e.copy()[mappings[n]]
            es.append(ee)

        #NOTE: super annoying trick for doing lambda functions inside list comprehensions, argh!
        if budget is not None:
            constraint1 = Bounds(0.0*np.ones((L+1,)), np.inf*np.ones((L+1,)), keep_feasible=True)
            constraint3 = [LinearConstraint(np.concatenate([[0],ee]), 1, np.inf, keep_feasible=True) for ee in es]
            constraint2 = LinearConstraint(np.concatenate([[0],w]), -np.inf, budget)
            constraint4 = [NonlinearConstraint(lambda x,nn=n : x[0] - self.SAPS[nn].variance(x[1:][mappings[nn]],delta=delta), 0, np.inf, jac = lambda x,nn=n : np.concatenate([[1],-self.SAPS[nn].variance_GH(x[1:][mappings[nn]],nohess=True,delta=delta)[1]]), hess = lambda x,p,nn=n : np.block([[0, np.zeros((1,len(x)-1))],[np.zeros((len(x)-1,1)), -self.SAPS[nn].variance_GH(x[1:][mappings[nn]],delta=delta)[2]]])*p) for n in range(No)]

            if x0 is None: x0 = np.ceil(budget*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w); x0 = np.concatenate([[max_variance(x0,delta=delta)], x0])
            res = minimize(lambda x : (x[0], eee), x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2]+constraint3+constraint4, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3}, tol = 1.0e-15)

        else:
            constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
            constraint3 = [LinearConstraint(ee.copy(), 1, np.inf, keep_feasible=True) for ee in es]
            epsq = eps**2
            constraint2 = [NonlinearConstraint(lambda x,n=nn : self.SAPS[n].variance(x[mappings[n]],delta=delta), -np.inf, epsq[n], jac = lambda x,n=nn : self.SAPS[n].variance_GH(x[mappings[n]],nohess=True,delta=delta)[1], hess=lambda x,p,n=nn : self.SAPS[n].variance_GH(x[mappings[n]],delta=delta)[2]*p) for nn in range(No)]
            if x0 is None: x0 = np.ceil(np.linalg.norm(eps)**-2*np.random.rand(L))
            res = minimize(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=constraint2 + constraint3, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":2500, 'verbose':3}, tol = 1.0e-15)

        if budget is not None: res.x = res.x[1:]

        print(res.x.round())

        return res.x

    def ipopt_solve(self, budget=None, eps=None, x0=None):
        from cyipopt import minimize_ipopt
        from scipy.sparse import csr_matrix

        budget, eps = self.check_input(budget, eps)

        delta = 1.0e-6

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        def max_variance(m,delta=0):
            return max(self.variances(m,delta=delta))

        print("Optimizing using ipopt...")

        options = {"maxiter":200, 'print_level':5, 'print_user_options' : 'yes', 'bound_relax_factor' : 1.e-30, 'honor_original_bounds' : 'yes', 'dual_inf_tol' : 1.0e-1}

        eee = np.zeros((L+1,)); eee[0] = 1
        es = []
        for n in range(No):
            ee = np.zeros((L,))
            ee[mappings[n]] = e.copy()[mappings[n]]
            es.append(ee)

        #NOTE: super annoying trick for doing lambda functions inside list comprehensions, argh!
        if budget is not None:
            constraint1 = [(0, np.inf) for i in range(L+1)]
            constraint3 = [{'type':'ineq', 'fun': lambda x,ee=ees : ee@x[1:]-1, 'jac': lambda x,ee=ees : np.concatenate([np.zeros((1,)),ee]), 'hess': lambda x,p : csr_matrix((len(x),len(x)))} for ees in es]
            constraint2 = [{'type':'ineq', 'fun': lambda x : budget - w@x[1:], 'jac': lambda x : np.concatenate([np.zeros((1,)),-w]), 'hess': lambda x,p : csr_matrix((len(x),len(x)))}]
            constraint4 = [{'type':'ineq', 'fun': lambda x,nn=n : x[0] - self.SAPS[nn].variance(x[1:][mappings[nn]],delta=delta), 'jac' : lambda x,nn=n : np.concatenate([[1],-self.SAPS[nn].variance_GH(x[1:][mappings[nn]],nohess=True,delta=delta)[1]]), 'hess': lambda x,p,nn=n : csr_matrix(np.block([[0, np.zeros((1,len(x)-1))],[np.zeros((len(x)-1,1)), -self.SAPS[nn].variance_GH(x[1:][mappings[nn]],delta=delta)[2]]])*p)} for n in range(No)]

            if x0 is None: x0 = np.ceil(budget*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w); x0 = np.concatenate([[max_variance(x0,delta=delta)], x0])
            res = minimize_ipopt(lambda x : (x[0], eee), x0, jac=True, hess=lambda x : csr_matrix((len(x),len(x))), bounds=constraint1, constraints=constraint2+constraint3+constraint4, options=options, tol = 1.0e-12)

        else:
            meps = max(eps)
            eps = eps/meps
            epsq = eps**2
            constraint1 = [(0, np.inf) for i in range(L)]
            constraint3 = [{'type':'ineq', 'fun': lambda x,ee=ees : ee@x-1*meps**2, 'jac': lambda x,ee=ees : ee, 'hess': lambda x,p : csr_matrix((len(x),len(x)))} for ees in es]
            constraint2 = [{'type':'ineq', 'fun': lambda x,n=nn : epsq[n] - self.SAPS[n].variance(x[mappings[n]],delta=delta), 'jac': lambda x,n=nn : -self.SAPS[n].variance_GH(x[mappings[n]],nohess=True,delta=delta)[1], 'hess': lambda x,p,n=nn : csr_matrix(-self.SAPS[n].variance_GH(x[mappings[n]],delta=delta)[2]*p)} for nn in range(No)]
            if x0 is None: x0 = np.ceil(np.linalg.norm(eps)**-2*np.random.rand(L))
            res = minimize_ipopt(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hess=lambda x : csr_matrix((len(x),len(x))), bounds=constraint1, constraints=constraint2 + constraint3, options=options, tol = 1.0e-12)

        if budget is not None: res.x = res.x[1:]
        elif eps is not None: res.x *= meps**-2

        print(res.x.round())

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

    problem = SAP(C, KK, groups, costs)

    scipy_sol,cvxpy_sol,gurobi_sol = None,None,None
    cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
    scipy_sol  = problem.solve(budget=budget, solver="scipy")
    gurobi_sol = problem.solve(budget=budget, solver="gurobi")
    #gurobi_eps_sol = problem.solve(eps=eps, solver="gurobi")

    sols = [gurobi_sol, cvxpy_sol, scipy_sol]
    fvals = [problem.variance(sol) for sol in sols if sol is not None]

    print(fvals)
