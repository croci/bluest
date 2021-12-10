import numpy as np
from itertools import combinations, product
from .blue_opt import BLUESampleAllocationProblem,best_closest_integer_solution, mosek_params

import cvxpy as cp
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

class BLUEMultiObjectiveSampleAllocationProblem(object):
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

        self.sizes = [0] + [len(groupsk) for groupsk in groups]
        self.cumsizes = np.cumsum(self.sizes)
        self.L = self.cumsizes[-1]
        self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])

        mappings = [[[] for k in range(Ks[n])] for n in range(self.n_outputs)]
        for n in range(self.n_outputs):
            for k in range(1, Ks[n]+1):
                groupsk = multi_groups[n][k-1]
                for i in range(len(groupsk)):
                    pos = [j for j,item in enumerate(self.groups[k-1]) if groupsk[i] == item]
                    assert len(pos) == 1
                    mappings[n][k-1].append(pos[0])

                invcovs[n][k-1]   = np.vstack(invcovs[n][k-1]).flatten()
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

        out = [self.SAPS[n].variance_with_grad_and_hess(m[mappings[n]],nohess=nohess,delta=delta) for n in range(No)]
        variances = [item[0] for item in out]
        gradients = [item[1] for item in out]
        hessians  = [item[2] for item in out]

        return variances,gradients,hessians

    def compute_BLUE_estimators(self, sums):
        out  = [self.SAPS[n].compute_BLUE_estimator(sums[self.mappings[n]]) for n in range(self.n_outputs)]
        mus  = np.array([item[0] for item in out])
        Vars = np.array([item[1] for item in out])
        return mus,Vars

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
                objective  = lambda m : max(self.variances(m))
                
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
                constraint = lambda m : m@self.e >= 1 and all(np.array(self.variances(m)) <= np.array(eps)**2)

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
        budget, eps = self.check_input(budget, eps)

        from gurobipy import Model,GRB

        L        = self.L
        N        = self.N
        No       = self.n_outputs
        w        = self.costs
        e        = self.e

        M = Model("BLUE")
        M.params.NonConvex = 2

        m = M.addMVar(shape=(int(L),), lb=np.zeros((L,)), ub=np.ones((L,))*np.inf,vtype=GRB.CONTINUOUS, name="m")
        if integer: m.vType = GRB.INTEGER

        t = [M.addMVar(shape=(N,), vtype=GRB.CONTINUOUS, name="t%d" % n) for n in range(No)]

        if eps is not None:
            M.setObjective(m@w, GRB.MINIMIZE)
            M.addConstrs((t[n][0] <= eps[n]**2 for n in range(No)), name="variance")
            M.addConstr(m@e >= 1, name="minimum_samples")
        else:
            tt = M.addVar(vtype=GRB.CONTINUOUS, name="tmax")
            M.setObjective(tt, GRB.MINIMIZE)
            M.addConstr(m@w <= budget, name="budget")
            M.addConstr(m@e >= 1, name="minimum_samples")
            M.addConstrs((tt >= t[n][0] for n in range(No)), name="max_constraint")

        # enforcing the constraint that PHI^{-1}[:,0] = t
        for n in range(No):
            constr = gurobi_constraint(m,t[n])
            M.addConstr(constr[0] == 1)
            M.addConstrs((constr[i] == 0 for i in range(1,N)))

        M.optimize()

        return np.array(m.X)

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0):
        budget, eps = self.check_input(budget, eps)

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        m = cp.Variable(L)
        t = cp.Variable(No)
        if budget is not None:
            obj = cp.Minimize(cp.max(t))
            constraints = [m >= 0.0*np.ones((L,)), w@m <= budget, m@e >= 1]
            constraints += [self.SAPS[n].cvxpy_fun(m[mappings[n]],t[n],delta=0) >> 0 for n in range(No)]
        else:
            obj = cp.Minimize(w@m)
            constraints = [m >= 0.0*np.ones((L,)), m@e >= 1, t <= eps**2]
            constraints += [self.SAPS[n].cvxpy_fun(m[mappings[n]],t[n],delta=0) >> 0 for n in range(No)]
        prob = cp.Problem(obj, constraints)
        
        #prob.solve(verbose=True, solver="MOSEK", mosek_params=mosek_params)
        prob.solve(verbose=True, solver="CVXOPT", abstol=1.0e-8, reltol=1.e-6, max_iters=1000, feastol=1.0e-4, kttsolver='chol',refinement=2)

        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None):
        budget, eps = self.check_input(budget, eps)

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        #NOTE: leave this for now, but if you want the max can just do min t where t >= var[n] for all n
        def variance(m,delta=0):
            return sum(self.variances(m,delta=delta))

        def variance_GH(m,nohess=False,delta=0):
            out = self.variance_GH(m,nohess=nohess,delta=delta)
            return (sum(out[i]) for i in range(3-int(nohess)))

        print("Optimizing using scipy...")

        constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint3 = LinearConstraint(e, 1, np.inf, keep_feasible=True)
        if budget is not None:
            constraint2 = LinearConstraint(w, -np.inf, budget)

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize(lambda x : variance_GH(x,nohess=True,delta=0), x0, jac=True, hess=lambda x : variance_GH(x,delta=0)[-1], bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3}, tol = 1.0e-8)

        else:
            epsq = eps**2
            constraint2 = [NonlinearConstraint(lambda x : self.SAPS[n].variance(x[mappings[n]],delta=0), epsq, epsq, jac = lambda x : self.SAPS[n].variance_with_grad_and_hess(x[mappings[n]],nohess=True,delta=0)[1], hess=lambda x,p : self.SAPS[n].variance_with_grad_and_hess(x[mappings[n]],delta=0)[2]*p) for n in range(No)]
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize(lambda x : [w@x,w], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=constraint2 + [constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":10000, 'verbose':3}, tol = 1.0e-6)


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
