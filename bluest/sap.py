import numpy as np
from itertools import combinations

import cvxpy as cp

from .misc import assemble_psi,get_phi_full,variance_full,variance_GH_full,PHIinvY0,best_closest_integer_solution_BLUE

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

cvxpy_default_params = {
        "abstol" : 1.e-7,
        "reltol" : 1.e-4,
        "max_iters" : 1000,
        "feastol" : 1.0e-6,
        "kttsolver" : 'chol',
        "refinement" : 2,
}

cvxopt_default_params = {
        "abstol" : 1.e-7,
        "reltol" : 1.e-4,
        "maxiters" : 1000,
        "feastol" : 1.0e-6,
        "refinement" : 1,
}

########################################################

class SAP(object):
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
                invcovs[k-1].append(np.linalg.pinv(C[index]))
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

        y = [0 for i in range(L)]
        sums = [sums[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
        for k in range(1, K+1):
            for i in range(sizes[k]):
                for j in range(k):
                    for s in range(k):
                        y[groups[k-1][i][j]] += invcovs[k-1][k*k*i + k*j + s]*sums[k-1][i][s]

        return PHIinvY0(samples, y, self.psi, groups, cumsizes)

    def get_variance_functions(self):
        N        = self.N
        K        = self.K
        groups   = self.groups
        invcovs  = self.invcovs
        sizes    = self.sizes
        cumsizes = self.cumsizes

        self.psi = np.hstack([assemble_psi(N,k,sizes[k],groups[k-1],invcovs[k-1]) for k in range(1,K+1)])

        def get_phi(m, delta=0):
            return get_phi_full(m, self.psi, delta=delta)
        def variance(m, delta=0):
            return variance_full(m, self.psi, groups, cumsizes, delta=delta)
        def variance_GH(m, delta=0, nohess=False):
            return variance_GH_full(m, self.psi, groups, sizes, invcovs, delta=delta, nohess=nohess)

        self.get_phi = get_phi
        self.variance = variance
        self.variance_GH = variance_GH

    def solve(self, budget=None, eps=None, solver="cvxpy", integer=False, x0=None, solver_params=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["gurobi", "scipy", "cvxpy", "ipopt", "cvxopt"]:
            raise ValueError("Optimization solvers available: 'gurobi', 'scipy', 'ipopt' or 'cvxpy'")
        if integer: solver="gurobi"

        if eps is None:
            print("Minimizing statistical error for fixed cost...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(budget=budget, integer=integer)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(budget=budget, cvxpy_params=solver_params)
            elif solver == "cvxopt": samples = self.cvxopt_solve(budget=budget, cvxopt_params=solver_params)
            elif solver == "scipy":  samples = self.scipy_solve(budget=budget, x0=x0)
            elif solver == "ipopt":  samples = self.ipopt_solve(budget=budget, x0=x0)

            if not integer:
                ss = samples.copy()
                samples,fval = best_closest_integer_solution_BLUE(samples, self.psi, self.costs, self.e, budget=budget)
                if np.isinf(fval):
                    print("WARNING! An integer solution satisfying the constraints could not be found. Running Gurobi optimizer with integer constraints.\n")
                    samples = self.gurobi_solve(budget=budget, integer=True)

        else:
            print("Minimizing cost given statistical error tolerance...\n")
            if   solver == "gurobi": samples = self.gurobi_solve(eps=eps, integer=integer)
            elif solver == "scipy":  samples = self.scipy_solve(eps=eps, x0=x0)
            elif solver == "ipopt":  samples = self.ipopt_solve(eps=eps, x0=x0)
            elif solver == "cvxpy":  samples = self.cvxpy_solve(eps=eps, cvxpy_params=solver_params)
            elif solver == "cvxopt": samples = self.cvxopt_solve(eps=eps, cvxopt_params=solver_params)

            if not integer:

                samples,fval = best_closest_integer_solution_BLUE(samples, self.psi, self.costs, self.e, eps=eps)

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

    def gurobi_solve(self, budget=None, eps=None, integer=False, extra_bounds=None):
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

        if extra_bounds is not None:
            M.addConstrs((m[i] >= extra_bounds[0][i] for i in range(L)))
            M.addConstrs((m[i] <= extra_bounds[1][i] for i in range(L)))
            m.vType = GRB.INTEGER

            hardlimit = 60
            tol = 1.0e-3

            M.setParam('TimeLimit', hardlimit)
            M.setParam('BestObjStop', (1+tol)*extra_bounds[2])

        # enforcing the constraint that PHI^{-1}[:,0] = t
        constr = self.gurobi_constraint(m,t)
        M.addConstr(constr[0] == 1)
        M.addConstrs((constr[i] == 0 for i in range(1,N)))

        M.optimize()

        return np.array(m.X)

    def cvxopt_solve(self, budget=None, eps=None, delta=0.0, cvxopt_params=None):
        from scipy.sparse import csr_matrix, bmat, find
        from cvxopt import matrix,spmatrix,solvers

        def csr_to_cvxopt(A):
            l = find(A)
            out = spmatrix(l[-1],l[0],l[1], A.shape)
            return out

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        N        = self.N
        L        = self.L
        w        = self.costs.copy()
        e        = self.e
        psi      = csr_matrix(self.psi); psi.eliminate_zeros()

        cvxopt_solver_params = cvxopt_default_params.copy()
        if cvxopt_params is not None:
            cvxopt_solver_params.update(cvxopt_params)

        scales = 1/abs(psi).sum(axis=0).mean()

        if budget is not None:
            wt = np.concatenate([[0], w])
            et = np.concatenate([[0], e])

            c = np.zeros((L+1,)); c[0] = 1.; c = matrix(c)

            G0 = csr_matrix(np.vstack([-np.eye(L+1),wt,-et])); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L+1,)), [1.], [-1./budget]]); h0 = matrix(h0)

            # NOTE: need to add a zero row every N entries since the last column of Phi is zero. Also here need to add a column to the left corresponding to the variable t
            l = list(find(psi)); l[0] += l[0]//N; l[1] += 1; l[-1] = -np.concatenate([scales*l[-1], [1]]); l[0] = np.concatenate([l[0],[(N+1)**2-1]]); l[1] = np.concatenate([l[1],[0]])
            G1 = csr_matrix((l[-1],(l[0],l[1])), shape=((N+1)**2,L+1)); G1 = csr_to_cvxopt(G1)
            h1 = np.zeros((N+1,N+1)); h1[-1,0] = np.sqrt(scales); h1[0,-1] = np.sqrt(scales); h1 = matrix(h1)
        else:
            c = matrix(w/np.linalg.norm(w))

            G0 = csr_matrix(np.vstack([-np.eye(L),-e])); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L,)), [-1.]]); h0 = matrix(h0)

            l = list(find(psi)); l[0] += l[0]//N; # NOTE: need to add a zero row every N entries since the last column of Phi is zero
            G1 = (-scales)*csr_matrix((l[-1],(l[0],l[1])), shape=((N+1)**2,L)); G1 = csr_to_cvxopt(G1)
            h1 = np.zeros((N+1,N+1)); h1[-1,0] = np.sqrt(scales)/eps; h1[0,-1] = np.sqrt(scales)/eps; h1[-1,-1] = 1; h1 = matrix(h1)

        res = solvers.sdp(c,Gl=G0,hl=h0,Gs=[G1],hs=[h1],solver=None, options=cvxopt_solver_params)
        
        print(res)

        if budget is not None:
            m = np.maximum(np.array(res["x"]).flatten()[1:],0)
            m *= budget
        else:
            m = np.maximum(np.array(res["x"]).flatten(),0)

        print(m.round())


        return m

    def cvxpy_to_cvxopt(self,probdata):
        from scipy.sparse import csr_matrix, bmat, find
        from cvxopt import matrix,spmatrix,solvers

        def csr_to_cvxopt(A):
            l = find(A)
            out = spmatrix(l[-1],l[0],l[1], A.shape)
            return out

        c = matrix(probdata['c'])
        G = csr_to_cvxopt(probdata['G'])
        h = matrix(probdata['h'])
        dims_tup = vars(probdata['dims'])
        dims = {'l' : dims_tup['nonneg'], 'q': [], 's': dims_tup['psd']}

        res = solvers.conelp(c, G, h, dims, options=cvxopt_default_params)
        return res

    def cvxpy_fun(self, m, t=None, eps=None):
        if t is None and eps is None: raise ValueError("eps and t cannot both be None")
        if t is None: t = 1
        if eps is None: eps = 1
        
        N = self.N;
        scales = 1/abs(self.psi).sum(axis=0).mean()
        PHI = cp.reshape((self.psi*scales)@m, (N,N))
        ee = np.zeros((N,1)); ee[0] = np.sqrt(scales)/eps
        return cp.bmat([[PHI,ee],[ee.T,cp.reshape(t,(1,1))]])

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0, cvxpy_params=None):

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        L        = self.L
        w        = self.costs
        e        = self.e

        cvxpy_solver_params = cvxpy_default_params.copy()
        if cvxpy_params is not None:
            cvxpy_solver_params.update(cvxpy_params)

        #scalings = np.array([np.linalg.norm(self.psi[:,i]) for i in range(L)])
        scales = 1/abs(self.psi).sum(axis=0).mean()

        m = cp.Variable(L, nonneg=True)
        if budget is not None:
            t = cp.Variable(nonneg=True)
            obj = cp.Minimize(t)

            constraints = [w@m <= 1, m@e >= 1/budget, self.cvxpy_fun(m,t=t,eps=None) >> 0]
        else:
            obj = cp.Minimize((w/np.linalg.norm(w))@m)
            constraints = [m@e >= 1, self.cvxpy_fun(m,t=None,eps=eps) >> 0]

        prob = cp.Problem(obj, constraints)

        #probdata, _, _ = prob.get_problem_data(cp.CVXOPT)
        #breakpoint()

        #prob.solve(verbose=True, solver="MOSEK", mosek_params=mosek_params)
        prob.solve(verbose=True, solver="CVXOPT", **cvxpy_solver_params)

        if eps is None: m.value *= budget

        print(m.value.round())

        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        delta = 0

        L = self.L
        w = self.costs
        e = self.e

        print("Optimizing using scipy...")

        constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint3 = LinearConstraint(e, 1, np.inf, keep_feasible=True)
        if budget is not None:
            constraint2 = LinearConstraint(w, -np.inf, budget)

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize(lambda x : self.variance_GH(x,nohess=True,delta=delta)[:-1], x0, jac=True, hess=lambda x : self.variance_GH(x,delta=delta)[-1], bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3}, tol = 1.0e-8)

        else:
            epsq = eps**2
            constraint2 = NonlinearConstraint(lambda x : self.variance(x,delta=delta), epsq, epsq, jac = lambda x : self.variance_GH(x,nohess=True,delta=delta)[1], hess=lambda x,p : self.variance_GH(x,delta=delta)[2]*p)
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2,constraint3], method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3}, tol = 1.0e-10)

        return res.x

    def ipopt_solve(self, budget=None, eps=None, x0=None):
        from cyipopt import minimize_ipopt

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        delta = 1.0e-6

        L = self.L
        w = self.costs
        e = self.e

        print("Optimizing using ipopt...")

        options = {"maxiter":200, 'print_level':5, 'print_user_options' : 'yes', 'bound_relax_factor' : 1.e-30, 'honor_original_bounds' : 'yes', 'dual_inf_tol' : 1.e-5}

        constraint1 = [(0, np.inf) for i in range(L)]
        constraint3 = {'type':'ineq', 'fun': lambda x : e@x-1, 'jac': lambda x : e, 'hess': lambda x,p : 0}
        if budget is not None:
            constraint2 = {'type':'ineq', 'fun': lambda x : budget - w@x, 'jac': lambda x : -w, 'hess': lambda x,p : 0}

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize_ipopt(lambda x : self.variance_GH(x,nohess=True,delta=delta)[:-1], x0, jac=True, hess=lambda x : self.variance_GH(x,delta=delta)[-1], bounds=constraint1, constraints=[constraint2,constraint3], options=options, tol = 1.0e-12)

        else:
            epsq = eps**2
            #constraint2 = NonlinearConstraint(lambda x : self.variance(x,delta=delta), epsq, epsq, jac = lambda x : self.variance_GH(x,nohess=True,delta=delta)[1], hess=lambda x,p : self.variance_GH(x,delta=delta)[2]*p)
            constraint2 = {'type':'ineq', 'fun': lambda x : epsq - self.variance(x,delta=delta), 'jac': lambda x : -self.variance_GH(x,nohess=True,delta=delta)[1], 'hess': lambda x,p : -self.variance_GH(x,delta=delta)[2]*p}
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize_ipopt(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hess=lambda x : np.zeros((len(x),len(x))), bounds=constraint1, constraints=[constraint2,constraint3], options=options, tol = 1.0e-12)

        print(res.x.round())

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

    problem = SAP(C, KK, groups, costs)

    scipy_sol,cvxpy_sol,gurobi_sol,ipopt_sol,cvxopt_sol = None,None,None,None,None
    if False:
        cvxpy_sol  = problem.solve(eps=eps, solver="cvxpy")
        cvxopt_sol = problem.solve(eps=eps, solver="cvxopt")
        #ipopt_sol  = problem.solve(eps=eps, solver="ipopt")
        #scipy_sol  = problem.solve(eps=eps, solver="scipy")
        #gurobi_sol = problem.solve(eps=eps, solver="gurobi")
        print("MSE tolerance: ", eps**2)
    else:
        cvxpy_sol  = problem.solve(budget=budget, solver="cvxpy")
        cvxopt_sol = problem.solve(budget=budget, solver="cvxopt")
        #ipopt_sol  = problem.solve(budget=budget, solver="ipopt")
        #scipy_sol  = problem.solve(budget=budget, solver="scipy")
        #gurobi_sol = problem.solve(budget=budget, solver="gurobi")
        print("Budget: ", budget)

    sols = [gurobi_sol, ipopt_sol, cvxpy_sol, scipy_sol, cvxopt_sol]
    fvals = [(costs@sol, problem.variance(sol)) for sol in sols if sol is not None]

    print(fvals)
