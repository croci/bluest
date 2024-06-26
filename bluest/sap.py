import numpy as np
from itertools import combinations

import cvxpy as cp
from scipy.sparse import csr_matrix, bmat, find
from cvxopt import matrix,spmatrix,solvers

from .misc import assemble_psi,get_phi_full,variance_full,variance_GH_full,PHIinvY0,best_closest_integer_solution_BLUE,assemble_cleanup_matrix

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

cvxpy_cvxopt_default_params = {
        "abstol" : 1.e-7,
        "reltol" : 1.e-4,
        "max_iters" : 1000,
        "feastol" : 1.0e-6,
        "kttsolver" : 'chol',
        "refinement" : 1,
}

cvxpy_default_params = {
        "solver" : "CVXOPT",
        "solver_params" : cvxpy_cvxopt_default_params,
}

cvxopt_default_params = {
        "abstol" : 1.e-7,
        "reltol" : 1.e-4,
        "maxiters" : 1000,
        "feastol" : 1.0e-6,
        "refinement" : 1,
}

########################################################

def csr_to_cvxopt(A):
    l = find(A)
    out = spmatrix(l[-1],l[0],l[1], A.shape)
    return out

class SAP(object):
    def __init__(self, C, K, groups, costs, verbose=True):

        self.verbose = verbose

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
                idx = np.array([groupsk[i]], dtype=np.int64)
                index = (idx.T, idx)
                invcovs[k-1].append(np.linalg.pinv(C[index]))
                flattened_groups.append(groupsk[i])

            groups[k-1] = np.array(groups[k-1], dtype=np.int64)
            if len(invcovs[k-1]) > 0: invcovs[k-1] = np.vstack(invcovs[k-1]).flatten()
            else: invcovs[k-1] = np.array([])

        self.sizes            = sizes
        self.groups           = groups
        self.flattened_groups = flattened_groups
        self.invcovs          = invcovs
        self.cumsizes         = np.cumsum(sizes)
        self.L                = self.cumsizes[-1]

        #self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])
        ES = [[] for i in range(self.N)]
        for groupsk in groups:
            for group in groupsk:
                for i in range(self.N):
                    ES[i].append(int(i in group))
        self.ES = [np.array(es) for es in ES]
        self.e = self.ES[0]

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

        self.psi = np.hstack([assemble_psi(N,k,sizes[k],groups[k-1],invcovs[k-1]) for k in range(1,K+1) if len(groups[k-1]) > 0])

        def get_phi(m, delta=0):
            return get_phi_full(m, self.psi, delta=delta)
        def variance(m, delta=0):
            return variance_full(m, self.psi, groups, cumsizes, delta=delta)
        def variance_GH(m, delta=0, nohess=False):
            return variance_GH_full(m, self.psi, groups, sizes, invcovs, delta=delta, nohess=nohess)
        def get_cleanup_matrix(m, delta=0):
            return assemble_cleanup_matrix(m, self.psi, groups, sizes, invcovs, delta=delta)

        self.get_phi = get_phi
        self.variance = variance
        self.variance_GH = variance_GH
        self.get_cleanup_matrix = get_cleanup_matrix

    def integer_projection(self, samples, budget=None, eps=None, max_model_samples=None): 
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        if self.verbose: print("Integer projection...")

        def increase_tolerance(budget, eps, fac):
            if budget is None: b = None
            else:              b = budget*(1 + fac)
            if eps is None: e = None
            else:              e = np.sqrt(eps**2*(1+fac))
            return b,e

        ss = samples.copy()

        es,rhs = self.get_max_sample_constraints(max_model_samples)

        # STEP 0: standard
        samples,fval = best_closest_integer_solution_BLUE(ss, self.psi, self.costs, self.e, budget=budget, eps=eps, max_samples_info=(es,rhs))

        # STEP 1: increase tolerances
        if np.isinf(fval):
            for i in reversed(range(4)):
                if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found. Increasing the tolerance/budget.\n")
                fac = 10.**-i
                new_budget,new_eps = increase_tolerance(budget,eps,fac)
                samples,fval = best_closest_integer_solution_BLUE(ss, self.psi, self.costs, self.e, budget=budget, eps=eps, max_samples_info=(es,rhs))
                if not np.isinf(fval): break

        # STEP 2: Round up or down
        if np.isinf(fval):
            if max_model_samples is not None and not all([np.ceil(ss)@ee <= rr for ee,rr in zip(es,rhs)]):
                samples = np.floor(ss)
                if samples@self.e >= 1.0:
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding down to satisfy max model sample constraints.\n")
                else:
                    samples = np.ceil(ss)
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget and the max model sample constraints could not be satisfied. Rounding up.\n")
            else:
                if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding up.\n")
                samples = np.ceil(ss)

        return samples.astype(int)
    
    def solve(self, budget=None, eps=None, solver="cvxpy", x0=None, continuous_relaxation=False, max_model_samples=None, solver_params=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["scipy", "cvxpy", "ipopt", "cvxopt"]:
            raise ValueError("Optimization solvers available: 'scipy', 'ipopt', 'cvxopt', or 'cvxpy'")

        if self.verbose:
            if eps is None: print("Minimizing statistical error for fixed cost...\n")
            else:           print("Minimizing cost given statistical error tolerance...\n")

        if   solver == "cvxpy":  samples = self.cvxpy_solve(budget=budget, eps=eps, max_model_samples=max_model_samples, cvxpy_params=solver_params)
        elif solver == "cvxopt": samples = self.cvxopt_solve(budget=budget, eps=eps, max_model_samples=max_model_samples, cvxopt_params=solver_params)
        elif solver == "scipy":  samples = self.scipy_solve(budget=budget, eps=eps, x0=x0, max_model_samples=max_model_samples)
        elif solver == "ipopt":  samples = self.ipopt_solve(budget=budget, eps=eps, x0=x0, max_model_samples=max_model_samples)

        if samples is None:
            self.samples = None
            return None

        if not continuous_relaxation:
            try: samples = self.integer_projection(samples, budget=budget, eps=eps, max_model_samples=max_model_samples)
            except AssertionError as e:
                print(str(e))
                self.samples = None
                return None

        self.samples = samples
        self.budget = budget
        self.eps = eps
        self.tot_cost = samples@self.costs

        return samples

    def get_max_sample_constraints(self, max_model_samples):
        if max_model_samples is None:
            return [],[]
        
        if not isinstance(max_model_samples, np.ndarray) or len(max_model_samples) != self.N:
            raise ValueError("The maximum number of model samples must be prescribed as a numpy array of the same length as the number of models.")

        if max_model_samples[0] < 1:
            raise ValueError("The high-fidelity model must be sampled at least once.")

        es  = []
        rhs = []
        for i in range(self.N):
            if np.isfinite(max_model_samples[i]):
                nmax = int(np.round(max_model_samples[i]))
                es.append(self.ES[i])
                rhs.append(nmax)

        return es,rhs

    def cvxopt_solve(self, budget=None, eps=None, delta=0.0, max_model_samples=None, cvxopt_params=None):
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

        es,rhs = self.get_max_sample_constraints(max_model_samples)

        scales = 1/abs(psi).sum(axis=0).mean()

        if budget is not None:
            wt = np.concatenate([[0], w])
            et = np.concatenate([[0], e])
            if len(es) > 0:
                est = [np.concatenate([[0], -ees]) for ees in es]
                et = np.vstack([et] + est)

            c = np.zeros((L+1,)); c[0] = 1.; c = matrix(c)

            G0 = csr_matrix(np.vstack([-np.eye(L+1),wt,-et])); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L+1,)), [1.], [-1./budget] + [item/budget for item in rhs]]); h0 = matrix(h0)

            # NOTE: need to add a zero row every N entries since the last column of Phi is zero. Also here need to add a column to the left corresponding to the variable t
            l = list(find(psi)); l[0] += l[0]//N; l[1] += 1; l[-1] = -np.concatenate([scales*l[-1], [1]]); l[0] = np.concatenate([l[0],[(N+1)**2-1]]); l[1] = np.concatenate([l[1],[0]])
            G1 = csr_matrix((l[-1],(l[0],l[1])), shape=((N+1)**2,L+1)); G1 = csr_to_cvxopt(G1)
            h1 = np.zeros((N+1,N+1)); h1[-1,0] = np.sqrt(scales); h1[0,-1] = np.sqrt(scales); h1 = matrix(h1)
        else:
            c = matrix(w/np.linalg.norm(w))
            if len(es) > 0:
                et = np.vstack([e] + [-item for item in es])

            G0 = csr_matrix(np.vstack([-np.eye(L),-et])); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L,)), [-1.], rhs]); h0 = matrix(h0)

            l = list(find(psi)); l[0] += l[0]//N; # NOTE: need to add a zero row every N entries since the last column of Phi is zero
            G1 = (-scales)*csr_matrix((l[-1],(l[0],l[1])), shape=((N+1)**2,L)); G1 = csr_to_cvxopt(G1)
            h1 = np.zeros((N+1,N+1)); h1[-1,0] = np.sqrt(scales)/eps; h1[0,-1] = np.sqrt(scales)/eps; h1[-1,-1] = 1; h1 = matrix(h1)

        cvxopt_solver_params['show_progress'] = self.verbose
        try: res = solvers.sdp(c,Gl=G0,hl=h0,Gs=[G1],hs=[h1],solver=None, options=cvxopt_solver_params)
        except ZeroDivisionError:
            return None

        if res["x"] is None:
            return None
        
        if self.verbose: print(res)

        if budget is not None:
            m = np.maximum(np.array(res["x"]).flatten()[1:],0)
            m *= budget
        else:
            m = np.maximum(np.array(res["x"]).flatten(),0)

        if self.verbose: print(m.round())


        return m

    def cvxpy_to_cvxopt(self,probdata):

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

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0, max_model_samples=None, cvxpy_params=None):

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        L        = self.L
        w        = self.costs
        e        = self.e

        if cvxpy_params is not None:
            if cvxpy_params["solver"] == "CVXOPT":
                cvxpy_solver_params = cvxpy_default_params.copy()
                cvxpy_solver_params["solver_params"].update(cvxpy_params["solver_params"])
            else:
                cvxpy_solver_params = cvxpy_params.copy()
                if cvxpy_solver_params.get("solver_params", None) is None:
                    cvxpy_solver_params["solver_params"] = {}
        else:
            cvxpy_solver_params = cvxpy_default_params.copy()

        es,rhs = self.get_max_sample_constraints(max_model_samples)

        #scalings = np.array([np.linalg.norm(self.psi[:,i]) for i in range(L)])
        scales = 1/abs(self.psi).sum(axis=0).mean()

        m = cp.Variable(L, nonneg=True)
        if budget is not None:
            t = cp.Variable(nonneg=True)
            obj = cp.Minimize(t)

            constraints = [w@m <= 1, m@e >= 1/budget, self.cvxpy_fun(m,t=t,eps=None) >> 0] + [m@ee <= rr/budget for ee,rr in zip(es,rhs)]
        else:
            obj = cp.Minimize((w/np.linalg.norm(w))@m)
            constraints = [m@e >= 1, self.cvxpy_fun(m,t=None,eps=eps) >> 0] + [m@ee <= rr for ee,rr in zip(es,rhs)]

        prob = cp.Problem(obj, constraints)

        #probdata, _, _ = prob.get_problem_data(cp.CVXOPT)
        #breakpoint()

        #prob.solve(verbose=self.verbose, solver="MOSEK", mosek_params=mosek_params)
        try:
            prob.solve(verbose=self.verbose, solver=cvxpy_solver_params["solver"], **cvxpy_solver_params["solver_params"])
        except (cp.SolverError,ZeroDivisionError):
            return None

        if m.value is None:
            return None

        if eps is None: m.value *= budget

        if self.verbose: print(m.value.round())

        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None, max_model_samples=None):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        delta = 0

        L = self.L
        w = self.costs
        e = self.e

        es,rhs = self.get_max_sample_constraints(max_model_samples)

        if self.verbose: print("Optimizing using scipy...")

        constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
        constraint3 = LinearConstraint(e, 1, np.inf, keep_feasible=True)
        constraint4 = [LinearConstraint(ee,-np.inf, rr) for ee,rr in zip(es,rhs)]
        if budget is not None:
            constraint2 = LinearConstraint(w, -np.inf, budget)

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize(lambda x : self.variance_GH(x,nohess=True,delta=delta)[:-1], x0, jac=True, hess=lambda x : self.variance_GH(x,delta=delta)[-1], bounds=constraint1, constraints=[constraint2,constraint3] + constraint4, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3*int(self.verbose)}, tol = 1.0e-8)

        else:
            epsq = eps**2
            constraint2 = NonlinearConstraint(lambda x : self.variance(x,delta=delta), epsq, epsq, jac = lambda x : self.variance_GH(x,nohess=True,delta=delta)[1], hess=lambda x,p : self.variance_GH(x,delta=delta)[2]*p)
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2,constraint3] + constraint4, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3*int(self.verbose)}, tol = 1.0e-10)

        return res.x

    def ipopt_solve(self, budget=None, eps=None, x0=None, max_model_samples=None):
        from cyipopt import minimize_ipopt

        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        delta = 1.0e-6

        L = self.L
        w = self.costs
        e = self.e

        es,rhs = self.get_max_sample_constraints(max_model_samples)

        if self.verbose: print("Optimizing using ipopt...")

        options = {"maxiter":200, 'print_level':5*int(self.verbose), 'print_user_options' : ['no','yes'][int(self.verbose)], 'bound_relax_factor' : 1.e-30, 'honor_original_bounds' : 'yes', 'dual_inf_tol' : 1.e-5}

        constraint1 = [(0, np.inf) for i in range(L)]
        constraint3 = {'type':'ineq', 'fun': lambda x : e@x-1, 'jac': lambda x : e, 'hess': lambda x,p : 0}
        constraint4 = [{'type':'ineq', 'fun': lambda x : rr-ee@x, 'jac': lambda x : -ee, 'hess': lambda x,p : 0} for ee,rr in zip(es,rhs)]
        if budget is not None:
            constraint2 = {'type':'ineq', 'fun': lambda x : budget - w@x, 'jac': lambda x : -w, 'hess': lambda x,p : 0}

            if x0 is None: x0 = np.ceil(10*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w)
            res = minimize_ipopt(lambda x : self.variance_GH(x,nohess=True,delta=delta)[:-1], x0, jac=True, hess=lambda x : self.variance_GH(x,delta=delta)[-1], bounds=constraint1, constraints=[constraint2,constraint3] + constraint4, options=options, tol = 1.0e-12)

        else:
            epsq = eps**2
            #constraint2 = NonlinearConstraint(lambda x : self.variance(x,delta=delta), epsq, epsq, jac = lambda x : self.variance_GH(x,nohess=True,delta=delta)[1], hess=lambda x,p : self.variance_GH(x,delta=delta)[2]*p)
            constraint2 = {'type':'ineq', 'fun': lambda x : epsq - self.variance(x,delta=delta), 'jac': lambda x : -self.variance_GH(x,nohess=True,delta=delta)[1], 'hess': lambda x,p : -self.variance_GH(x,delta=delta)[2]*p}
            if x0 is None: x0 = np.ceil(eps**-2*np.random.rand(L))
            res = minimize_ipopt(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hess=lambda x : np.zeros((len(x),len(x))), bounds=constraint1, constraints=[constraint2,constraint3] + constraint4, options=options, tol = 1.0e-12)

        if self.verbose: print(res.x.round())

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

    max_model_samples = np.inf*np.ones((N,)); max_model_samples[-4:] = 10.**(2*np.arange(4))

    scipy_sol,cvxpy_sol,ipopt_sol,cvxopt_sol = None,None,None,None
    if False:
        cvxopt_sol = problem.solve(eps=eps, max_model_samples=max_model_samples, solver="cvxopt")
        cvxpy_sol  = problem.solve(eps=eps, max_model_samples=max_model_samples, solver="cvxpy")
        ipopt_sol  = problem.solve(eps=eps, max_model_samples=max_model_samples, solver="ipopt")
        scipy_sol  = problem.solve(eps=eps, max_model_samples=max_model_samples, solver="scipy")
        print("MSE tolerance: ", eps**2)
    else:
        cvxopt_sol = problem.solve(budget=budget, max_model_samples=max_model_samples, solver="cvxopt")
        cvxpy_sol  = problem.solve(budget=budget, max_model_samples=max_model_samples, solver="cvxpy")
        ipopt_sol  = problem.solve(budget=budget, max_model_samples=max_model_samples, solver="ipopt")
        scipy_sol  = problem.solve(budget=budget, max_model_samples=max_model_samples, solver="scipy")
        print("Budget: ", budget)

    sols = [cvxopt_sol, cvxpy_sol, ipopt_sol, scipy_sol]
    fvals = [(costs@sol, problem.variance(sol)) for sol in sols if sol is not None]

    es,rhs = problem.get_max_sample_constraints(max_model_samples)
    assert all([[ee@sol <= rr for ee,rr in zip(es,rhs)] for sol in sols])

    print(fvals)
