import numpy as np
from itertools import combinations, product
from .sap import SAP,mosek_params,cvxpy_default_params,cvxopt_default_params
from .misc import best_closest_integer_solution_BLUE_multi

import cvxpy as cp
from scipy.sparse import csr_matrix, bmat, find
from cvxopt import matrix,spmatrix,solvers

def csr_to_cvxopt(A):
    l = find(A)
    out = spmatrix(l[-1],l[0],l[1], A.shape)
    return out

class BLUESTError(RuntimeError):
    pass

class MOSAP(object):
    ''' MOSAP, MultiObjectiveSampleAllocationProblem '''
    def __init__(self, C, K, Ks, groups, multi_groups, costs, multi_costs, verbose=True):
        self.verbose = verbose

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

        self.SAPS = [SAP(C[n], Ks[n], multi_groups[n], multi_costs[n], verbose=self.verbose) for n in range(self.n_outputs)]

        self.sizes = [0] + [len(groupsk) for groupsk in groups]
        self.cumsizes = np.cumsum(self.sizes)
        self.L = self.cumsizes[-1]

        #self.e = np.array([int(0 in group) for groupsk in groups for group in groupsk])
        ES = [[] for i in range(self.N)]
        for groupsk in groups:
            for group in groupsk:
                for i in range(self.N):
                    ES[i].append(int(i in group))
        self.ES = [np.array(es) for es in ES]
        self.e = self.ES[0]

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

    def get_cleanup_matrices(self, m, delta=0):
        No       = self.n_outputs
        mappings = self.mappings
        Xs = []
        for n in range(No):
            X = np.zeros((self.N, self.L))
            X[:,mappings[n]] = self.SAPS[n].get_cleanup_matrix(m[mappings[n]],delta=delta)
            Xs.append(X)

        return np.vstack(Xs)

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

    def cleanup_solution(self, m, delta=0, tol=0):
        # Attempts to find a solution that is sparser without changing the variance or the total cost

        from scipy.linalg import null_space
        # E is the matrix of the es constructed from self.e and the mappings in mosap.py
        N = self.N
        L = self.L
        w = self.costs
        No = self.n_outputs
        mappings = self.mappings

        E = []
        for n in range(No):
            ee = np.zeros((L,))
            ee[mappings[n]] = self.e.copy()[mappings[n]]
            E.append(ee)
        E = np.vstack(E)


        idx = np.argwhere(m > tol).flatten()
        m0 = m.copy(); mlast = m.copy(); V0 = max(self.variances(m, delta=delta))
        it = 0; nullsize = -1; V = V0*1
        if self.verbose:
            print("\nSolution cleanup started!")
            print("It %3d: Solution cleanup, L = %d, N = %d, nnz = %d, nullspace size = n/a, variance = %e." % (it, L, N, len(idx), V)) 
        while len(idx) > N:
            idx = np.argwhere(m > tol).flatten()
            m[m < tol] = 0
            wr = w[idx]
            Er = E[:,idx]

            if it > 0 and L >= 1000 and self.verbose: print("It %3d: Solution cleanup, L = %d, N = %d, nnz = %d, nullspace size = %3d, variance = %e." % (it, L, N, len(idx), nullsize, V)) 

            it += 1

            X = self.get_cleanup_matrices(m, delta=delta)

            X = X[:,idx]
            NN = null_space(X)
            vals = wr@NN
            signs = np.sign(vals)
            NN[:, signs > 0] *= -1
            vals[signs > 0] *= -1
            NN   = NN[:, abs(signs) > 0]
            vals = vals[abs(signs) > 0]
            indexes = np.argsort(abs(vals))[::-1]
            nullsize = len(vals)
            if nullsize == 0: break
            em = Er@m[idx]

            for j in range(nullsize):
                i = indexes[j]
                t = NN[:,i]
                evals = Er@t
                negidx = np.argwhere(evals < 0).flatten()
                if len(negidx) == 0: smax1 = np.inf
                else:                smax1 = min(abs(em[negidx]-1)/abs(evals[negidx]))
                negidx = np.argwhere(t < 0).flatten()
                if len(negidx) == 0: smax2 = np.inf
                else:                smax2 = min(m[idx][negidx]/abs(t[negidx]))
                smax = max(min(smax1,smax2),0)
                if smax > 5*tol:
                    tt = np.zeros_like(m); tt[idx] = t
                    mnew = m + smax*tt
                    V = max(self.variances(mnew,delta=delta))
                    if V < V0 or abs(V-V0)/abs(V0) < 1.0e-4:
                        m = mnew.copy()
                        break;
                    else:
                        smax = 0
                        continue

            if smax <= 5*tol:
                break
            else:
                V = max(self.variances(m,delta=delta))
                mlast = m.copy()

        idx = np.argwhere(m > tol).flatten()
        m[m < tol] = 0
        V = max(self.variances(m,delta=delta))
        if self.verbose:
            print("It %3d: Solution cleanup, L = %d, N = %d, nnz = %d, nullspace size = %3d, variance = %e." % (it, L, N, len(idx), nullsize, V)) 
            print("Solution cleanup completed.\n") 

        return m

    def integer_projection(self, samples, budget=None, eps=None, max_model_samples=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")

        if self.verbose: print("Integer projection...")

        def increase_tolerance(budget, eps, fac):
            if budget is None: b = None
            else:              b = budget*(1 + fac)
            if eps is None: e = None
            else:              e = np.sqrt(np.array(eps)**2*(1+fac))
            return b,e

        ss = samples.copy()

        ES,rhs = self.get_max_sample_constraints(max_model_samples)

        # STEP 0: standard
        samples,fval = best_closest_integer_solution_BLUE_multi(ss, [self.SAPS[n].psi for n in range(self.n_outputs)], self.costs, self.e, self.mappings, budget=budget, eps=eps, max_samples_info=(ES,rhs))

        # STEP 1: cleanup + standard
        if np.isinf(fval):
            if self.verbose: print("Integer projection failed. Trying to recover by cleanup...")
            css = self.cleanup_solution(ss)
            samples,fval = best_closest_integer_solution_BLUE_multi(css, [self.SAPS[n].psi for n in range(self.n_outputs)], self.costs, self.e, self.mappings, budget=budget, eps=eps, max_samples_info=(ES,rhs))

        # STEP 2: increase tolerances
        if np.isinf(fval):
            for i in reversed(range(4)):
                if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found. Increasing the tolerance/budget.\n")
                fac = 10.**-i
                new_budget,new_eps = increase_tolerance(budget,eps,fac)

                samples,fval = best_closest_integer_solution_BLUE_multi(ss, [self.SAPS[n].psi for n in range(self.n_outputs)],  self.costs, self.e, self.mappings, budget=new_budget, eps=new_eps, max_samples_info=(ES,rhs))
                if np.isinf(fval): samples,fval = best_closest_integer_solution_BLUE_multi(css, [self.SAPS[n].psi for n in range(self.n_outputs)], self.costs, self.e, self.mappings, budget=new_budget, eps=new_eps, max_samples_info=(ES,rhs))
                if not np.isinf(fval): break

        # STEP 3: Round up/down
        if np.isinf(fval):
            ssf  = np.floor(ss);  tot_cost_ssf = ssf@self.costs; var_ssf = max(self.variances(ssf))
            ss   = np.ceil(ss);   tot_cost_ss  = ss@self.costs;   var_ss = max(self.variances(ss))

            cssf = np.floor(css); tot_cost_cssf = cssf@self.costs; var_cssf = max(self.variances(cssf))
            css  = np.ceil(css);  tot_cost_css  = css@self.costs;   var_css = max(self.variances(css))

            if max_model_samples is not None:
                check1 = all([ ss@ees <= rr for ees,rr in zip(ES,rhs)])
                check2 = all([css@ees <= rr for ees,rr in zip(ES,rhs)])
                if check1:
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding up.\n")
                    samples = ss
                elif check2:
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding up.\n")
                    samples = css
                elif all([ssf[self.mappings[n]]@self.e[self.mappings[n]] >= 1 for n in range(self.n_outputs)]):
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding down to satisfy max model sample constraints.\n")
                    samples = ssf
                elif all([cssf[self.mappings[n]]@self.e[self.mappings[n]] >= 1 for n in range(self.n_outputs)]):
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding down to satisfy max model sample constraints.\n")
                    samples = cssf
                else:
                    if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget and the max model sample constraints could not be satisfied. Rounding up.\n")
                    if eps is None:
                        if tot_cost_ss < tot_cost_css: samples = ss
                        else:                          samples = css
                    else:
                        if var_ss < var_css: samples = ss
                        else:                samples = css
            else:
                if self.verbose: print("WARNING! An integer solution satisfying the constraints could not be found even after increasing the tolerance/budget. Rounding up.\n")
                if eps is None:
                    if tot_cost_ss < tot_cost_css: samples = ss
                    else:                          samples = css
                else:
                    if var_ss < var_css: samples = ss
                    else:                samples = css

        return samples.astype(int)

    def solve(self, budget=None, eps=None, solver="cvxpy", x0=None, continuous_relaxation=False, max_model_samples=None, solver_params=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        if solver not in ["scipy", "cvxpy", "ipopt", "cvxopt"]:
            raise ValueError("Optimization solvers available: 'scipy', 'ipopt', 'cvxopt' or 'cvxpy'")

        if self.verbose:
            if eps is None: print("Minimizing statistical error for fixed cost...\n")
            else:           print("Minimizing cost given statistical error tolerance...\n")

        if   solver == "cvxpy":  samples = self.cvxpy_solve(budget=budget, eps=eps, max_model_samples=max_model_samples, cvxpy_params=solver_params)
        elif solver == "cvxopt": samples = self.cvxopt_solve(budget=budget, eps=eps, max_model_samples=max_model_samples, cvxopt_params=solver_params)
        elif solver == "ipopt":  samples = self.ipopt_solve(budget=budget, eps=eps, x0=x0, max_model_samples=max_model_samples)
        elif solver == "scipy":  samples = self.scipy_solve(budget=budget, eps=eps, x0=x0, max_model_samples=max_model_samples)
        
        if samples is None:
            self.samples = None
            return None

        if not continuous_relaxation:
            samples = self.integer_projection(samples, budget=budget, eps=eps, max_model_samples=max_model_samples)

        self.samples = samples
        self.budget = budget
        self.eps = eps
        self.tot_cost = samples@self.costs
        for n in range(self.n_outputs):
            self.SAPS[n].samples = samples[self.mappings[n]]

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

    def cvxopt_get_sdp_constraints(self, budget=None, eps=None):
        No       = self.n_outputs
        N        = self.N
        L        = self.L
        mappings = self.mappings
        psis     = [csr_matrix(self.SAPS[n].psi) for n in range(No)];
        [psi.eliminate_zeros() for psi in psis]

        assert all(psi.shape[1] == len(mapping) for psi,mapping in zip(psis, mappings))

        scales = np.array([1/abs(psi).sum(axis=0).mean() for psi in psis])

        Gs = []
        hs = []
        for n in range(No):
            Nn = self.SAPS[n].N
            Ln = len(mappings[n])

            #sizes = self.SAPS[n].sizes
            #a = np.concatenate([np.repeat(i**2,sizes[i]) for i in range(1,len(sizes))])
            #b = np.repeat(mappings[n], a)

            l = list(find(psis[n]))
            idx = np.argsort(l[1]); l[1] = l[1][idx]; l[0] = l[0][idx]; l[-1] = l[-1][idx] #NOTE: l[1] should be already sorted

            #NOTE: the following line maps the local Nn^2-by-Ln psi to a Nn^2-by-L psi by adding zero columns and reordering
            l[1] = np.repeat(mappings[n], np.unique(l[1], return_counts=True)[1])

            ##NOTE: the following 3 lines are just a check. Can remove if it works.
            #psi_mat = csr_matrix((l[-1], (l[0], l[1]) ), shape=(Nn**2,L)) 
            #mat = csr_matrix((np.ones((Ln,), dtype=bool), (np.arange(Ln), mappings[n]) ), shape=(Ln,L)) 
            #assert(abs(psi_mat-psis[n]@mat).max() < 1.0e-14)

            # can do the constraints in (Nn+1)-by-(Nn+1) shape
            if budget is not None:
                l[0] += l[0]//Nn; l[1] += 1; l[-1] = -np.concatenate([scales[n]*l[-1], [1]]); l[0] = np.concatenate([l[0],[(Nn+1)**2-1]]); l[1] = np.concatenate([l[1],[0]])
                G1 = csr_matrix((l[-1],(l[0],l[1])), shape=((Nn+1)**2,L+1)); G1 = csr_to_cvxopt(G1)
                h1 = np.zeros((Nn+1,Nn+1)); h1[-1,0] = np.sqrt(scales[n]); h1[0,-1] = np.sqrt(scales[n]); h1 = matrix(h1)

            else:
                l[0] += l[0]//Nn; l[-1] *= (-scales[n])
                G1 = csr_matrix((l[-1], (l[0],l[1])), shape=((Nn+1)**2,L)); G1 = csr_to_cvxopt(G1)
                h1 = np.zeros((Nn+1,Nn+1)); h1[-1,0] = np.sqrt(scales[n])/eps[n]; h1[0,-1] = np.sqrt(scales[n])/eps[n]; h1[-1,-1] = 1; h1 = matrix(h1)

            Gs.append(G1)
            hs.append(h1)

        return Gs,hs

    def cvxopt_solve(self, budget=None, eps=None, delta=0.0, max_model_samples=None, cvxopt_params=None):
        budget, eps = self.check_input(budget, eps)

        No       = self.n_outputs
        N        = self.N
        L        = self.L
        w        = self.costs.copy()
        e        = self.e
        mappings = self.mappings

        cvxopt_solver_params = cvxopt_default_params.copy()
        if cvxopt_params is not None:
            cvxopt_solver_params.update(cvxopt_params)

        ES,rhs = self.get_max_sample_constraints(max_model_samples)

        es = []
        for n in range(No):
            ee = np.zeros((L,))
            ee[mappings[n]] = e.copy()[mappings[n]]
            es.append(ee)

        if budget is not None:
            wt  = np.concatenate([[0], w])
            ets = [np.concatenate([[0], -ee]) for ee in es]
            EST = []
            if len(ES) > 0:
                EST = [np.concatenate([[0], ees]) for ees in ES]

            c = np.zeros((L+1,)); c[0] = 1.; c = matrix(c)

            G0 = csr_matrix(np.vstack([-np.eye(L+1),wt] + ets + EST)); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L+1,)), [1.], -np.ones((No,))/budget, [item/budget for item in rhs]]); h0 = matrix(h0)

        else:
            # compute scaling heuristically based on MC samples
            n_MC_samples = max([CC[0,0]/ep**2 for CC,ep in zip(self.C,eps)])
            meps = max(eps); meps=100/np.sqrt(n_MC_samples)
            eps = eps/meps

            c = matrix(w/np.linalg.norm(w))
            ets = [-ee for ee in es]
            EST = [ees for ees in ES]

            G0 = csr_matrix(np.vstack([-np.eye(L)] + ets + EST)); G0.eliminate_zeros(); G0 = csr_to_cvxopt(G0)
            h0 = np.concatenate([np.zeros((L,)), (-meps**2)*np.ones((No,)), [meps**2*item for item in rhs]]); h0 = matrix(h0)

        Gs,hs = self.cvxopt_get_sdp_constraints(budget=budget, eps=eps)

        cvxopt_solver_params['show_progress'] = self.verbose
        try: res = solvers.sdp(c,Gl=G0,hl=h0,Gs=Gs,hs=hs,solver=None, options=cvxopt_solver_params)
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
            m *= meps**-2

        if self.verbose: print(m.round())

        return m

    def cvxpy_to_cvxopt(self,prob,cvxopt_params=cvxopt_default_params):
        probdata, _, _ = prob.get_problem_data(cp.CVXOPT)

        c = matrix(probdata['c'])
        G = csr_to_cvxopt(probdata['G'])
        h = matrix(probdata['h'])
        dims_tup = vars(probdata['dims'])
        dims = {'l' : dims_tup['nonneg'], 'q': [], 's': dims_tup['psd']}

        res = solvers.conelp(c, G, h, dims, options=cvxopt_params)

        return res

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

    def cvxpy_solve(self, budget=None, eps=None, delta=0.0, max_model_samples=None, cvxpy_params=None):
        budget, eps = self.check_input(budget, eps)

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        cvxpy_solver_params = cvxpy_default_params.copy()
        if cvxpy_params is not None:
            cvxpy_solver_params.update(cvxpy_params)

        ES,rhs = self.get_max_sample_constraints(max_model_samples)

        scales = np.array([1/abs(self.SAPS[n].psi).sum(axis=0).mean() for n in range(No)])

        m = cp.Variable(L, nonneg=True)
        if budget is not None:
            t = cp.Variable(nonneg=True)
            obj = cp.Minimize(t)
            constraints = [w@m <= 1, *self.cvxpy_get_multi_constraints(m, t=t, eps=None)]
            constraints += [m[mappings[n]]@e[mappings[n]] >= 1./budget for n in range(self.n_outputs)]
            constraints += [m@ees <= rr/budget for ees,rr in zip(ES,rhs)]
        else:
            # compute scaling heuristically based on MC samples
            n_MC_samples = max([CC[0,0]/ep**2 for CC,ep in zip(self.C,eps)])
            meps = max(eps); meps=100/np.sqrt(n_MC_samples)
            eps = eps/meps
            obj = cp.Minimize((w/np.linalg.norm(w))@m)
            constraints = [*self.cvxpy_get_multi_constraints(m, t=None, eps=eps)]
            constraints += [m[mappings[n]]@e[mappings[n]] >= 1*meps**2 for n in range(self.n_outputs)]
            constraints += [m@ees <= rr*meps**2 for ees,rr in zip(ES,rhs)]

        prob = cp.Problem(obj, constraints)

        #res = self.cvxpy_to_cvxopt(prob)

        #prob.solve(verbose=self.verbose, solver="SCS")#, acceleration_lookback=0, acceleration_interval=0)
        #prob.solve(verbose=self.verbose, solver="MOSEK", mosek_params=mosek_params)
        try: prob.solve(verbose=self.verbose, solver="CVXOPT", **cvxpy_solver_params)
        except (cp.SolverError,ZeroDivisionError):
            return None

        if m.value is None:
            return None

        if self.verbose: print(m.value)
        if budget is not None: m.value *= budget
        elif eps  is not None: m.value /= meps**2

        if self.verbose: print(m.value.round())
        return m.value

    def scipy_solve(self, budget=None, eps=None, x0=None, max_model_samples=None):
        from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

        budget, eps = self.check_input(budget, eps)

        delta = 1.0e-6

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        ES,rhs = self.get_max_sample_constraints(max_model_samples)

        def max_variance(m,delta=0):
            return max(self.variances(m,delta=delta))

        if self.verbose: print("Optimizing using scipy...")

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
            constraint5 = [LinearConstraint(np.concatenate([[0],ees]),-np.inf,rr) for ees,rr in zip(ES,rhs)]
            constraint2 = LinearConstraint(np.concatenate([[0],w]), -np.inf, budget)
            constraint4 = [NonlinearConstraint(lambda x,nn=n : x[0] - self.SAPS[nn].variance(x[1:][mappings[nn]],delta=delta), 0, np.inf, jac = lambda x,nn=n : np.concatenate([[1],-self.SAPS[nn].variance_GH(x[1:][mappings[nn]],nohess=True,delta=delta)[1]]), hess = lambda x,p,nn=n : np.block([[0, np.zeros((1,len(x)-1))],[np.zeros((len(x)-1,1)), -self.SAPS[nn].variance_GH(x[1:][mappings[nn]],delta=delta)[2]]])*p) for n in range(No)]

            if x0 is None: x0 = np.ceil(budget*abs(np.random.randn(L))); x0 - (x0@w-budget)*w/(w@w); x0 = np.concatenate([[max_variance(x0,delta=delta)], x0])
            res = minimize(lambda x : (x[0], eee), x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=[constraint2]+constraint3+constraint4+constraint5, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":1000, 'verbose':3*int(self.verbose)}, tol = 1.0e-15)

        else:
            constraint1 = Bounds(0.0*np.ones((L,)), np.inf*np.ones((L,)), keep_feasible=True)
            constraint3 = [LinearConstraint(ee.copy(), 1, np.inf, keep_feasible=True) for ee in es]
            constraint5 = [LinearConstraint(ees,-np.inf,rr) for ees,rr in zip(ES,rhs)]
            epsq = eps**2
            constraint2 = [NonlinearConstraint(lambda x,n=nn : self.SAPS[n].variance(x[mappings[n]],delta=delta), -np.inf, epsq[n], jac = lambda x,n=nn : self.SAPS[n].variance_GH(x[mappings[n]],nohess=True,delta=delta)[1], hess=lambda x,p,n=nn : self.SAPS[n].variance_GH(x[mappings[n]],delta=delta)[2]*p) for nn in range(No)]
            if x0 is None: x0 = np.ceil(np.linalg.norm(eps)**-2*np.random.rand(L))
            res = minimize(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hessp=lambda x,p : np.zeros((len(x),)), bounds=constraint1, constraints=constraint2 + constraint3 + constraint5, method="trust-constr", options={"factorization_method" : [None,"SVDFactorization"][0], "disp" : True, "maxiter":2500, 'verbose':3*int(self.verbose)}, tol = 1.0e-15)

        if budget is not None: res.x = res.x[1:]

        if self.verbose: print(res.x.round())

        return res.x

    def ipopt_solve(self, budget=None, eps=None, x0=None, max_model_samples=None):
        from cyipopt import minimize_ipopt

        budget, eps = self.check_input(budget, eps)

        ES,rhs = self.get_max_sample_constraints(max_model_samples)

        delta = 1.0e5

        L        = self.L
        No       = self.n_outputs
        w        = self.costs
        e        = self.e
        mappings = self.mappings

        def max_variance(m,delta=0):
            return max(self.variances(m,delta=delta))

        if self.verbose: print("Optimizing using ipopt...")

        options = {"maxiter":200, 'print_level':5*int(self.verbose), 'print_user_options' : ['no', 'yes'][int(self.verbose)], 'bound_relax_factor' : 1.e-30, 'honor_original_bounds' : 'yes', 'dual_inf_tol' : 1.0e-1}

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
            constraint2 = [{'type':'ineq', 'fun': lambda x : budget - w@x[1:], 'jac': lambda x : np.concatenate([np.zeros((1,)),-w]), 'hess': lambda x,p : csr_matrix((len(x),len(x)))*float(p)}]
            constraint4 = [{'type':'ineq', 'fun': lambda x,nn=n : x[0] - self.SAPS[nn].variance(x[1:][mappings[nn]],delta=delta), 'jac' : lambda x,nn=n : np.concatenate([[1],-self.SAPS[nn].variance_GH(x[1:][mappings[nn]],nohess=True,delta=delta)[1]]), 'hess': lambda x,p,nn=n : np.block([[0, np.zeros((1,len(x)-1))],[np.zeros((len(x)-1,1)), -self.SAPS[nn].variance_GH(x[1:][mappings[nn]],delta=delta)[2]]])*p} for n in range(No)]
            constraint5 = [{'type':'ineq', 'fun': lambda x : rr-ees@x[1:], 'jac': lambda x : np.concatenate([np.zeros((1,)),-ees]), 'hess': lambda x,p : 0} for ees,rr in zip(ES,rhs)]

            if x0 is None: x0 = np.ceil(budget*abs(np.ones((L,)))); x0 - (x0@w-budget)*w/(w@w); x0 = np.concatenate([[max_variance(x0,delta=delta)], x0])
            res = minimize_ipopt(lambda x : (x[0], eee), x0, jac=True, hess=lambda x : csr_matrix((len(x),len(x))), bounds=constraint1, constraints=constraint2+constraint3+constraint4+constraint5, options=options, tol = 1.0e-14)

        else:
            # compute scaling heuristically based on MC samples
            n_MC_samples = max([CC[0,0]/ep**2 for CC,ep in zip(self.C,eps)])
            meps = max(eps); meps=100/np.sqrt(n_MC_samples);
            eps = eps/meps
            epsq = eps**2
            constraint1 = [(0, np.inf) for i in range(L)]
            constraint3 = [{'type':'ineq', 'fun': lambda x,ee=ees : ee@x-1*meps**2, 'jac': lambda x,ee=ees : ee, 'hess': lambda x,p : csr_matrix((len(x),len(x)))*float(p)} for ees in es]
            constraint5 = [{'type':'ineq', 'fun': lambda x : (meps**2)*rr-ees@x, 'jac': lambda x : -ees, 'hess': lambda x,p : 0} for ees,rr in zip(ES,rhs)]
            constraint2 = [{'type':'ineq', 'fun': lambda x,n=nn : epsq[n] - self.SAPS[n].variance(x[mappings[n]],delta=delta), 'jac': lambda x,n=nn : -self.SAPS[n].variance_GH(x[mappings[n]],nohess=True,delta=delta)[1], 'hess': lambda x,p,n=nn : -self.SAPS[n].variance_GH(x[mappings[n]],delta=delta)[2]*p} for nn in range(No)]
            if x0 is None: x0 = np.ceil(np.linalg.norm(eps)**-2*np.ones((L,)))
            res = minimize_ipopt(lambda x : [(w/np.linalg.norm(w))@x,w/np.linalg.norm(w)], x0, jac=True, hess=lambda x : csr_matrix((len(x),len(x))), bounds=constraint1, constraints=constraint2 + constraint3 + constraint5, options=options, tol = 1.0e-12)

        if budget is not None: res.x = res.x[1:]
        elif eps is not None: res.x *= meps**-2

        if self.verbose: print(res.x.round())

        return res.x
