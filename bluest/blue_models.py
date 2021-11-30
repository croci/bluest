import numpy as np
import networkx as nx
from .blue_fn import blue_fn
from .blue_opt import BLUESampleAllocationProblem
from .spg import spg

spg_default_params = {"maxit" : 200,
                      "maxfc" : 200**2,
                      "verbose" : False,
                      "eps"     : 1.0e-4,
                      "lmbda_min" : 10.**-30,
                      "lmbda_max" : 10.**30,
                      "linesearch_history_length" : 10,
                     }

default_params = {
                    "remove_uncorrelated" : True,
                    "optimization_solver" : "gurobi",
                    "covariance_estimation_samples" : 100,
                    "sample_batch_size": 1,
                    "spg_params" : spg_default_params,
                    }

# Are any checks on the correlation matrix necessary?
class BLUEProblem(object):
    def __init__(self, M, C=None, costs=None, **params):
        '''
            INPUT:

            - M is the number of models,

            - C is the M-by-M covariance matrix (if not
            provided, it will be estimated). NaN in C correspond to unknown
            values and  infinite values correspond to models that won't be coupled.

            - costs is a length-M array such that costs[i] = cost of computing 1 sample from model i.

            - params is a dictionary containing optional parameters (see default in self.default_params)
        '''
        if C is None: C = np.nan*np.ones((M,M))
        self.M = M

        self.sample_allocation_problem = None

        self.default_params = default_params
        self.params = default_params.copy()
        spg_params = spg_default_params.copy()
        spg_params.update(params.get("spg_params", {}))
        params["spg_params"] = spg_params
        self.params.update(params)

        self.G = self.get_model_graph(C, costs=costs)

        self.estimate_missing_covariances(self.params["covariance_estimation_samples"])
        self.project_covariance()
        if costs is None: self.estimate_costs()

        self.check_graph(remove_uncorrelated=self.params["remove_uncorrelated"])

        print("BLUE estimator ready.")

    #NOTE: N here is the number of samples to be taken per time
    def evaluate(self, ls, samples, N=1):
        ''' must be implemented by the user'''
        raise NotImplementedError
    
    #NOTE: N here is the number of samples to be taken per time
    def sampler(self, ls, N=1):
        ''' must be implemented by the user'''
        raise NotImplementedError

    def get_costs(self):
        return np.array([self.G.nodes[l]['cost'] for l in range(self.M)])

    def get_group_costs(self, groups):
        model_costs = self.get_costs()
        costs = np.array([sum(model_costs[group]) for groupsk in groups for group in groupsk])
        return costs

    def setup_solver(self, K=3, budget=None, eps=None, groups=None, solver=None, integer=False):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None
        if solver is None:
            solver=self.params["optimization_solver"]

        if groups is None:
            K = min(K, self.M)
            flattened_groups = []
            groups = [[] for k in range(K)]
            for clique in nx.enumerate_all_cliques(self.G):
                kn = len(clique)
                if kn > K: break
                groups[kn-1].append(clique)
                flattened_groups.append(clique)

            groups = [item for item in groups if len(item) > 0]
            K = min(K, len(groups))

        else:
            K = len(groups[-1])
            new_groups = [[] for k in range(K)]
            flattened_groups = []
            for group in groups:
                if is_subclique(self.G, group):
                    new_groups[len(group)-1].append(group)
                    flattened_groups.append(group)

            groups = new_groups

        C = self.get_covariance() # this has some NaNs, but these should never come up
        costs = self.get_group_costs(groups)

        print("Computing optimal sample allocation...")
        self.sample_allocation_problem = BLUESampleAllocationProblem(C, K, groups, costs)
        self.sample_allocation_problem.solve(budget=budget, eps=eps, solver=solver, integer=integer)

        if budget is not None:
            var = self.sample_allocation_problem.variance(self.sample_allocation_problem.samples)
            N_MC = C[0,0]/var
            cost_MC = N_MC*costs[0] 
            print("\nBLUE cost: ", budget, "MC cost: ", cost_MC, "Savings: ", cost_MC/budget)
        else:
            N_MC = C[0,0]/eps**2
            cost_MC = N_MC*costs[0] 
            cost_BLUE = self.sample_allocation_problem.samples@costs
            print("\nBLUE cost: ", cost_BLUE, "MC cost: ", cost_MC, "Savings: ", cost_MC/cost_BLUE)

        which_groups = [self.sample_allocation_problem.flattened_groups[item] for item in np.argwhere(self.sample_allocation_problem.samples > 0).flatten()]
        print("\nModel groups selected: %s\n" % which_groups)

    def solve(self, K=3, budget=None, eps=None, groups=None, integer=False):
        if self.sample_allocation_problem is None:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups, integer=integer)

        elif budget is not None and budget != self.sample_allocation_problem.budget or eps is not None and eps != self.sample_allocation_problem.eps:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups, integer=integer)
        elif budget is None and eps is None and self.sample_allocation_problem.samples is None:
            raise ValueError("Need to prescribe either a budget or an error tolerance to run the BLUE estimator")

        print("Sampling BLUE...\n")

        flattened_groups = self.sample_allocation_problem.flattened_groups
        sample_list      = self.sample_allocation_problem.samples
        
        sums = []
        total_cost = 0
        for ls,N in zip(flattened_groups, sample_list):
            if N == 0:
                sums.append(np.zeros_like(ls))
                continue
            sumse,sumsc,cost = self.blue_fn(ls, N)
            sums.append(sumse)
            total_cost += cost

        mu,var = self.sample_allocation_problem.compute_BLUE_estimator(sums)

        return mu,var,total_cost

    def complexity_test(self, eps, K=3):
        print("Running cost complexity_test...")
        tot_cost = []
        for i in range(len(eps)):
            self.setup_solver(K=K, eps=eps[i], solver="gurobi", integer=True)
            tot_cost.append(sum(self.sample_allocation_problem.samples*self.sample_allocation_problem.costs))
        tot_cost = np.array(tot_cost)
        rate = np.polyfit(np.arange(len(tot_cost)), np.log2(tot_cost), 1)[0]
        print("Total costs   :", tot_cost)
        print("Estimated rate:", rate)
        return tot_cost, rate

    def get_model_graph(self, C, costs=None):
        '''
           Creates a graph of the models available from
           their (possibly partial) covariance structure C.
           Input:
               C - spd covariance matrix of the models,
                   NaN entries correspond to values to be
                   estimated and infinite entries correspond
                   to models that will not be coupled.

           Output:
               G - model graph. Its adjacency matrix is almost
               the model covariance matrix
        '''

        M = self.M

        maskinf = np.isinf(C)
        mask0   = C == 0
        C[mask0] = np.inf
        C[maskinf] = 0    # if set to zero, then the graph won't have an edge there

        G = nx.from_numpy_matrix(C)

        if costs is not None:
            for l in range(M):
                G.nodes[l]['cost'] = costs[l]

        return G

    def check_graph(self, remove_uncorrelated=False):

        if remove_uncorrelated:
            for i in range(self.M):
                for j in range(i,self.M):
                    if self.G.has_edge(i,j) and np.isinf(self.G[i][j]["weight"]):
                        self.G.remove_edge(i,j)

        if not nx.is_connected(self.G):
            print("WARNING! Model graph is not connected, pruning disconnected subgraph...")
            comp = nx.node_connected_component(self.G, 0)
            SG = self.G.subgraph(comp).copy()
            SG = nx.convert_node_labels_to_integers(SG, label_attribute="model_number")
            print("Done. New model graph size is %d. Model numbers saved as new graph attributes." % SG.number_of_nodes())
            self.G = SG
            self.M = SG.number_of_nodes()

    def get_covariance(self):
        '''
            Computes the model covariance matrix
            this is just the model graph adjacency
            matrix with 0 replaced by NaNs (models
            that cannot be coupled), and with infs
            replaced by 0 (uncorrelated models).
        '''
        C = nx.adjacency_matrix(self.G).toarray()
        mask0 = C == 0
        maskinf = np.isinf(C)
        C[mask0] = np.nan
        C[maskinf] = 0
        return C

    def get_correlation(self):
        C = self.get_covariance()
        s = np.sqrt(np.diag(C))
        return C/np.outer(s,s)

    def estimate_missing_covariances(self, N):
        print("Covariance estimation with %d samples..." % N)
        C = nx.adjacency_matrix(self.G).toarray()
        ls = list(np.where(np.isnan(np.sum(C,1)))[0])
        sumse,sumsc,cost = self.blue_fn(ls, N)
        C_hat = sumsc/N - np.outer(sumse/N,sumse/N)
        for i,j,c in self.G.edges(data=True):
            if np.isnan(c['weight']):
                if abs(C_hat[i,j]/np.sqrt(C_hat[i,i]*C_hat[j,j])) < 1.0e-7:
                    C_hat[i,j] = np.inf # mark as uncorrelated
                self.G[i][j]['weight'] = C_hat[i,j]

    def project_covariance(self):
        
        spg_params = self.params["spg_params"]

        # the covariance will have NaNs corresponding to entries that cannot be coupled
        C = self.get_covariance().flatten()
        mask = (~np.isnan(C)).astype(int)
        invmask = np.isnan(C).astype(int)

        # Penalty parameter for regularisation. Probably not needed.
        gamma = 0.0

        # projection onto SPD matrices
        def proj(X):
            l = int(np.sqrt(len(X)).round())
            X = X.reshape((l,l))
            l,V = np.linalg.eigh((X + X.T)/2)
            l[l<0] = 0
            return (V@np.diag(l)@V.T).flatten()

        def am(C,mask):
            X = C.copy()
            X[abs(mask) < 1.0e-14] = 0
            return X*mask

        def evalf(x):
            return 0.5*sum(am(x - C, mask**2)**2 + gamma*am(x**2, invmask**2))

        def evalg(x):
            return am(x - C, mask**2) + gamma*am(x, invmask**2)

        x = proj(am(C,abs(mask) > 1.0e-14))
        print("Running Spectral Gradient Descent for Covariance projection...")
        res = spg(evalf, evalg, proj, x, eps=spg_params["eps"], maxit=spg_params["maxit"], maxfc=spg_params["maxfc"], iprint=spg_params["verbose"], lmbda_min=spg_params["lmbda_min"], lmbda_max=spg_params["lmbda_max"], M=spg_params["linesearch_history_length"])

        if res["spginfo"] == 0:
            print("Covariance projected, projection error: ", res["f"])
        else:
            raise RuntimeError("Could not find good enough Covariance projection. Solver info:\n%s" % res)

        C_new = res["x"].reshape((self.M, self.M))
        s = np.sqrt(np.diag(C_new))
        rho_new = C_new/np.outer(s,s)
        C_new[abs(rho_new) < 1.0e-7] = np.inf # mark uncorrelated models
        C_new[np.isnan(C).reshape((self.M, self.M))] = np.nan # keep uncoupled models uncoupled

        for i in range(self.M):
            for j in range(self.M):
                coupled = not np.isnan(C_new[i,j])
                if self.G.has_edge(i,j):
                    if coupled: self.G[i][j]['weight'] = C_new[i,j]
                    else:       self.G[i][j]['weight'] = 0
                elif coupled:
                    self.G.add_edge(i,j)
                    self.G[i][j]['weight'] = C_new[i,j]

    def estimate_costs(self, N=2):
        print("Cost estimation via sampling...")
        for l in range(self.M):
            _,_,cost = self.blue_fn([l], N, verbose=False)
            self.G.nodes[l]['cost'] = cost/N

    def blue_fn(self, ls, N, verbose=True):
        return blue_fn(ls, N, self, self.sampler, N1=self.params["sample_batch_size"], verbose=verbose)

    def draw_model_graph(self):
        import matplotlib.pyplot as plt
        nx.draw(self.G)
        plt.show()

def is_subclique(G,nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == (n*(n-1))//2

if __name__ == '__main__':
    pass
