import numpy as np
import networkx as nx
from .blue_fn import blue_fn
from .blue_opt import BLUESampleAllocationProblem
from .spg import spg

default_params = {
                    "remove_uncorrelated" : True,
                    "optimization_solver" : "gurobi",
                    "covariance_estimation_samples" : 100,
                    }

# Are any checks on the correlation matrix necessary?
class BLUEProblem(object):
    def __init__(self, M, C=None, costs=None, params={}):
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
        self.params = default_params
        for key,value in params:
            self.params[key] = value

        self.G = self.get_model_graph(C, costs=costs)

        self.estimate_missing_covariances(self.params["covariance_estimation_samples"])
        self.project_covariance()
        if costs is None: self.estimate_costs()

        self.check_graph(remove_uncorrelated=self.params["remove_uncorrelated"])

    def evaluate(ls, samples):
        ''' must be implemented by the user'''
        raise NotImplementedError
    
    def sampler(N, ls):
        ''' must be implemented by the user'''
        raise NotImplementedError

    def get_costs(self):
        return np.array([self.G.nodes[l]['cost'] for l in range(self.M)])

    def get_group_costs(self, groups):
        model_costs = self.get_costs()
        costs = np.array([sum(model_costs[group]) for group in groupsk for groupsk in groups])
        return costs

    def setup_solver(self, K=3, budget=None, eps=None, groups=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None

        if groups is None:
            flattened_groups = []
            groups = [[] for k in range(K)]
            for clique in nx.enumerate_all_cliques(self.G):
                if len(clique) > K:
                    break
                groups[len(clique)-1].append(clique)
                flattened_groups.append(clique)
        else:
            K = len(groups[-1])
            new_groups = [[] for k in range(K)]
            for group in groups:
                new_groups[len(group)-1].append(group)

            flattened_groups = groups.copy()
            groups = new_groups

        C = self.get_covariance() # this has some NaNs, but these should never come up
        costs = self.get_group_costs(groups)

        print("Computing optimal sample allocation...")
        self.sample_allocation_problem = BLUESampleAllocationProblem(C, K, groups, costs)
        self.sample_allocation_problem.solve(budget=budget, eps=eps, solver=self.params["optimization_solver"])

        if budget is not None:
            var = self.sample_allocation_problem.variance(self.sample_allocation_problem.samples)
            N_MC = C[0,0]/var
            cost_MC = N_MC*costs[0] 
            print("BLUE cost: ", budget, "MC cost: ", cost_MC, "Savings: ", cost_MC/budget)
        else:
            N_MC = C[0,0]/eps**2
            cost_MC = N_MC*costs[0] 
            cost_BLUE = self.sample_allocation_problem.samples@costs
            print("BLUE cost: ", cost_BLUE, "MC cost: ", cost_MC, "Savings: ", cost_MC/cost_BLUE)

    def solve(self, K=3, budget=None, eps=None, groups=None):
        if self.sample_allocation_problem is None:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups)

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
            for i in range(M):
                for j in range(i,M):
                    if np.isinf(self.G[i][j]["weight"]):
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
        C = nx.adjacency_matrix(self.G)
        mask0 = C == 0
        maskinf = np.isinf(C)
        C[mask0] = np.nan
        C[maskinf] = 0
        return C

    def get_correlation_matrix(self):
        C = self.get_covariance()
        s = np.sqrt(np.diag(C))
        return C/np.outer(s,s)

    def estimate_missing_covariances(self, N):
        C = nx.adjacency_matrix(self.G)
        ls = list(np.where(np.isnan(np.sum(CC,1)))[0])
        sumse,sumsc,cost = self.blue_fn(ls, N)
        C_hat = sumsc/N - np.outer(sumse/N,sumse/N)
        for i,j,c in self.G.edges(data=True):
            if np.isnan(c['weight']):
                if abs(C_hat[i,j]/np.sqrt(C_hat[i,i]*C_hat[j,j])) < 1.0e-7:
                    C_hat[i,j] = np.inf # mark as uncorrelated
                self.G[i][j]['weight'] = C_hat[i,j]

    def project_covariance(self):
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

        def evalf(x):
            return 0.5*sum((mask**2*(x - C))**2 + gamma*invmask**2*x**2)

        def evalg(x):
            return (mask**2*(x - C)) + gamma*(invmask**2*x)

        x = proj(mask*C)
        print("Running Spectral Gradient Descent for Covariance projection...")
        res = spg(evalf, evalg, proj, x, iprint=False)

        if res["spginfo"] == 0:
            print("Covariance projected, projection error: ", res["f"])
        else:
            raise RuntimeError("Could not find good enough Covariance projection. Solver info:\n%s" % res)

        C_new = res["x"].reshape((self.M, self.M))
        s = np.sqrt(np.diag(C_new))
        rho_new = C_new/np.outer(s,s)
        C_new[abs(rho_new) < 1.0e-7] = np.inf # mark uncorrelated models
        C_new[np.isnan(C)] = np.nan           # keep uncoupled models uncoupled

        for i in range(self.M):
            for j in range(self.M):
                coupled = not np.isnan(C_new[i,j])
                if self.G.has_edge(i,j):
                    if coupled: self.G[i,j]['weight'] = C_new[i,j]
                    else:       self.G[i,j]['weight'] = 0
                elif coupled:
                    self.G.add_edge(i,j)
                    self.G[i,j]['weight'] = C_new[i,j]

    def estimate_costs(self, N=2):
        for l in range(self.M):
            _,_,cost = self.blue_fn([l], N)
            self.G.nodes[l]['cost'] = cost/N

    def blue_fn(self, ls, N):
        return blue_fn(ls, N, self, self.sampler)

    def draw_model_graph(self):
        import matplotlib.pyplot as plt
        nx.draw(self.G)
        plt.show()

if __name__ == '__main__':
    pass
