import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from blue_fn import blue_fn
from blue_opt import BLUESampleAllocationProblem

default_params = {
                    "remove_uncorrelated" = True
                    "optimization_solver" = "gurobi"
                    }

class BLUEProblem(object):
    def __init__(self, M, C=None, N=100, nodeargs={}, params={}):
        if C is None: C = np.nan*np.ones((M,M))
        self.M = M

        self.what_to_sample = (None,None)

        self.params = default_params
        for key,value in params:
            self.params[key] = value

        self.G = self.get_model_graph(C, **nodeargs)

        self.estimate_missing_covariances(N)
        if nodeargs.get('cost') is None:
            self.estimate_costs()

        self.check_graph(remove_uncorrelated=self.params["remove_uncorrelated"])

    def evaluate(ls, samples):
        raise NotImplementedError
    
    def sampler(N, ls):
        raise NotImplementedError

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

        sample_allocation_problem = BLUESampleAllocationProblem(C, K, groups, costs)
        sample_list = sample_allocation_problem.solve(budget=budget, eps=eps, solver=self.params["optimization_solver"])

        self.what_to_sample = (flattened_groups, sample_list)

    #FIXME
    def solve(self, K=3, budget=None, eps=None, groups=None):
        if self.what_to_sample[0] is None:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups)

        flattened_groups, sample_list = self.what_to_sample
        
        for ls,N in zip(flattened_groups, sample_list):
            if N == 0: continue
            sumse,sumsc,cost = self.blue_fn(ls, N)
            do_something_with_this_output() #FIXME

    def get_model_graph(self, C, **nodeargs):
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

        K = C.shape[0]

        maskinf = np.isinf(C)
        mask0   = C == 0
        C[mask0] = np.inf
        C[maskinf] = 0    # if set to zero, then the graph won't have an edge there

        G = nx.from_numpy_matrix(C)

        for key,values in nodeargs:
            for i in range(K):
                G.nodes[i][key] = values[i]

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

    def estimate_missing_covariances(self, N):
        #NOTE: here we can add Nick Higham's projection or matrix completion or both
        C = nx.adjacency_matrix(self.G)
        ls = list(np.where(np.isnan(np.sum(CC,1)))[0])
        sumse,sumsc,cost = self.blue_fn(ls, N)
        C_hat = sumsc/N - np.outer(sumse/N,sumse/N)
        for i,j,c in self.G.edges(data=True):
            if np.isnan(c['weight']):
                if abs(C_hat[i,j]/np.sqrt(C_hat[i,i]*C_hat[j,j])) < 1.0e-7:
                    C_hat[i,j] = np.inf # mark as uncorrelated
                self.G[i][j]['weight'] = C_hat[i,j]

    def estimate_costs(self, N=2):
        for l in range(self.M):
            _,_,cost = self.blue_fn([l], N)
            self.G.nodes[l]['cost'] = cost/N

    def blue_fn(self, ls, N):
        return blue_fn(ls, N, self, self.sampler)

if __name__ == '__main__':
    A = np.random.randn(5,5)
    A = A.T@A
    A[3,4],A[4,3] = 0,0
    A[2,3],A[3,2] = np.nan,np.nan
    costs = np.arange(5)

    G = get_model_graph(A, costs)

    nx.draw(G)
    plt.show()
