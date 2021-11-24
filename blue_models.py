import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class BLUEProblem(object):
    def __init__(self, M, C=None, N=100, **nodeargs):
        if C is None: C = np.nan*np.ones((M,M))
        self.M = M
        self.G = self.get_model_graph(C, **nodeargs)

        self.estimate_missing_covariances(N)
        if nodeargs.get('cost') is None:
            self.estimate_costs()

        self.check_graph()

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

    #FIXME
    def solve(self, K=3, remove_uncorrelated=False, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None
        elif eps is not None:
            raise NotImplementedError

        if remove_uncorrelated:
            for i in range(M):
                for j in range(i,M):
                    if np.isinf(self.G[i][j]["weight"]):
                        self.G.remove_edge(i,j)

        groups = []
        for clique in nx.enumerate_all_cliques(self.G):
            if len(clique) > K:
                break
            groups.append(clique)

        sample_list = self.find_optimal_sample_allocation(K, groups, budget)
        
        for ls,N in zip(groups, sample_list):
            sumse,sumsc,cost = self.blue_fn(ls, N)
            do_something_with_this_output() #FIXME

    #FIXME
    def find_optimal_sample_allocation(K, groups, budget):
        C = self.get_covariance()
        #FIXME: filter groups out by groupsize and set up the
        #       optimizaton problem



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
        C[maskinf] = 0

        G = nx.from_numpy_matrix(C)

        for key,values in nodeargs:
            for i in range(K):
                G.nodes[i][key] = values[i]

        return G

    def check_graph(self):
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
