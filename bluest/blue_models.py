import numpy as np
import networkx as nx
from itertools import combinations
from .blue_fn import blue_fn
from .mosap import MOSAP
from .misc import attempt_mlmc_setup,attempt_mfmc_setup
from .layered_network_graph import LayeredNetworkGraph
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
                    "optimization_solver" : "cvxpy",
                    "covariance_estimation_samples" : 100,
                    "sample_batch_size": 1,
                    "spg_params" : spg_default_params,
                    }

def is_subclique(G,nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == (n*(n-1))//2

# Are any checks on the correlation matrix necessary?
class BLUEProblem(object):
    def __init__(self, M, C=None, costs=None, n_outputs=1, **params):
        '''
            INPUT:

            - M is the number of models,

            - C is the M-by-M covariance matrix (if not
            provided, it will be estimated). NaN in C correspond to unknown
            values and  infinite values correspond to models that won't be coupled.

            - costs is a length-M array such that costs[i] = cost of computing 1 sample from model i.

            - params is a dictionary containing optional parameters (see default in self.default_params)
        '''
        self.M = M
        self.n_outputs = n_outputs

        if C is None: C = [np.nan*np.ones((M,M)) for n in range(n_outputs)]
        if not isinstance(C,(list,tuple)): C = [C]

        self.MOSAP = None

        self.default_params = default_params
        self.params = default_params.copy()
        spg_params = spg_default_params.copy()
        spg_params.update(params.get("spg_params", {}))
        params["spg_params"] = spg_params
        self.params.update(params)

        self.G = [self.get_model_graph(C[n], costs=costs) for n in range(n_outputs)]
        self.SG = [list(range(M)) for n in range(n_outputs)]

        if costs is None: self.estimate_costs()
        self.check_costs(warning=True) # Sending a warning just in case
        
        self.estimate_missing_covariances(self.params["covariance_estimation_samples"])
        self.project_covariances()

        self.check_graphs(remove_uncorrelated=self.params["remove_uncorrelated"])

        print("BLUE estimator ready.\n")

    #################### FUNCTIONS TO BE OVERLOADED #######################

    #NOTE: N here is the number of samples to be taken per time
    def evaluate(self, ls, samples, N=1):
        ''' must be implemented by the user'''
        raise NotImplementedError
    
    #NOTE: N here is the number of samples to be taken per time
    def sampler(self, ls, N=1):
        ''' must be implemented by the user'''
        raise NotImplementedError

    def get_models_inner_products(self):
        ''' must be overloaded by the user if non-scalar outputs are present'''
        return [lambda a,b : a*b for n in range(self.n_outputs)]

    #################### UTILITY FUNCTIONS #######################

    def get_costs(self):
        return np.array([self.G[0].nodes[l]['cost'] for l in range(self.M)])

    def get_group_costs(self, groups):
        model_costs = self.get_costs()
        costs = np.array([sum(model_costs[group]) for groupsk in groups for group in groupsk])
        return costs

    def check_costs(self, warning=False):
        costs = self.get_costs()
        if costs[0] != costs.max():
            more_expensive_models = [self.G[0].nodes[i]["model_number"] for i in costs[costs > costs[0]]]
            message_error = "Model zero is not the most expensive model. Consider removing the more expensive models %s" % more_expensive_models
            message_warning = "WARNING! Model zero is not the most expensive model and some estimators won't run in this setting. The more expensive models are: %s" % more_expensive_models
            if warning: print(message_warning)
            else: raise ValueError(message_error)

    def get_covariances(self):
        return [self.get_covariance(n) for n in range(self.n_outputs)]

    def get_correlations(self):
        return [self.get_correlation(n) for n in range(self.n_outputs)]

    def get_covariance(self, n=0):
        '''
            Computes the model covariance matrix
            this is just the model graph adjacency
            matrix with 0 replaced by NaNs (models
            that cannot be coupled), and with infs
            replaced by 0 (uncorrelated models).
        '''
        C = nx.adjacency_matrix(self.G[n]).toarray()
        mask0 = C == 0
        maskinf = np.isinf(C)
        C[mask0] = np.nan
        C[maskinf] = 0
        return C

    def get_correlation(self,n=0):
        C = self.get_covariance(n)
        s = np.sqrt(np.diag(C))
        return C/np.outer(s,s)

    def outer(self, a, b, inner):
        assert len(a) == len(b)
        L = len(a)

        out = np.zeros((L,L))

        for i in range(L):
            for j in range(L):
                out[i,j] = inner(a[i],b[j])

        return out

    #################### GRAPH MANIPULATION #######################

    def reorder_all_graph_nodes(self, ordering=None):
        for n in range(self.n_outputs):
            self.reorder_graph_nodes(n, ordering=ordering)

    def reorder_graph_nodes(self, n=0, ordering=None):
        M = self.M
        G = self.G[n]
        H = nx.Graph()
        sorted_nodes = sorted(G.nodes(data=True))
        if ordering is None or (isinstance(ordering, str) and "asc" in ordering):
            mapping = {i:i for i in range(M)}
        elif isinstance(ordering, str) and "desc" in ordering:
            mapping = {i:len(sorted_nodes)-i-1 for i in range(M)}
            sorted_nodes = reversed(sorted_nodes)
        elif isinstance(ordering, (list, np.ndarray)) and M == len(ordering):
            mapping = {i:item for i,item in enumerate(ordering)}
            sorted_nodes = [sorted_nodes[i] for i in ordering]
        else:
            raise ValueError("ordering must be None, 'asc', 'desc', 'ascending', 'descending' or an array/list with length equal to the number of nodes.")

        H.add_nodes_from(sorted_nodes)
        H.add_edges_from(G.edges(data=True))
        H = nx.relabel_nodes(H, mapping)
        self.G[n] = H

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

        for l in range(M):
            G.nodes[l]['model_number'] = l

        return G

    def check_graphs(self, remove_uncorrelated=False):
        for n in range(self.n_outputs):
            self.check_graph(n, remove_uncorrelated=remove_uncorrelated)

    def check_graph(self, n=0, remove_uncorrelated=False):

        if remove_uncorrelated:
            for i in range(self.M):
                for j in range(i,self.M):
                    if self.G[n].has_edge(i,j) and np.isinf(self.G[n][i][j]["weight"]):
                        self.G[n].remove_edge(i,j)

        if not nx.is_connected(self.G[n]):
            comp = nx.node_connected_component(self.G[n], 0)
            self.SG[n] = comp
            print("WARNING! Model graph %d is not connected. Connected graph size: %d" % (n,len(comp)))
            #print("WARNING! Model graph is not connected, pruning disconnected subgraph...")
            #SG = self.G.subgraph(comp).copy()
            #SG = nx.convert_node_labels_to_integers(SG)
            #print("Done. New model graph size is %d." % SG.number_of_nodes())
            #self.G = SG
            #self.M = SG.number_of_nodes()

    def draw_model_graph(self, n=0):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        G = self.G[n]
        rho = self.get_correlation(n)
        edge_color_pair = [to_rgba("seagreen"), to_rgba("tab:blue")]
        pos = nx.shell_layout(G)
        edges = G.edges()
        nodes = list(G)
        weights = np.array([abs(rho[u][v]) for u,v in edges]); weights -= min(weights); weights /= max(weights); weights = 4*weights + 1
        edge_colors = np.array([edge_color_pair[int(rho[u][v] > 0)] for u,v in edges])
        node_colors = np.array([G.nodes[l]['cost'] for l in nodes]); node_colors = plt.cm.jet(plt.Normalize()(np.log(node_colors)))
        nx.draw(G, pos=pos, nodelist=nodes, edgelist=edges, width=weights, edge_color=edge_colors, node_color=node_colors)
        plt.show()

    def draw_multilayer_model_graph(self):
        #NOTE: can probably modify this so that it looks more like the single graph plot
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node_labels = {n : str(n) for n in range(self.M)}
        LayeredNetworkGraph(self.G, node_labels=node_labels, ax=ax, layout=nx.shell_layout)
        ax.set_axis_off()
        plt.show()

    #################### COVARIANCE AND COST ESTIMATORS #######################

    def estimate_missing_covariances(self, N):
        print("Covariance estimation with %d samples..." % N)
        C = [nx.adjacency_matrix(self.G[n]).toarray() for n in range(self.n_outputs)]
        ls = list(np.where(np.isnan(np.sum(sum(C),1)))[0])
        sumse,sumsc,cost = self.blue_fn(ls, N)
        inners = self.get_models_inner_products()
        C_hat = [sumsc[n]/N - self.outer(sumse[n],sumse[n],inners[n])/N**2 for n in range(self.n_outputs)]
        for n in range(self.n_outputs):
            for i,j,c in self.G[n].edges(data=True):
                if np.isnan(c['weight']):
                    if abs(C_hat[n][i,j]/np.sqrt(C_hat[n][i,i]*C_hat[n][j,j])) < 1.0e-7:
                        C_hat[n][i,j] = np.inf # mark as uncorrelated
                    self.G[n][i][j]['weight'] = C_hat[n][i,j]

    def project_covariances(self):
        for n in range(self.n_outputs):
            self.project_covariance(n)

    def project_covariance(self, n=0):
        
        spg_params = self.params["spg_params"]

        # the covariance will have NaNs corresponding to entries that cannot be coupled
        C = self.get_covariance(n).flatten()
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
                if self.G[n].has_edge(i,j):
                    if coupled: self.G[n][i][j]['weight'] = C_new[i,j]
                    else:       self.G[n][i][j]['weight'] = 0
                elif coupled:
                    self.G[n].add_edge(i,j)
                    self.G[n][i][j]['weight'] = C_new[i,j]

    def estimate_costs(self, N=2):
        print("Cost estimation via sampling...")
        for l in range(self.M):
            _,_,cost = self.blue_fn([l], N, verbose=False)
            for n in range(self.n_outputs):
                self.G[n].nodes[l]['cost'] = cost/N

    #################### SOLVERS #######################

    def blue_fn(self, ls, N, verbose=True):
        return blue_fn(ls, N, self, sampler=self.sampler, inners=self.get_models_inner_products(), N1=self.params["sample_batch_size"], No=self.n_outputs, verbose=verbose)

    def setup_solver(self, K=3, budget=None, eps=None, groups=None, multi_groups=None, solver=None, integer=False):
        if budget is None and eps is None: raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None: eps = None
        if eps is not None and isinstance(eps,(int,float,np.int,np.float)): eps = [eps for n in range(self.n_outputs)]
        if solver is None: solver=self.params["optimization_solver"]
        if multi_groups is not None and len(multi_groups) != self.M:
            raise ValueError("multi_groups must be a list of groupings of the same length as the number of models.")
        if groups is not None and multi_groups is None:
            multi_groups = [groups for n in range(self.n_outputs)]

        if multi_groups is None:
            Ks = []
            multi_groups = []
            K = min(K, self.M)
            for n in range(self.n_outputs):
                KK = K
                groups = [[] for k in range(K)]
                for clique in nx.enumerate_all_cliques(self.G[n]):
                    kn = len(clique)
                    if kn > K: break
                    if all(node in self.SG[n] for node in clique): # filter out the cliques in a non-connected component. See self.check_graph
                        groups[kn-1].append(clique)

                groups = [item for item in groups if len(item) > 0]
                KK = min(KK, len(groups))
                multi_groups.append(groups)
                Ks.append(KK)

            K = max(Ks)

        else:
            Ks = [min(max(len(item) for item in groups), self.M) for groups in multi_groups]
            for n in range(self.n_outputs):
                groups = multi_groups[n]
                new_groups = [[] for k in range(Ks[i])]
                for group in groups:
                    if is_subclique(self.G[n], group) and all(node in self.SG[n] for node in group):
                        new_groups[len(group)-1].append(group)

                multi_groups[n] = new_groups

            Ks = [min(max(len(item) for item in groups), self.M) for groups in multi_groups]
            K  = max(Ks)

        groups = [[] for k in range(K)]
        for n in range(self.n_outputs):
            for k in range(Ks[n]):
                for group in multi_groups[n][k]:
                    if group not in groups[k]:
                        groups[k].append(group)

        for k in range(K):
            groups[k].sort()

        C = self.get_covariances() # this has some NaNs, but these should never come up
        costs = self.get_group_costs(groups)
        multi_costs = [self.get_group_costs(item) for item in multi_groups]

        print("Computing optimal sample allocation...")
        self.MOSAP = MOSAP(C, K, Ks, groups, multi_groups, costs, multi_costs)
        self.MOSAP.solve(budget=budget, eps=eps, solver=solver, integer=integer)

        Vs = self.MOSAP.variances(self.MOSAP.samples)
        cost_BLUE = self.MOSAP.samples@costs
        N_MC = max(C[n][0,0]/Vs[n] for n in range(self.n_outputs))
        cost_MC = N_MC*costs[0] 
        print("\nBLUE cost: ", cost_BLUE, "MC cost: ", cost_MC, "Savings: ", cost_MC/cost_BLUE)

        # FIXME: flattened groups will be the union between all the possible groups selected above
        which_groups = [self.MOSAP.flattened_groups[item] for item in np.argwhere(self.MOSAP.samples > 0).flatten()]
        print("\nModel groups selected: %s\n" % which_groups)
        print("BLUE estimator setup. Max error: ", np.sqrt(max(Vs)), " Cost: ", cost_BLUE, "\n")

        blue_data = {"samples" : self.MOSAP.samples, "errors" : np.sqrt(Vs), "total_cost" : cost_BLUE}

        return which_groups, blue_data

    def solve(self, K=3, budget=None, eps=None, groups=None, multi_groups=None, integer=False, solver=None):
        if solver is None: solver = self.params["optimization_solver"]
        if self.MOSAP is None:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups, multi_groups=multi_groups, integer=integer, solver=solver)

        elif budget is not None and budget != self.MOSAP.budget or eps is not None and eps != self.MOSAP.eps:
            self.setup_solver(K=K, budget=budget, eps=eps, groups=groups, multi_groups=multi_groups, integer=integer, solver=solver)
        elif budget is None and eps is None and self.MOSAP.samples is None:
            raise ValueError("Need to prescribe either a budget or an error tolerance to run the BLUE estimator")

        print("Sampling BLUE...\n")

        flattened_groups = self.MOSAP.flattened_groups
        sample_list      = self.MOSAP.samples
        
        sums = [[] for n in range(self.n_outputs)]
        for ls,N in zip(flattened_groups, sample_list):
            if N == 0:
                for n in range(self.n_outputs):
                    sums[n].append([0 for l in range(len(ls))])
                continue
            sumse,_,_ = self.blue_fn(ls, N)
            for n in range(self.n_outputs):
                sums[n].append(sumse[n])

        mus,Vs = self.MOSAP.compute_BLUE_estimators(sums, sample_list)
        errs = np.sqrt(Vs)
        tot_cost = self.MOSAP.tot_cost

        return mus,errs,tot_cost

    def setup_mlmc(self, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None
        if eps is not None and isinstance(eps,(int,float,np.int,np.float)): eps = [eps for n in range(self.n_outputs)]
        if eps is None: eps = [None for n in range(self.n_outputs)]

        M = self.M

        self.check_costs()

        w = self.get_costs()
        idx = np.argsort(w)[::-1]
        assert idx[0] == 0

        print("Setting up optimal MLMC estimator...\n") 

        # get all groups that incude model 0 and are feasible for MLMC with models ordered by cost
        GG = nx.intersection_all(self.G)
        groups = [[0]]
        for i in range(M-1):
            for remove in combinations(range(1,M),i):
                keep = np.array([i for i in range(M) if i not in remove])
                group = list(idx[keep])
                if all([GG.has_edge(i,j) for i,j in zip(group[:-1],group[1:])]):
                    groups.append(group)

        best_group = None
        min_err  = np.inf
        min_cost = np.inf
        best_data = [{} for n in range(self.n_outputs)]
        CC = self.get_covariances()
        for group in groups:
            assert group[0] == 0
            mlmc_data_list = [{} for n in range(self.n_outputs)]
            # getting actual MLMC costs and variances
            for n in range(self.n_outputs):
                C = CC[n]
                subC = C[np.ix_(group,group)]
                subw = w[group].copy()
                if len(group) > 1:
                    v,corrs = np.diag(subC).copy(), np.diag(subC,1)
                    v[:-1]    += v[1:] - 2*corrs
                    subw[:-1] += subw[1:]
                else: v = subC[0]
                feasible, mlmc_data_list[n] = attempt_mlmc_setup(v, subw, budget=budget, eps=eps[n])
                if not feasible: break

            if not feasible: continue

            if budget is not None:
                err = max(mlmc_data["error"] for mlmc_data in mlmc_data_list)
                if err < min_err:
                    min_err = err
                    best_group = group
                    for n in range(self.n_outputs):
                        best_data[n].update(mlmc_data_list[n])
            else:
                cost = np.max(np.vstack([mlmc_data["samples"] for mlmc_data in mlmc_data_list]), axis=0)@w[group]
                if cost < min_cost:
                    min_cost = cost
                    best_group = group
                    for n in range(self.n_outputs):
                        best_data[n].update(mlmc_data_list[n])

        samples = np.max(np.vstack([mlmc_data["samples"] for mlmc_data in best_data]), axis=0)
        cost = samples@w[best_group]
        if budget is not None: # adjust if budget bound. The max above already takes care of the variance bound.
            samples = np.floor(samples - (max(cost-budget,0)/(w[best_group]@w[best_group]))*w[best_group]).astype(int)
            samples[0] = max(samples[0], 1) # need at least one sample on level 0
            cost = samples@w[best_group]

        errs = [np.sqrt(mlmc_data["variance"](samples)) for mlmc_data in best_data]

        mlmc_data = {"samples" : samples, "errors" : errs, "total_cost" : cost}
        print("Best MLMC estimator found. Coupled models:", best_group, " Max error: ", max(errs), " Cost: ", cost, "\n")
        return best_group, mlmc_data

    def solve_mlmc(self, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None

        best_group, mlmc_data = self.setup_mlmc(budget=budget, eps=eps)

        samples  = mlmc_data["samples"]
        errs      = mlmc_data["errors"]
        tot_cost = mlmc_data["total_cost"]

        print("\nSampling optimal MLMC estimator...\n")

        L = len(best_group)
        groups = [item for item in zip(best_group[:-1],best_group[1:])] + [[best_group[-1]]]
        mu = [0 for n in range(self.n_outputs)]
        for i in range(L):
            N = samples[i]
            sumse,_,_ = self.blue_fn(groups[i], N)
            for n in range(self.n_outputs):
                if i < L-1: mu[n] += (sumse[n][0]-sumse[n][1])/N
                else:       mu[n] += sumse[n][0]/N

        return mu, errs, tot_cost

    def setup_mfmc(self, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None
        if eps is not None and isinstance(eps,(int,float,np.int,np.float)): eps = [eps for n in range(self.n_outputs)]
        if eps is None: eps = [None for n in range(self.n_outputs)]

        sigmas = [np.sqrt(np.diag(self.get_covariance(n))) for n in range(self.n_outputs)]
        rhos = [self.get_correlation(n)[0,:] for n in range(self.n_outputs)]
        w = self.get_costs()

        print("Setting up optimal MFMC estimator...\n") 

        GG = nx.intersection_all(self.G)

        best_group = None
        min_err  = np.inf
        min_cost = np.inf
        best_data = [{} for n in range(self.n_outputs)]
        clique_list = [clique for clique in nx.enumerate_all_cliques(GG) if 0 in clique]
        for clique in clique_list:
            assert clique[0] == 0
            mfmc_data_list = [{} for n in range(self.n_outputs)]
            for n in range(self.n_outputs):
                feasible,mfmc_data_list[n] = attempt_mfmc_setup(sigmas[n][clique], rhos[n][clique], w[clique], budget=budget, eps=eps[n])
                if not feasible: break

            if not feasible: continue

            if budget is not None:
                err = max(mfmc_data["error"] for mfmc_data in mfmc_data_list)
                if err < min_err:
                    min_err = err
                    best_group = clique
                    for n in range(self.n_outputs):
                        best_data[n].update(mfmc_data_list[n])
            else:
                cost = np.max(np.vstack([mfmc_data["samples"] for mfmc_data in mfmc_data_list]), axis=0)@w[clique]
                if cost < min_cost:
                    min_cost = cost
                    best_group = clique
                    for n in range(self.n_outputs):
                        best_data[n].update(mfmc_data_list[n])

        samples = np.max(np.vstack([mfmc_data["samples"] for mfmc_data in best_data]), axis=0)
        cost = samples@w[best_group]
        if budget is not None: # adjust if budget bound. The max above already takes care of the variance bound.
            samples = np.floor(samples - (max(cost-budget,0)/(w[best_group]@w[best_group]))*w[best_group]).astype(int)
            samples[0] = max(samples[0], 1) # need at least one sample on level 0
            cost = samples@w[best_group]

        errs = [np.sqrt(mfmc_data["variance"](samples)) for mfmc_data in best_data]

        alphas = [mfmc_data["alphas"] for mfmc_data in best_data]
        mfmc_data = {"samples" : samples, "errors" : errs, "total_cost" : cost, "alphas" : alphas}
        print("Best MFMC estimator found. Coupled models:", best_group, " Max error: ", errs, " Cost: ", cost, "\n")
        return best_group, mfmc_data

    def solve_mfmc(self, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None

        best_group, mfmc_data = self.setup_mfmc(budget=budget, eps=eps)

        samples  = mfmc_data["samples"]
        errs     = mfmc_data["errors"]
        tot_cost = mfmc_data["total_cost"]
        alphas   = mfmc_data["alphas"]

        print("\nSampling optimal MFMC estimator...\n")

        L = len(best_group)
        y  = [[0 for i in range(L)] for n in range(self.n_outputs)]
        y1 = [[0 for i in range(L-1)] for n in range(self.n_outputs)]
        for i in range(L):
            N = samples[i]
            if i > 0: N -= samples[i-1]
            sumse,_,_ = self.blue_fn(best_group[i:], N)
            for n in range(self.n_outputs):
                for j in range(i,L):
                    y[n][j] += sumse[n][j-i]
                    if j < L-1: y1[n][j] += sumse[n][j-i+1]

        for n in range(self.n_outputs):
            for i in range(L):
                y[n][i]  /= samples[i]
                if i < L-1: y1[n][i] /= samples[i]

        mu = [y[n][0] + sum(alphas[n][i]*(y[n][i+1]-y1[n][i]) for i in range(L-1)) for n in range(self.n_outputs)]

        return mu, errs, tot_cost

    def solve_mc(self, budget=None, eps=None):
        if budget is None and eps is None:
            raise ValueError("Need to specify either budget or RMSE tolerance")
        elif budget is not None and eps is not None:
            eps = None
        if eps is not None and isinstance(eps,(int,float,np.int,np.float)): eps = [eps for n in range(self.n_outputs)]

        Vs  = np.array([self.get_covariance(n)[0,0] for n in range(self.n_outputs)])
        cost = self.get_costs()[0]

        if budget is not None:
            N_MC = int(np.floor(budget/cost))
            errs  = np.sqrt(Vs/N_MC) 
            tot_cost = N_MC*cost
        else:
            N_MC = max(int(np.ceil(Vs[n]/eps[n]**2)) for n in range(self.n_outputs))
            tot_cost = N_MC*cost
            errs = np.sqrt(Vs/N_MC)

        print("Standard MC estimator ready. Max error: ", max(errs), "Cost: ", tot_cost)

        print("\nSampling standard MC estimator...\n")
        sumse,_,_ = self.blue_fn([0], N_MC)
        mu = [sumse[n][0]/N_MC for n in range(self.n_outputs)]

        return mu,errs,tot_cost

    def complexity_test(self, eps, K=3):
        print("Running cost complexity_test...")
        tot_cost = []
        for i in range(len(eps)):
            self.setup_solver(K=K, eps=eps[i], solver="cvxpy")
            #self.setup_solver(K=K, eps=eps[i], solver="gurobi", integer=True)
            tot_cost.append(sum(self.MOSAP.samples*self.MOSAP.costs))
        tot_cost = np.array(tot_cost)
        rate = np.polyfit(np.arange(len(tot_cost)), np.log2(tot_cost), 1)[0]
        print("Total costs   :", tot_cost)
        print("Estimated rate:", rate)
        return tot_cost, rate

if __name__ == '__main__':
    pass
