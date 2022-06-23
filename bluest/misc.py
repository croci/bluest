import numpy as np
from itertools import combinations, product

try: from numba import njit
except ImportError: 
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .cmisc import assemble_psi_c,objectiveK_c,gradK_c,hessKQ_c,cleanupK_c

##################################################################################################################

def attempt_mlmc_setup(v, w, budget=None, eps=None):
    if budget is None and eps is None:
        raise ValueError("Need to specify either budget or RMSE tolerance")
    elif budget is not None and eps is not None:
        eps = None

    if not all(np.isfinite(v)): return False,None

    q = sum(np.sqrt(v*w))
    if budget is not None: mu = budget/q
    else:                  mu = q/eps**2
    m = mu*np.sqrt(v/w)
    m = np.maximum(m,1)

    variance = lambda m : sum(v[m>0]/m[m>0])
    if budget is not None:
        constraint = lambda m : m@w <= budget and all(m >= 1)
        obj = variance
    else:
        constraint = lambda m : variance(m) <= eps**2 and all(m >= 1)
        obj = lambda m : m@w

    m,fval = best_closest_integer_solution(m, obj, constraint, len(v))
    if np.isinf(fval): return False,None

    err = np.sqrt(variance(m))
    tot_cost = m@w

    mlmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost, "variance" : variance}

    return True,mlmc_data

def attempt_mfmc_setup(sigmas, rhos, costs, budget=None, eps=None):
    if budget is None and eps is None:
        raise ValueError("Need to specify either budget or RMSE tolerance")
    elif budget is not None and eps is not None:
        eps = None

    if not all(np.isfinite(sigmas)): return False,None

    idx = np.argsort(abs(rhos))[::-1]
    assert idx[0] == 0

    s = sigmas[idx]
    rho = np.concatenate([rhos[idx], [0]])
    w = costs[idx]
    cost_ratio = w[:-1]/w[1:]
    rho_ratio = (rho[:-2]**2 - rho[1:-1]**2)/(rho[1:-1]**2 - rho[2:]**2)

    feasible = all(cost_ratio > rho_ratio)
    if not feasible: return feasible,None

    alphas = rho[1:-1]*s[0]/s[1:]

    r = np.sqrt(w[0]/w*(rho[:-1]**2 - rho[1:]**2)/(1-rho[1]**2))
    if budget is not None: m1 = budget/(w@r)
    else:                  m1 = eps**-2*(w@r)*(s[0]**2/w[0])*(1-rho[1]**2)
    m = np.concatenate([[m1], m1*r[1:]])
    m = np.maximum(m, 1)

    variance = lambda m : s[0]**2/m[0] + sum((1/m[:-1]-1/m[1:])*(alphas**2*s[1:]**2 - 2*alphas*rho[1:-1]*s[0]*s[1:]))
    if budget is not None:
        constraint = lambda m : m@w <= budget and m[0] >= 1 and all(m[:-1] <= m[1:])
        obj = variance
    else:
        constraint = lambda m : variance(m) <= eps**2 and m[0] >= 1 and all(m[:-1] <= m[1:])
        obj = lambda m : m@w

    m,fval = best_closest_integer_solution(m, obj, constraint, len(sigmas))
    if np.isinf(fval): return False,None

    err = np.sqrt(variance(m))
    tot_cost = m@w

    mfmc_data = {"samples" : m, "error" : err, "total_cost" : tot_cost, "alphas" : alphas, "variance" : variance}

    return feasible,mfmc_data

##################################################################################################################

def check_solution(item, ss, bnds, idx, constr, obj):
    fval = np.inf
    val = ss.copy()
    val[idx] = bnds[item, idx]
    if constr(val): fval = obj(val)
    return val,fval

def get_feasible_integer_bounds(sol, N, e=None):
    L = len(sol)
    #val = np.sort(sol)[-int(1.2*N):][0] # you need to take at least -1*N or it will cause trouble to MLMC and MFMC routines
    idx = np.argsort(sol)[-int(1.2*N):] # you need to take at least -1*N or it will cause trouble to MLMC and MFMC routines
    idx = np.array([item for item in idx if sol[item] > 1.0e-8])
    ss = np.round(sol).astype(int)
    #idx = np.argwhere(sol >= val).flatten()
    if e is not None:
        if sum(e > 0.99) == 0:
            val = 1/sum(e)/2
            while sum(e > val) == 0: val /= 2
        else: val = 0.99
        idx2 = np.argwhere(e > val).flatten()
        temp = np.argsort(sol[e>val])[::-1]
        idx2 = idx2[temp[:N]]
        idx = np.unique(np.concatenate([idx, idx2]))

    LL = len(idx)

    lb = np.zeros((L,),dtype=int); ub = np.zeros((L,),dtype=int)
    lb[idx] = np.floor(sol).astype(int)[idx]
    ub[idx] = np.ceil(sol).astype(int)[idx]

    temp = np.argsort(lb[idx])[::-1]
    idx = idx[temp]

    return lb[idx],ub[idx],idx

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(np.uint8).reshape(xshape + [num_bits])

def best_closest_integer_solution_BLUE_multi(sol, psis, w, e, mappings, budget=None, eps=None):
    
    No = len(mappings)

    N = int(round(np.sqrt(psis[0].shape[0])))

    lb_full,ub_full,idx_full = get_feasible_integer_bounds(sol, N, e=e)

    LL = len(idx_full)

    LL_max = 15

    if LL <= LL_max:
        best_val, best_fval = best_closest_integer_solution_BLUE_multi_helper(sol, psis, w, e, mappings, budget, eps, lb_full, ub_full, idx_full)

    else:
        print('WARNING! Too many dimensions to brute-force it. Randomising search. Note: result might not be optimal.')
        best_val = None; best_fval = np.inf
        trial = 0
        while best_val is None and trial < 250:
            trial += 1
            print("Randomisation n %d." % trial)

            sample = np.random.permutation(LL)
            brute_force = sample[:LL_max]
            random_choice = sample[LL_max:]

            idx = idx_full[brute_force]
            lb  = lb_full[brute_force]
            ub  = ub_full[brute_force]

            LR = LL-LL_max
            r_idx = idx_full[random_choice]
            r_lb  = lb_full[random_choice]
            r_ub  = ub_full[random_choice]
            r_sol = sol.copy()
            r_bnds = np.vstack([r_lb,r_ub])
            ee = np.ones((LR,), dtype=bool)
            comb = np.random.randint(2, size=LR)
            r_sol[r_idx] = r_bnds[comb, ee].T

            best_val, best_fval = best_closest_integer_solution_BLUE_multi_helper(r_sol, psis, w, e, mappings, budget, eps, lb, ub, idx)

        if trial >= 100 and best_val is None:
            print("Unable to find feasible integer solution.")
            return None,np.inf

        print("Success!")

    return best_val, best_fval

def best_closest_integer_solution_BLUE_multi_helper(sol, psis, w, e, mappings, budget, eps, lb, ub, idx):
    from functools import reduce

    No = len(mappings)

    N = int(round(np.sqrt(psis[0].shape[0])))

    LL = len(idx)

    combs = unpackbits(np.arange(2**LL, dtype=int), LL)
    bnds = np.vstack([lb,ub])
    ee = np.ones((LL,),dtype=bool)
    ms = bnds[combs, ee].T

    val = np.round(sol).astype(int)
    subval = val[idx].copy()

    baseval  = val.copy(); baseval[idx] = 0 
    basephis  = [psis[n]@baseval[mappings[n]] for n in range(No)]
    basecost = w@baseval
    basees    = [e[mappings[n]]@baseval[mappings[n]] for n in range(No)]

    redmaps = [np.array([i for i in range(len(idx)) if idx[i] in mappings[n]]) for n in range(No)]
    #redmaps = [np.array([np.argwhere(idx == item).flatten()[0] for item in mappings[n] if item in idx]) for n in range(No)]
    idxs = [np.array([np.argwhere(mappings[n] == item).flatten()[0] for item in idx if item in mappings[n]]) for n in range(No)]

    es = []
    for n in range(No):
        basee = basees[n]
        if basee < 1:
            es.append(np.argwhere(basee + e[idx][redmaps[n]]@ms[redmaps[n],:] >= 1).flatten())

    if len(es) == 0: return None, np.inf
    es = np.unique(np.concatenate(es))
    ms = ms[:, es]

    if budget is not None and basecost > budget: return None,np.inf

    costs = basecost + w[idx]@ms

    if budget is not None:
        # the ms are roughly ordered from largest to smallest
        ind = np.argwhere(costs <= 1.0001*budget).flatten()
        if len(ind) > 0: ms = ms[:, ind][:,::-1]
        else: return None, np.inf
    else:
        # larger costs should correspond to smaller variances so reordering
        ms = ms[:, np.argsort(costs)[::-1]]

    phis  = [(basephis[n].reshape((-1,1)) + psis[n][:,idxs[n]]@ms[redmaps[n],:]).T.reshape((-1,N,N)) for n in range(No)]
    Vs    = [np.linalg.pinv(phis[n], hermitian=True)[:,0,0] for n in range(No)]
    V_max = reduce(np.maximum, Vs)

    if budget is not None:
        i = np.argmin(V_max)
    else:
        # we ordered ms by cost earlier so this is optimal
        i = np.argwhere(reduce(np.logical_and, [Vs[n] <= 1.0001*eps[n]**2 for n in range(No)])).flatten()
        if len(i) > 0: i = i[-1]
        else: return None, np.inf

    val[idx] = ms[:, i]
    best_val = val
    best_fval = V_max[i]

    #assert all(abs(Vs[n][i] - np.linalg.pinv((psis[n]@best_val[mappings[n]]).reshape((N,N)), hermitian=True)[0,0]) < 1.0e-6 for n in range(No))

    return best_val, best_fval

def best_closest_integer_solution_BLUE(sol, psi, w, e, budget=None, eps=None):
    N = int(round(np.sqrt(psi.shape[0])))

    lb,ub,idx = get_feasible_integer_bounds(sol, N, e=e)

    LL = len(idx)
    if LL > 24:
        raise ValueError('Too many dimensions to brute-force it')

    combs = unpackbits(np.arange(2**LL, dtype=int), LL)
    bnds = np.vstack([lb,ub])
    ee = np.ones((LL,),dtype=bool)
    ms = bnds[combs, ee].T

    val = np.round(sol).astype(int)
    subval = val[idx].copy()

    baseval  = val.copy(); baseval[idx] = 0 
    basephi  = psi@baseval
    basecost = w@baseval
    basee    = e@baseval

    if basee < 1:
        es = basee + e[idx]@ms
        try: ms = ms[:, np.argwhere(es >= 1).flatten()]
        except IndexError: return None,np.inf

    if budget is not None and basecost > budget: return None,np.inf

    costs = basecost + w[idx]@ms

    if budget is not None:
        # the ms are roughly ordered from largest to smallest
        ms = ms[:, np.argwhere(costs <= 1.0001*budget).flatten()][:,::-1]
    else:
        # larger costs should correspond to smaller variances so reordering
        ms = ms[:, np.argsort(costs)[::-1]]

    phis  = (basephi.reshape((-1,1)) + psi[:,idx]@ms).T.reshape((-1,N,N))
    Vs    = np.linalg.pinv(phis, hermitian=True, rcond=1.e-10)[:,0,0]

    if budget is not None:
        i = np.argmin(Vs)
    else:
        # we ordered ms by cost earlier so this is optimal
        try: i = np.argwhere(Vs <= 1.0001*eps**2).flatten()[-1]
        except IndexError: return None, np.inf

    val[idx] = ms[:, i]
    best_val = val
    best_fval = Vs[i]

    return best_val, best_fval

def best_closest_integer_solution(sol, obj, constr, N, e=None):

    lb,ub,idx = get_feasible_integer_bounds(sol, N, e=e)

    LL = len(idx)
    if LL > 24:
        raise ValueError('Too many dimensions to brute-force it')

    combs = unpackbits(np.arange(2**LL, dtype=int), LL)
    bnds = np.vstack([lb,ub])
    ee = np.ones((LL,),dtype=bool)
    ms = bnds[combs, ee]

    val = np.round(sol).astype(int)
    subval = val[idx].copy()

    def check_sol(i):
        fval = np.inf
        val[idx] = ms[i]
        if constr(val): fval = obj(val)
        val[idx] = subval
        return fval

    fvals = np.array([check_sol(i) for i in range(ms.shape[0])])
    i = np.argmin(fvals)
    best_val = val.copy()
    best_val[idx] = ms[i].copy()
    best_fval = fvals[i]

    return best_val, best_fval

##################################################################################################################

def get_nnz_rows_cols(m,groups,cumsizes):
    K = len(cumsizes)-1
    m = [m[cumsizes[k]:cumsizes[k+1]] for k in range(K)]
    out = np.unique(np.concatenate([groups[k][abs(m[k]) > 1.0e-6].flatten() for k in range(K)]))
    return out.reshape((len(out),1)), out.reshape((1,len(out)))

def get_phi_full(m, psi, delta=0.0):
    N = int(round(np.sqrt(psi.shape[0])))
    return delta*np.eye(N) + (psi@m).reshape((N,N))

def variance_full(m, psi, groups, cumsizes, delta=0.0):
    if abs(m).max() < 0.05: return np.inf
    PHI = get_phi_full(m, psi, delta=delta)

    idx = get_nnz_rows_cols(m,groups,cumsizes)
    PHI = PHI[idx]

    assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

    try: out = np.linalg.solve(PHI,np.eye(len(idx[0]),1).flatten())[0]
    except np.linalg.LinAlgError:
        assert False # after the above fix we should never get here
        out = np.linalg.pinv(PHI)[0,0]

    return out

def variance_GH_full(m, psi, groups, sizes, invcovs, delta=0.0, nohess=False):
    K = len(groups)
    L = len(m)
    cumsizes = np.cumsum(sizes)

    if abs(m).max() < 0.05: return np.inf, np.inf*np.ones((L,))
    PHI = get_phi_full(m,psi,delta=delta)

    invPHI = np.linalg.pinv(PHI)

    idx = get_nnz_rows_cols(m,groups,cumsizes)
    var = np.linalg.pinv(PHI[idx])[0,0]
    #var = invPHI[0,0]

    grad = -np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

    if nohess: return var,grad,None

    hess = np.zeros((L,L))

    for k in range(1,K+1):
        for q in range(1,K+1):
            hess[cumsizes[k-1]:cumsizes[k],:][:,cumsizes[q-1]:cumsizes[q]] = hessKQ(k, q, sizes[k], sizes[q], groups[k-1], groups[q-1], invcovs[k-1], invcovs[q-1], invPHI)

    hess += hess.T

    return var,grad,hess

def assemble_cleanup_matrix(m, psi, groups, sizes, invcovs, delta=0.0):
    K = len(groups)

    if abs(m).max() < 0.05: raise ValueError("No entry greater or equal than 1 found in m.")
    PHI = get_phi_full(m,psi,delta=delta)

    invPHI = np.linalg.pinv(PHI)

    X = np.hstack([cleanupK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])
    return X

def PHIinvY0(m, y, psi, groups, cumsizes, delta=0.0):
    if abs(m).max() < 0.05: return np.inf

    PHI = get_phi_full(m, psi, delta=delta)

    idx = get_nnz_rows_cols(m,groups,cumsizes)
    PHI = PHI[idx]
    y   = [y[item] for item in idx[0].flatten()]

    assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

    pinvPHI = np.linalg.pinv(PHI)
    var = pinvPHI[0,0]
    mu = 0
    for j in range(len(y)):
        mu += pinvPHI[0,j]*y[j]

    #try:
    #    mu = np.linalg.solve(PHI,y)[0]
    #    var = np.linalg.solve(PHI, np.eye(len(y), 1).flatten())[0]
    #except np.linalg.LinAlgError:
    #    assert False # after the above fix we should never get here
    #    pinvPHI = np.linalg.pinv(PHI)
    #    mu  = pinvPHI[0,:]@y
    #    var = pinvPHI[0,0] 

    return mu, var

###################################################################################################################

@njit(fastmath=True)
def hessKQ_numba(k, q, Lk, Lq, groupsk, groupsq, invcovsk, invcovsq, invPHI):
    hess = np.zeros((Lk,Lq))
    ip = invPHI[:,0]
    ksq = k*k; qsq = q*q
    for ik in range(Lk):
        ipk = ip[groupsk[ik]]
        for iq in range(Lq):
            ipq = ip[groupsq[iq]]
            iP = invPHI[groupsk[ik],:][:,groupsq[iq]]
            #ipk@icovk@invPHI@icovq@ipq + the simmetric
            #ipk_m Ck_ms*(sum_j iP_sj*(sumi_l Cq_jl*ipq_l)_j)_s
            for lk in range(k):
                for jk in range(k):
                    for jq in range(q):
                        for lq in range(q):
                            hess[ik,iq] += ipk[lk]*invcovsk[ksq*ik + k*lk + jk]*iP[jk,jq]*invcovsq[qsq*iq + q*jq + lq]*ipq[lq]

    return hess

@njit(fastmath=True)
def gradK_numba(k, Lk,groupsk,invcovsk,invPHI):
    grad = np.zeros((Lk,))
    for i in range(Lk):
        temp = invPHI[groupsk[i],0] # PHI is symmetric
        for j in range(k):
            for l in range(k):
                grad[i] += temp[j]*invcovsk[k*k*i + k*j + l]*temp[l]

    return grad

@njit(fastmath=True)
def objectiveK_numba(N, k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                PHI[N*groupsk[j]+group[l]] += mk[i]*invcovsk[k*k*i + k*j + l]

    return PHI

@njit(fastmath=True)
def assemble_psi_numba(N,k,Lk,groupsk,invcovsk):
    psi = np.zeros((N*N,Lk))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                psi[N*group[j]+group[l], i] += invcovsk[k*k*i + k*j + l]
    return psi

def assemble_psi(N,k,Lk,groupsk,invcovsk):
    psi = np.zeros((N*N,Lk), order='C')
    assemble_psi_c(psi.ravel(order='C'), N, k, Lk, groupsk.ravel(order='C'), invcovsk)
    #assert np.allclose(psi, assemble_psi_numba(N,k,Lk,groupsk,invcovsk))
    return psi

def cleanupK(k, Lk, groupsk, invcovsk, invPHI):
    N = invPHI.shape[0]
    X = np.zeros((N,Lk), order='C')
    cleanupK_c(X.ravel(order='C'), k, Lk, groupsk.ravel(order='C'), invcovsk, invPHI[0])
    return X

def objectiveK(N, k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    objectiveK_c(PHI, k,Lk,mk,groupsk.ravel(order='C'),invcovsk)
    #assert np.allclose(PHI, objectiveK_numba(N,k,Lk,mk,groupsk,invcovsk))
    return PHI

def gradK(k, Lk, groupsk, invcovsk, invPHI):
    grad = np.zeros((Lk,))
    gradK_c(grad, k, Lk, groupsk.ravel(order='C'), invcovsk, invPHI[0])
    #assert np.allclose(grad, gradK_numba(k,Lk,groupsk,invcovsk,invPHI))
    return grad

def hessKQ(k, q, Lk, Lq, groupsk, groupsq, invcovsk, invcovsq, invPHI):
    N = invPHI.shape[0]
    hess = np.zeros((Lk,Lq), order='C')
    hessKQ_c(hess.ravel(order='C'), N, k, q, Lk, Lq, groupsk.ravel(order='C'), groupsq.ravel(order='C'), invcovsk, invcovsq, invPHI.ravel(order='C'))
    #assert np.allclose(hess, hessKQ_numba(k,q,Lk,Lq,groupsk,groupsq,invcovsk,invcovsq,invPHI))
    return hess
