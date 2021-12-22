import numpy as np
from numba import njit,jit
from itertools import combinations, product

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
    val = np.sort(sol)[-int(1.5*N):][0] # you need to take at least -1*N or it will cause trouble to MLMC and MFMC routines
    ss = np.round(sol).astype(int)
    idx = np.argwhere(sol >= val).flatten()
    if e is not None:
        idx2 = np.argwhere(e > 0).flatten()
        temp = np.argsort(sol[e>0])[::-1]
        idx2 = idx2[temp[:N]]
        idx = np.unique(np.concatenate([idx, idx2]))

    LL = len(idx)

    lb = np.zeros((L,)); ub = np.zeros((L,))
    lb[idx] = np.floor(sol).astype(int)[idx]
    ub[idx] = np.ceil(sol).astype(int)[idx]

    return lb,ub,idx

def best_closest_integer_solution(sol, obj, constr, N, e=None):
    from functools import reduce

    lb,ub,idx = get_feasible_integer_bounds(sol, N, e=e)

    LL = len(idx)
    bnds = np.vstack([lb,ub])[:,idx]

    val = np.round(sol).astype(int)

    ee = np.ones((LL,),dtype=bool)

    def check_sol(item):
        fval = np.inf
        val[idx] = bnds[item,ee]
        if constr(val): fval = obj(val)
        val[idx] = np.round(sol[idx]).astype(int)
        return item,fval

    def reduce_func(a,b):
        if a[1] < b[1]: return a
        else:           return b

    best_item, best_fval = reduce(reduce_func, (check_sol(item) for item in product([0,1], repeat=LL)))
    val[idx] = bnds[best_item,ee]
    best_val = val.astype(int)

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
    L = len(m)
    cumsizes = np.cumsum(sizes)

    if abs(m).max() < 0.05: return np.inf, np.inf*np.ones((L,))
    PHI = get_phi_full(m,psi,delta=delta)

    invPHI = np.linalg.pinv(PHI)

    idx = get_nnz_rows_cols(m,groups,cumsizes)
    var = np.linalg.inv(PHI[idx])[0,0]
    #var = invPHI[0,0]

    grad = -np.concatenate([gradK(k, sizes[k], groups[k-1], invcovs[k-1], invPHI) for k in range(1,K+1)])

    if nohess: return var,grad,None

    hess = np.zeros((L,L))

    for k in range(1,K+1):
        for q in range(1,K+1):
            hess[cumsizes[k-1]:cumsizes[k],:][:,cumsizes[q-1]:cumsizes[q]] = hessKQ(k, q, sizes[k], sizes[q], groups[k-1], groups[q-1], invcovs[k-1], invcovs[q-1], invPHI)

    hess += hess.T

    return var,grad,hess

def PHIinvY0(m, y, psi, groups, cumsizes, delta=0.0):
    if abs(m).max() < 0.05: return np.inf

    PHI = get_phi(m, psi, delta=delta)

    idx = get_nnz_rows_cols(m,groups,cumsizes)
    PHI = PHI[idx]
    y   = y[idx[0].flatten()]

    assert idx[0].min() == 0 # the model 0 must always be sampled if this triggers something is wrong

    try:
        mu = np.linalg.solve(PHI,y)[0]
        var = np.linalg.solve(PHI, np.eye(len(y), 1).flatten())[0]
    except np.linalg.LinAlgError:
        assert False # after the above fix we should never get here
        pinvPHI = np.linalg.pinv(PHI)
        mu  = (pinvPHI@y)[0]
        var = pinvPHI[0,0] 

    return mu, var

##################################################################################################################
    
@njit(fastmath=True)
def hessKQ(k, q, Lk, Lq, groupsk, groupsq, invcovsk, invcovsq, invPHI):
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
def gradK(k, Lk,groupsk,invcovsk,invPHI):
    grad = np.zeros((Lk,))
    for i in range(Lk):
        temp = invPHI[groupsk[i],0] # PHI is symmetric
        for j in range(k):
            for l in range(k):
                grad[i] += temp[j]*invcovsk[k*k*i + k*j + l]*temp[l]

    return grad

@njit(fastmath=True)
def objectiveK(N, k,Lk,mk,groupsk,invcovsk):
    PHI = np.zeros((N*N,))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                PHI[N*group[j]+group[l]] += mk[i]*invcovsk[k*k*i + k*j + l]

    return PHI

@njit(fastmath=True)
def assemble_psi(N,k,Lk,groupsk,invcovsk):
    psi = np.zeros((N*N,Lk))
    for i in range(Lk):
        group = groupsk[i]
        for j in range(k):
            for l in range(k):
                psi[N*group[j]+group[l], i] += invcovsk[k*k*i + k*j + l]
    return psi
