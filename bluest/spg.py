import numpy as np

def linesearch(feval, x, f, g, d, Hlength, last_fval, max_fevals, count):

    sigma_min = 0.1
    sigma_max = 0.9
    gamma = 10**-4

    fmax = max(last_fval)
    gdotd = g@d

    alpha = 1.0
    xnew = x + alpha*d
    fnew = feval(xnew)
    count += 1

    while fnew > fmax + gamma*alpha*gdotd and count < max_fevals:
        if alpha <= sigma_min:
            alpha *= 0.5
        else:
            alpha_t = -0.5*(alpha**2)*gdotd / (fnew - f - alpha*gdotd)

            if alpha_t < sigma_min or alpha_t > sigma_max*alpha:
                alpha_t = 0.5*alpha

            alpha = alpha_t

        xnew = x + alpha*d
        fnew = feval(xnew)
        count += 1

    if fnew <= fmax + gamma*alpha*gdotd:
        linesearch_info = 0
    else:
        linesearch_info = 2

    return count, fnew, xnew, linesearch_info

def spg(feval, geval, proj, x, eps=1.0e-4, maxit=200, max_fevals=10**5, verbose=True, lmbda_min=10.**-30, lmbda_max=10.**30, Hlength=10):

    n = len(x)

    if verbose:
        print("\nSPECTRAL PROJECTED GRADIENT METHOD.\n")
        print("Problem size:\t%d\n" % n)
        print(" ITER\t      F\t\t   GPINFNORM\n")

    it = 0
    count = 0

    last_fval = -np.inf*np.ones((Hlength,))

    x = proj(x)
    f = feval(x)
    g = geval(x)

    count += 1
    last_fval[0] = f

    gp = proj(x-g) - x

    gpmax = abs(gp).max()
    if (gpmax > 1.0e-15):
        lmbda = min(lmbda_max, max(lmbda_min, 1.0/gpmax))
    else:
        lmbda = 0.0

    while gpmax > eps and it < maxit and count < max_fevals:

        if verbose:
            print(" %d\t %e\t %e" % (it, f, gpmax)) 

        it += 1

        d = proj(x - lmbda*g) - x

        count, fnew, xnew, linesearch_info = linesearch(feval, x, f, g, d, Hlength, last_fval, max_fevals, count)

        if (linesearch_info == 2):
            solver_info = 2 
            if verbose:
                print("WARNING! SPG: Maximum of functional evaluations reached.\n");
            return {"x": x, "f": f, "gpmax": gpmax, "it": it, "count": count, "solver_info": solver_info}

        f = fnew

        last_fval[it%Hlength] = f

        gnew = geval(xnew)

        s = xnew - x
        y = gnew - g
        sdots = s@s
        sdoty = s@y

        x = xnew
        g = gnew   

        gp = proj(x-g) - x

        gpmax = abs(gp).max()

        if sdoty <= 0:
            lmbda = lmbda_max
        else:
            lmbda = min(lmbda_max, max(lmbda_min, sdots/sdoty))

    if verbose:
        print(" %d\t %e\t %e" % (it, f, gpmax)) 
        print("\n")
        print("Number of iterations               : %d\n" % it)
        print("Number of functional evaluations   : %d\n" % count)
        print("Objective function value           : %e\n" % f)
        print("Sup-norm of the projected gradient : %e\n" % gpmax)

    if gpmax <= eps:
        solver_info = 0
        if verbose:
            print("SPG: Optimal solution found.\n")
        return {"x": x, "f": f, "gpmax": gpmax, "it": it, "count": count, "solver_info": solver_info}

    if it >= maxit:
        solver_info = 1
        if verbose:
            print("WARNING! SPG: Maximum number of iterations reached.\n")
        return {"x": x, "f": f, "gpmax": gpmax, "it": it, "count": count, "solver_info": solver_info}

    if count >= max_fevals:
        solver_info = 2
        if verbose:
            print("WARNING! SPG: Maximum number of functional evaluations reached.\n")
        return {"x": x, "f": f, "gpmax": gpmax, "it": it, "count": count, "solver_info": solver_info}

if __name__ == '__main__':

    N = 20

    A = np.random.randn(N,N)
    A = A.T@A
    l,V = np.linalg.eigh(A)
    l[N//2] *= -1
    A = V@np.diag(l)@V.T

    def proj(X):
        l = int(np.sqrt(len(X)).round())
        X = X.reshape((l,l))
        l,V = np.linalg.eigh((X + X.T)/2)
        l[l<0] = 0
        return (V@np.diag(l)@V.T).flatten()

    mask = (np.random.rand(N*N) > 0.1).reshape((N,N))
    mask[np.arange(N),np.arange(N)] = True
    mask = mask.flatten().astype(np.int64)
    invmask = 1-mask
    gamma = 0.

    def feval(x):
        return 0.5*sum((mask**2*(x - A.flatten()))**2 + gamma*invmask**2*x**2)

    def geval(x):
        return (mask**2*(x - A.flatten())) + gamma*(invmask**2*x)

    x = proj(mask*A.flatten())
    x = proj(np.random.randn(N*N))
    res = spg(feval, geval, proj, x, verbose=True)
