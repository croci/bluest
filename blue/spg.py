import numpy as np

def linesearch(evalf, x, f, g, d, M, lastfv, maxfc, fcnt):

    sigma_min = 0.1
    sigma_max = 0.9
    gamma = 10**-4

    fmax = max(lastfv)
    gtd = g@d

    alpha = 1.0
    xnew = x + alpha*d
    fnew = evalf(xnew)
    fcnt += 1

    while fnew > fmax + gamma*alpha*gtd and fcnt < maxfc:
        if alpha <= sigma_min:
            alpha *= 0.5
        else:
            a_temp = -0.5*(alpha**2)*gtd / (fnew - f - alpha*gtd)

            if a_temp < sigma_min or a_temp > sigma_max*alpha:
                a_temp = 0.5*alpha

            alpha = a_temp

        xnew = x + alpha*d
        fnew = evalf(xnew)
        fcnt += 1

    if fnew <= fmax + gamma*alpha*gtd:
        lsinfo = 0
    else:
        lsinfo = 2

    return fcnt, fnew, xnew, lsinfo


def spg(evalf, evalg, proj, x, eps=1.0e-4, maxit=200, maxfc=10**5, iprint=True, lmbda_min=10.**-30, lmbda_max=10.**30, M=10):

    n = len(x)

    if iprint:
        print("\nSPECTRAL PROJECTED GRADIENT METHOD.\n")
        print("Problem size:\t%d\n" % n)
        print(" ITER\t      F\t\t   GPSUPNORM\n")


    it = 0
    fcnt = 0

    lastfv = -np.inf*np.ones((M,))

    x = proj(x)
    f = evalf(x)
    g = evalg(x)

    fcnt += 1
    lastfv[0] = f

    gp = proj(x-g) - x

    gpsupn = abs(gp).max()
    if (gpsupn > 1.0e-15):
        lmbda = min(lmbda_max, max(lmbda_min, 1.0/gpsupn))
    else:
        lmbda = 0.0

    while gpsupn > eps and it < maxit and fcnt < maxfc:

        if iprint:
            print(" %d\t %e\t %e" % (it, f, gpsupn)) 

        it += 1

        d = proj(x - lmbda*g) - x

        fcnt, fnew, xnew, lsinfo = linesearch(evalf, x, f, g, d, M, lastfv, maxfc, fcnt)

        if (lsinfo == 2):
            spginfo = 2 
            if iprint:
                print("WARNING! SPG: Maximum of functional evaluations reached.\n");
            return {"x": x, "f": f, "gsupn": gpsupn, "it": it, "fcnt": fcnt, "spginfo": spginfo}

        f = fnew

        lastfv[it%M] = f

        gnew = evalg(xnew)

        s = xnew - x
        y = gnew - g
        sts = s@s
        sty = s@y

        x = xnew
        g = gnew   

        gp = proj(x-g) - x

        gpsupn = abs(gp).max()

        if sty <= 0:
            lmbda = lmbda_max
        else:
            lmbda = min(lmbda_max, max(lmbda_min, sts/sty))

    if iprint:
        print(" %d\t %e\t %e" % (it, f, gpsupn)) 

        print("\n")
        print("Number of iterations               : %d\n" % it)
        print("Number of functional evaluations   : %d\n" % fcnt)
        print("Objective function value           : %e\n" % f)
        print("Sup-norm of the projected gradient : %e\n" % gpsupn)

    if gpsupn <= eps:
        spginfo = 0
        if iprint:
            print("SPG: Optimal solution found.\n")
        return {"x": x, "f": f, "gpsupn": gpsupn, "it": it, "fcnt": fcnt, "spginfo": spginfo}

    if it >= maxit:
        spginfo = 1
        if iprint:
            print("WARNING! SPG: Maximum number of iterations reached.\n")
        return {"x": x, "f": f, "gpsupn": gpsupn, "it": it, "fcnt": fcnt, "spginfo": spginfo}

    if fcnt >= maxfc:
        spginfo = 2
        if iprint:
            print("WARNING! SPG: Maximum number of functional evaluations reached.\n")
        return {"x": x, "f": f, "gpsupn": gpsupn, "it": it, "fcnt": fcnt, "spginfo": spginfo}

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
    mask = mask.flatten().astype(int)
    invmask = 1-mask
    gamma = 0.

    def evalf(x):
        return 0.5*sum((mask**2*(x - A.flatten()))**2 + gamma*invmask**2*x**2)

    def evalg(x):
        return (mask**2*(x - A.flatten())) + gamma*(invmask**2*x)

    x = proj(mask*A.flatten())
    x = proj(np.random.randn(N*N))
    res = spg(evalf, evalg, proj, x, iprint=True)

