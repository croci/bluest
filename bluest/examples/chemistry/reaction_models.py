from dolfin import *
from numpy import array

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
set_log_level(LogLevel.ERROR)

class Brussellator(object):
    def __init__(self, N, Nt, T=10, a=1, b=3, alpha=1./50):
        self.N = N
        self.Nt = Nt
        self.T = T
        self.dt = T/Nt

        mesh = UnitIntervalMesh(MPI.comm_self, N)

        fe = FiniteElement('CG', mesh.ufl_cell(), 1)
        re = FiniteElement('R', mesh.ufl_cell(), 0)
        me = MixedElement([fe, fe])
        mre = MixedElement([fe, re])

        self.V = FunctionSpace(mesh, fe)
        self.VR = FunctionSpace(mesh, mre)
        self.W = FunctionSpace(mesh, me)

        self.nominal_params = (a,b,alpha)
        self.nominal = None
        self.solve(a,b,alpha,save_nominal=True)

    def get_nominal(self, which, i):
        w = Function(self.W)
        out = Function(self.V)
        w.vector()[:] = self.nominal[i,:]
        assign(out, w.sub(which))
        return out

    def solve_avg(self, which, a=1, b=3, alpha=1./50, save=False, verbose=False):
        assert isinstance(which, int) and which in [0,1]

        N  = self.N
        T  = self.T
        dt = self.dt
        V  = self.V

        z = TrialFunction(V)
        s = TestFunction(V)

        avg = self.get_nominal(which, 0)

        bcs = [DirichletBC(V, Constant(a), "on_boundary"), DirichletBC(V, Constant(b), "on_boundary")][which]

        z0 = Function(V)
        if which == 0:
            assign(z0, interpolate(Expression('a + sin(2*DOLFIN_PI*x[0])', a=a, degree = 4, mpi_comm=MPI.comm_self), V))
        else:
            assign(z0, interpolate(Constant(b),V))

        if which == 0:
            L = Constant(alpha)*inner(grad(z),grad(s))*dx + Constant(b+1)*z*s*dx
            N = -z0*z0*avg*s*dx
            f = -Constant(a)*s*dx
        else:
            L = Constant(alpha)*inner(grad(z),grad(s))*dx - Constant(b)*avg*s*dx
            N = avg*avg*z*s*dx
            f = -Constant(0.0)*s*dx

        F = (z-z0)*s*dx + Constant(dt)*(L+N+f)

        if save: out = File("z%d.pvd" % which); out << z0

        t = 0.0
        maxz = 0.0
        cz = 0
        for i in range(1, Nt+1):
            t = i*dt

            assign(avg, self.get_nominal(which, i))
            solve(lhs(F) == rhs(F), z0, bcs)

            maxz = max(maxz, z0.vector().get_local().max())
            cz += assemble(z0*dx)

            if save: out << z0

        cz /= Nt
        if verbose: print(cz, maxz)
        return cz,maxz

    def solve_ode(self, which, a=1, b=3, alpha=1./50, save=False, verbose=False):
        N  = self.N
        T  = self.T
        dt = self.dt
        V  = self.V
        VR  = self.VR

        z,r = TrialFunctions(VR)
        s,k = TestFunctions(VR)

        bcs = [DirichletBC(VR.sub(0), Constant(a), "on_boundary"), DirichletBC(VR.sub(0), Constant(b), "on_boundary")][which]

        zr0 = Function(VR)
        z0,r0 = split(zr0)
        if which == 0:
            assign(zr0.sub(0), interpolate(Expression('a + sin(2*DOLFIN_PI*x[0])', a=a, degree = 4, mpi_comm=MPI.comm_self), V))
        else:
            assign(zr0.sub(0), interpolate(Constant(b),V))

        if which == 0:
            L1 = Constant(alpha)*inner(grad(z),grad(s))*dx + Constant(b+1)*z*s*dx
            L2 = - Constant(b) * z*k*dx
            N1 = -z0*z0*r*s*dx
            N2 =  z0*z0*r*k*dx
            f1 = -Constant(a)*s*dx
            f2 = -Constant(0.0)*k*dx
        else:
            L1 = Constant(b+1)*r*k*dx
            L2 = Constant(alpha)*inner(grad(z),grad(s))*dx - Constant(b) * r*s*dx
            N1 = -r0*r0*z0*k*dx
            N2 =  r0*r0*z0*s*dx
            f1 = -Constant(a)*k*dx
            f2 = -Constant(0.0)*s*dx

        F = (z-z0)*s*dx + (r-r0)*k*dx + Constant(dt)*(L1+L2+N1+N2+f1+f2)

        if save:
            out = File("ode_z%d.pvd" % which); out << zr0.sub(0)

        t = 0.0
        maxz,maxr = 0,0
        cz,cr = 0,0
        for i in range(1, Nt+1):
            t = i*dt
            solve(lhs(F) == rhs(F), zr0, bcs)

            maxz = max(maxz, zr0.sub(0).vector().get_local().max())
            maxr = max(maxr, zr0.sub(1).vector().get_local().max())

            cz += assemble(z0*dx)
            cr += assemble(r0*dx)

            if save:
                out << zr0.sub(0)

        cz /= Nt
        cr /= Nt
        if verbose: print(cz, maxz, cr, maxr)
        return cz,maxz,cr,maxr

    def solve(self, a=1, b=3, alpha=1./50, save=False, verbose=False, save_nominal=False):
        N  = self.N
        T  = self.T
        dt = self.dt
        V  = self.V
        W  = self.W

        u,v = TrialFunctions(W)
        p,q = TestFunctions(W)

        bcs = [DirichletBC(W.sub(0), Constant(a), "on_boundary"), DirichletBC(W.sub(1), Constant(b), "on_boundary")] 

        w0 = Function(W)
        u0,v0 = split(w0)
        assign(w0.sub(0), interpolate(Expression('a + sin(2*DOLFIN_PI*x[0])', a=a, degree = 4, mpi_comm=MPI.comm_self), V))
        assign(w0.sub(1), interpolate(Constant(b),V))

        L1 = Constant(alpha)*inner(grad(u),grad(p))*dx + Constant(b+1)*u*p*dx
        L2 = Constant(alpha)*inner(grad(v),grad(q))*dx - Constant(b) * u*q*dx
        N1 = -u0*u0*v0*p*dx
        N2 =  u0*u0*v0*q*dx
        f1 = -Constant(a)*p*dx
        f2 = -Constant(0.0)*q*dx

        F = (u-u0)*p*dx + (v-v0)*q*dx + Constant(dt)*(L1+L2+N1+N2+f1+f2)

        if save:
            out1 = File("u.pvd"); out1 << w0.sub(0)
            out2 = File("v.pvd"); out2 << w0.sub(1)

        t = 0.0
        maxu,maxv = 0,0
        cu,cv = 0,0
        if save_nominal: self.nominal = [w0.vector().get_local()]
        for i in range(1, Nt+1):
            t = i*dt
            solve(lhs(F) == rhs(F), w0, bcs)

            maxu = max(maxu, w0.sub(0).vector().get_local().max())
            maxv = max(maxv, w0.sub(1).vector().get_local().max())
            cu += assemble(u0*dx)
            cv += assemble(v0*dx)

            if save_nominal: self.nominal.append(w0.vector().get_local())

            if save:
                out1 << w0.sub(0)
                out2 << w0.sub(1)

        cu /= Nt
        cv /= Nt
        if verbose: print(cu, maxu, cv, maxv)
        if save_nominal: self.nominal = array(self.nominal)
        return cu,maxu,cv,maxv

if __name__ == "__main__":
    N = 64
    Nt = 100

    problem = Brussellator(N,Nt, T=10)
    problem.solve(a=1,b=4,alpha=2/50, save=True, verbose=True)
    problem.solve_avg(0, a=1, b=4, alpha = 2/50, save=True, verbose=True)
    problem.solve_avg(1, a=1, b=4, alpha = 2/50, save=True, verbose=True)
    problem.solve_ode(0, a=1, b=4, alpha = 2/50, save=True, verbose=True)
    problem.solve_ode(1, a=1, b=4, alpha = 2/50, save=True, verbose=True)

