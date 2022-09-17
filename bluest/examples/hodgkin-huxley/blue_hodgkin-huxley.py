from dolfin import *
from bluest import *
import numpy as np
from numpy.random import RandomState
from mpi4py import MPI
import sys
import os

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

set_log_level(LogLevel.ERROR)

worldcomm = MPI.COMM_WORLD
mpiRank = worldcomm.Get_rank()
mpiSize = worldcomm.Get_size()

RNG = RandomState(mpiRank)

################################################################################################################

gna = 120
gk = 36
gl = 0.3

vna = 56
vk = -77
vl = -60

a = 0.0238 # cm
R = 0.0354 # kOhm*cm

Cm = 1
I = 0.005*gna*vna
eps = a/(2*R)

avg_params = (Cm, I, eps)
params = avg_params

def alphan(v):
    y = np.exp(1-v/10)
    return 0.1*np.log(y)/(y-1)
def alpham(v):
    y = np.exp(2.5-v/10)
    return np.log(y)/(y-1)
alphah = lambda v : 0.07*np.exp(-v/20)
#alphan = lambda v : 0.01*(10-v)/(np.exp(1-v/10)-1)
#alpham = lambda v : 0.1*(25-v)/(np.exp(2.5-v/10)-1)
betan  = lambda v : np.exp(-v/80)/8
betam  = lambda v : 4*np.exp(-v/18)
betah  = lambda v : 1/(1 + np.exp(3-v/10))
ninf   = lambda v : alphan(v)/(alphan(v) + betan(v))
minf   = lambda v : alpham(v)/(alpham(v) + betam(v))
hinf   = lambda v : alphah(v)/(alphah(v) + betah(v))

veq = (vk*gk*ninf(0)**4 + gna*vna*minf(0)**3*hinf(0) + gl*vl)/(gk*ninf(0)**4 + gna*minf(0)**3*hinf(0) + gl)

V0 = veq
n0 = ninf(V0 - veq)
m0 = minf(V0 - veq)
h0 = hinf(V0 - veq)
ics = (V0, n0, m0, h0)

Vfn0 = veq
nfn0 = n0
hbar = ninf(Vfn0 - veq) + hinf(Vfn0 - veq)
ics_fn = (Vfn0, nfn0)

################################################################################################################

class NeuronProblem(object):
    def __init__(self, N, dt, T = 20, mpi_comm = MPI.COMM_SELF):

        self.N = N
        self.T = T

        self.T0 = 2.0
        self.M = int(np.ceil(T/dt))
        self.dt = T/self.M

        mesh = UnitIntervalMesh(mpi_comm, N)
        self.kappa = mesh.hmax()*self.dt/(self.T-self.T0)

        fe = FiniteElement('CG', mesh.ufl_cell(), 1)
        me = MixedElement([fe, fe, fe, fe])
        mefn = MixedElement([fe, fe])

        W = FunctionSpace(mesh, me)
        Wfn = FunctionSpace(mesh, mefn)
        self.W = W; self.Wfn = Wfn

        self.df    = [np.array(W.sub(i).dofmap().dofs()) for i in range(W.num_sub_spaces())]
        self.df_fn = [np.array(Wfn.sub(i).dofmap().dofs()) for i in range(Wfn.num_sub_spaces())]
        self.dfs = np.sort(np.concatenate(self.df[1:]))

        X = W.tabulate_dof_coordinates()[self.df[0],:].flatten()
        self.ix = np.argsort(X)
        self.X = X[self.ix]

        Xfn = Wfn.tabulate_dof_coordinates()[self.df_fn[0],:].flatten()
        self.ixfn = np.argsort(Xfn)
        self.Xfn = Xfn[self.ixfn]

        left  = CompiledSubDomain('x[0] < DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)
        right = CompiledSubDomain('x[0] > 1-DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)

        self.bcs    = [DirichletBC(W.sub(0),   Constant(veq), right)]
        self.bcs_fn = [DirichletBC(Wfn.sub(0), Constant(veq), right)]

        self.LHS   = lhs(self.HH_residual(Function(W),avg_params)[0])
        self.LHSfn = lhs(self.FN_residual(Function(Wfn),avg_params)[0])

    def split_dofs(self,w):
        wvec = w.vector().get_local().copy()
        L = len(w)
        if L == 4: return (wvec[self.df[i]] for i in range(L))
        else:      return (wvec[self.df_fn[i]] for i in range(L))

    def assign_dofs(self,w,vals):
        L = len(w)
        assert L == len(vals)
        if L == 4:
            for i in range(L):
                w.vector()[self.df[i]] = vals[i].copy()
        else:
            for i in range(L):
                w.vector()[self.df_fn[i]] = vals[i].copy()
        return w

    def get_deltas(self, w0, params):
        Cm,I,eps = params

        w0.vector()[self.dfs] = np.clip(w0.vector()[self.dfs], 0, 1)
        v,n,m,h = self.split_dofs(w0)

        Dn = alphan(v-veq)*(1-n) - betan(v-veq)*n
        Dm = alpham(v-veq)*(1-m) - betam(v-veq)*m
        Dh = alphah(v-veq)*(1-h) - betah(v-veq)*h

        Ina = gna*m**3*h*(v - vna)
        Ik  = gk*n**4*(v - vk)
        Il  = gl*(v - vl)

        Dv = 1/Cm*(I - (Ina + Ik + Il))

        currents = np.array([sum(Cm*Dv), sum(Ina), sum(Ik), sum(Il)])

        Dw = self.assign_dofs(Function(self.W), (-Dv, -Dn, -Dm, -Dh))
        Dv,Dn,Dm,Dh = Dw.split()

        return w0,Dv,Dn,Dm,Dh,currents

    def get_deltas_fn(self, w0, params):
        Cm,I,eps = params

        w0.vector()[self.df_fn[1]] = np.clip(w0.vector()[self.df_fn[1]], 0, 1)
        v,n = self.split_dofs(w0)

        Dn = alphan(v-veq)*(1-n) - betan(v-veq)*n

        Ina = gna*minf(v-veq)**3*(hbar - n)*(v - vna)
        Ik  = gk*n**4*(v - vk)
        Il  = gl*(v - vl)

        Dv = 1/Cm*(I - (Ina + Ik + Il))

        currents = np.array([sum(Cm*Dv), sum(Ina), sum(Ik), sum(Il)])

        Dw = self.assign_dofs(Function(self.Wfn), (-Dv, -Dn))
        Dv,Dn = Dw.split()

        return w0,Dv,Dn,currents

    def HH_residual(self, w0,params):
        dt = self.dt

        v,n,m,h = TrialFunctions(self.W)
        vt,nt,mt,ht = TestFunctions(self.W)

        Cm,I,eps = params

        w0,Dv,Dn,Dm,Dh,currents = self.get_deltas(w0,params)

        v0,n0,m0,h0 = w0.split()

        n_form = (n-n0)*nt*dx + Constant(dt)*Dn*nt*dx
        m_form = (m-m0)*mt*dx + Constant(dt)*Dm*mt*dx
        h_form = (h-h0)*ht*dx + Constant(dt)*Dh*ht*dx
        v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)

        H = v_form + n_form + m_form + h_form
        return H,currents

    def FN_residual(self, w0, params):
        dt = self.dt

        v,n   = TrialFunctions(self.Wfn)
        vt,nt = TestFunctions(self.Wfn)

        Cm,I,eps = params

        w0,Dv,Dn,currents = self.get_deltas_fn(w0,params)

        v0,n0 = w0.split()

        n_form = (n-n0)*nt*dx + Constant(dt)*Dn*nt*dx
        v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)

        F = v_form + n_form
        return F,currents

    def solve(self, model, params=avg_params, save=False, verbose=False):
        if model == 'HH':
            w0 = self.assign_dofs(Function(self.W), ics)
            LHS = self.LHS
            residual = self.HH_residual
            bcs = self.bcs
            df = self.df
            ix = self.ix
            filename = './results/voltage.pvd'
        elif model == 'FN':
            w0 = self.assign_dofs(Function(self.Wfn), ics_fn)
            LHS = self.LHSfn
            residual = self.FN_residual
            bcs = self.bcs_fn
            df = self.df_fn
            ix = self.ixfn
            filename = './results/voltage_fn.pvd'
        else:
            raise ValueError("Supported models are 'HH' or 'FN'")

        peak = 0
        currs = np.zeros((4,))
        v0 = w0.split()[0]

        if save: v_file = File(filename); v_file << v0;
        if verbose: print("t = %.2f : " % 0.0, assemble(v0*v0*dx))

        for i in range(1,self.M+1):
            H,currents = residual(w0,params)
            solve(LHS == rhs(H), w0, bcs, solver_parameters={'linear_solver':'mumps'})

            v_vals = w0.vector().get_local()[df[0]][ix]

            peak = max(peak, max(v_vals))
            if i*self.dt > self.T0:
                currs += self.kappa*currents

            if i%10 == 0:
                if verbose: print("t = %.2f : " % (i*dt), assemble(v0*v0*dx))
                if save: v_file << v0

        if verbose: print(peak, currs)
        return [peak] + list(currs)


    def solve_ODE(self, model, params=avg_params, save=False, verbose=False, plot=False):

        M  = 2*self.M
        dt = self.dt/2
        t = np.linspace(0,self.T, M+1)
        kappa = dt/(self.T-self.T0)

        Cm,I,eps = params

        if model == 'HH':
            # H-H variables
            V = np.zeros((M+1,)); V[0] = ics[0]
            n = np.zeros((M+1,)); n[0] = ics[1]
            m = np.zeros((M+1,)); m[0] = ics[2]
            h = np.zeros((M+1,)); h[0] = ics[3]

            filename = './results/voltage.npz'

        else:
            # F-N variables
            V = np.zeros((M+1,)); V[0] = ics_fn[0]
            n = np.zeros((M+1,)); n[0] = ics_fn[1]

            filename = './results/voltage_fn.npz'

        currs = np.zeros((4,))

        for i in range(M):
            if model == 'HH':
                # Hodgkin-Huxley
                Dn = alphan(V[i]-veq)*(1-n[i]) - betan(V[i]-veq)*n[i]
                Dm = alpham(V[i]-veq)*(1-m[i]) - betam(V[i]-veq)*m[i]
                Dh = alphah(V[i]-veq)*(1-h[i]) - betah(V[i]-veq)*h[i]
                n[i+1] = n[i] + dt*Dn
                m[i+1] = m[i] + dt*Dm
                h[i+1] = h[i] + dt*Dh

                Ina = gna*m[i]**3*h[i]*(V[i] - vna)
                Ik  = gk*n[i]**4*(V[i] - vk)
                Il  = gl*(V[i] - vl)

                DV = (1/Cm)*(I - (Ina + Ik + Il))
                V[i+1] = V[i] + dt*DV

                if i*dt > self.T0:
                    currs += kappa*np.array([Cm*DV, Ina, Ik, Il])

            else:
                # Fitzhugh-Nagumo
                Dn = alphan(V[i]-veq)*(1-n[i]) - betan(V[i]-veq)*n[i]
                n[i+1] = n[i] + dt*Dn

                Ina = gna*minf(V[i]-veq)**3*(hbar - n[i])*(V[i] - vna)
                Ik  = gk*n[i]**4*(V[i] - vk)
                Il  = gl*(V[i] - vl)

                DV = (1/Cm)*(I - (Ina + Ik + Il))
                V[i+1] = V[i] + dt*DV
                
                if i*dt > self.T0:
                    currs += kappa*np.array([Cm*DV, Ina, Ik, Il])

        peak = max(V)

        if verbose: print(peak, currs)
        if save: np.savez(filename, t=t, v=V)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(t, V)
            plt.show()

        return [peak] + list(currs)

    def solve_all(self, params=avg_params, save=True, verbose=True, plot=True):
        self.solve_ODE('HH', params=params, save=save, verbose=verbose, plot=plot)
        self.solve_ODE('FN', params=params, save=save, verbose=verbose, plot=plot)
        self.solve('HH', params=params, save=save, verbose=verbose)
        self.solve('FN', params=params, save=save, verbose=verbose)

no_Na_curr = False

#mesh_sizes = [16,   32,    64][::-1]
#timesteps  = [0.025, 0.0125, 0.00625][::-1]
mesh_sizes = [8,   32,    128][::-1]
timesteps  = [0.05, 0.0125, 0.003125][::-1]
neuron_problems = [NeuronProblem(N, dt, T=20) for N,dt in zip(mesh_sizes,timesteps)]

Np = len(neuron_problems)
M = 4*Np
No = 5 - int(no_Na_curr)

#NOTE: estimated from dofs. Constant in front estimated via CPU timings
costs = np.array([8*problem.M*problem.W.dim() for problem in neuron_problems] + [8*problem.M*problem.Wfn.dim() for problem in neuron_problems] + [4*problem.M for problem in neuron_problems] + [2*problem.M for problem in neuron_problems]); costs = costs/np.mean(costs)

class BLUENeuronProblem(BLUEProblem):
    def sampler(self, ls, N=1):
        L = len(ls)
        Cm = 1 + 0.2*RNG.randn()**2

        I = avg_params[1]*5.**(2*RNG.rand() - 1)

        target_mu_eps = avg_params[2]
        target_var_eps = (avg_params[2]*0.2)**2
        mu = np.log(target_mu_eps/np.sqrt(target_mu_eps**2 + target_var_eps))
        sigma = np.sqrt(np.log(1 + target_var_eps/target_mu_eps**2))
        eps  = np.exp(mu + sigma*RNG.randn())

        sample = np.array([Cm, I, eps])
        return [sample.copy() for i in range(L)]

    def evaluate(self, ls, samples, N=1):

        L = len(ls)
        out = [[0 for i in range(L)]  for n in range(No)]

        for i in range(L):

            l = ls[i]
            problem_n = l%Np
            model_to_run = ['HH', 'FN'][(l//Np)%2]
            pde = bool((l//Np) < 2)
            if pde: outputs = neuron_problems[problem_n].solve(model_to_run, params=tuple(samples[i]))
            else:   outputs = neuron_problems[problem_n].solve_ODE(model_to_run, params=tuple(samples[i]))

            if no_Na_curr: outputs = [item for i,item in enumerate(outputs) if i != 2]

            for n in range(No):
                out[n][i] = outputs[n]

        return out

if __name__ == '__main__':
    run_single = True
    try: sys.argv[1]
    except IndexError: run_single = False

    filename = './model_graph_data.npz'
    if no_Na_curr:
        filename = './model_graph_data_no_Na_curr.npz'

    if run_single:
        N = 32
        dt = 0.025
        T = 20

        problem = NeuronProblem(N, dt, T=T, mpi_comm=MPI.COMM_WORLD)
        problem.solve_all()

    else:
        if mpiRank == 0: print(costs)

        load_model_graph = os.path.exists(filename)
        if load_model_graph:
            problem = BLUENeuronProblem(M, n_outputs=No, costs = costs, datafile=filename)
        else:
            problem = BLUENeuronProblem(M, n_outputs=No, costs=costs, covariance_estimation_samples=max(min(100,mpiSize*50),50))
            problem.save_graph_data(filename)

        C = problem.get_covariances()

        vals = np.array([c[0,0] for c in C])
        eps = np.sqrt(vals)/1000; budget = None

        solver_test = False
        if solver_test:
            from time import time
            K = 7; eps = np.sqrt(vals)/1000; budget = max(costs)*10**4
            OUT = [[],[]]

            out_cvxpy,out_cvxopt,out_ipopt,out_scipy = None, None, None, None
            for i in range(2):
                for solver in ["cvxopt", "ipopt"]:
                    tic = time()
                    if i == 0: out = problem.setup_solver(K=K, budget=budget, solver=solver, continuous_relaxation=True, optimization_solver_params={'feastol':1.e-7, 'abstol':1e-6, 'reltol':1e-3})[1]
                    else:      out = problem.setup_solver(K=K, eps=eps, solver=solver, continuous_relaxation=True, optimization_solver_params={'feastol':1.e-7, 'abstol':1e-6, 'reltol':1e-3})[1]
                    toc = time() - tic
                    out = np.array([max(out['errors']), out['total_cost'], toc])
                    OUT[i].append(out)

                OUT[i] = np.vstack(OUT[i])

            for i in range(2):
                if i == 0: print("Budget: ", budget, "\n")
                else:      print("Tolerance: ", max(eps), "\n")
                print("\terrors\t   total cost\t   time\n")
                print(OUT[i], "\n")

            sys.exit(0)

        out_BLUE = problem.setup_solver(K=7, budget=budget, eps=eps)
        out_MLMC = problem.setup_mlmc(eps=eps, budget=budget)
        out_MFMC = problem.setup_mfmc(eps=eps, budget=budget)

        print("\n\n\n", out_BLUE, "\n\n", out_MLMC, "\n\n", out_MFMC)
        print("\n\n\n", out_BLUE[1]["total_cost"], "\n", out_MLMC[1]["total_cost"], "\n", out_MFMC[1]["total_cost"])
        if not no_Na_curr: np.savez("samples.npz",samples=out_BLUE[1]["samples"])
