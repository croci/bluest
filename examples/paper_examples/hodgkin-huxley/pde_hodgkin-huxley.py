from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

set_log_level(LogLevel.ERROR)

mpi_comm = MPI.comm_world

T = 200 # ms
T0 = 2.0
dt = 0.025
N = int(np.ceil(T/dt))
t = np.linspace(0,T,N+1)
dt = T/N

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
hbar = ninf(V0 - veq) + hinf(V0 - veq)
ics_fn = (V0, n0)

mesh = UnitIntervalMesh(mpi_comm, 32)
fe = FiniteElement('CG', mesh.ufl_cell(), 1)
me = MixedElement([fe, fe, fe, fe])
mefn = MixedElement([fe, fe])

W = FunctionSpace(mesh, me)
Wfn = FunctionSpace(mesh, mefn)

w0 = Function(W)
w0fn = Function(Wfn)

df    = [np.array(W.sub(i).dofmap().dofs()) for i in range(W.num_sub_spaces())]
df_fn = [np.array(Wfn.sub(i).dofmap().dofs()) for i in range(Wfn.num_sub_spaces())]
dfs = np.sort(np.concatenate(df[1:]))

X = W.tabulate_dof_coordinates()[df[0],:].flatten()
ix = np.argsort(X)
X = X[ix]

Xfn = Wfn.tabulate_dof_coordinates()[df_fn[0],:].flatten()
ixfn = np.argsort(Xfn)
Xfn = Xfn[ixfn]

left  = CompiledSubDomain('x[0] < DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)
right = CompiledSubDomain('x[0] > 1-DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
ds = Measure("ds")(subdomain_data=boundaries)

bcs = []; bcs_fn = []
bc_expr = Expression('c*(5.e4*pow(t,3)*exp(-15*t))', c=0*R/np.pi/a**2, t = 0, degree = 0, mpi_comm=mpi_comm)
#bc_expr = Expression('veq + 5.e4*pow(t,3)*exp(-15*t)', veq=veq, t = 0, degree = 0, mpi_comm=mpi_comm)
#bcs = [DirichletBC(W.sub(0), bc_expr, left), DirichletBC(W.sub(0), Constant(veq), right)]
bcs    = [DirichletBC(W.sub(0),   Constant(veq), right)]
bcs_fn = [DirichletBC(Wfn.sub(0), Constant(veq), right)]

def split_dofs(w):
    wvec = w.vector().get_local().copy()
    L = len(w)
    if L == 4: return (wvec[df[i]] for i in range(L))
    else:      return (wvec[df_fn[i]] for i in range(L))

def assign_dofs(w,vals):
    L = len(w)
    assert L == len(vals)
    if L == 4:
        for i in range(L):
            w.vector()[df[i]] = vals[i].copy()
    else:
        for i in range(L):
            w.vector()[df_fn[i]] = vals[i].copy()
    return w

w0   = assign_dofs(w0, ics)
w0fn = assign_dofs(w0fn, ics_fn)

def get_deltas(w0):
    w0.vector()[dfs] = np.clip(w0.vector()[dfs], 0, 1)
    v,n,m,h = split_dofs(w0)

    Dn = alphan(v-veq)*(1-n) - betan(v-veq)*n
    Dm = alpham(v-veq)*(1-m) - betam(v-veq)*m
    Dh = alphah(v-veq)*(1-h) - betah(v-veq)*h

    Ina = gna*m**3*h*(v - vna)
    Ik  = gk*n**4*(v - vk)
    Il  = gl*(v - vl)

    Dv = 1/Cm*(I - (Ina + Ik + Il))

    currents = np.array([sum(Cm*Dv), sum(Ina), sum(Ik), sum(Il)])

    Dw = assign_dofs(Function(W), (-Dv, -Dn, -Dm, -Dh))
    Dv,Dn,Dm,Dh = Dw.split()

    return w0,Dv,Dn,Dm,Dh,currents

def get_deltas_fn(w0):
    w0.vector()[df_fn[1]] = np.clip(w0.vector()[df_fn[1]], 0, 1)
    v,n = split_dofs(w0)

    Dn = alphan(v-veq)*(1-n) - betan(v-veq)*n

    Ina = gna*minf(v-veq)**3*(hbar - n)*(v - vna)
    Ik  = gk*n**4*(v - vk)
    Il  = gl*(v - vl)

    Dv = 1/Cm*(I - (Ina + Ik + Il))

    currents = np.array([sum(Cm*Dv), sum(Ina), sum(Ik), sum(Il)])

    Dw = assign_dofs(Function(Wfn), (-Dv, -Dn))
    Dv,Dn = Dw.split()

    return w0,Dv,Dn,currents

def HH_residual(w0, t):
    v,n,m,h = TrialFunctions(W)
    vt,nt,mt,ht = TestFunctions(W)

    bc_expr.t = t

    w0,Dv,Dn,Dm,Dh,currents = get_deltas(w0)

    v0,n0,m0,h0 = w0.split()

    n_form = (n-n0)*nt*dx + Constant(dt)*Dn*nt*dx
    m_form = (m-m0)*mt*dx + Constant(dt)*Dm*mt*dx
    h_form = (h-h0)*ht*dx + Constant(dt)*Dh*ht*dx
    #v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)
    v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx - bc_expr*vt*ds(1))

    H = v_form + n_form + m_form + h_form
    return H,currents

def FN_residual(w0, t):
    v,n   = TrialFunctions(Wfn)
    vt,nt = TestFunctions(Wfn)

    bc_expr.t = t

    w0,Dv,Dn,currents = get_deltas_fn(w0)

    v0,n0 = w0.split()

    n_form = (n-n0)*nt*dx + Constant(dt)*Dn*nt*dx
    #v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)
    v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx - bc_expr*vt*ds(1))

    F = v_form + n_form
    return F,currents

Vout = []
Vfnout = []

kappa = mesh.hmax()*dt/(T-T0)
peak = 0
peak_fn = 0
currs = np.zeros((4,))
currs_fn = np.zeros((4,))

v_file = File('./results/voltage.pvd')
vfn_file = File('./results/voltage_fn.pvd')
v0,n0,m0,h0 = w0.split()
v0fn,n0fn = w0fn.split()
v_file << v0
vfn_file << v0fn
Vout.append(w0.vector().get_local().copy()[df[0]][ix])
Vfnout.append(w0fn.vector().get_local().copy()[df_fn[0]][ix])
print(assemble(v0*v0*dx), assemble(v0fn*v0fn*dx))
for i in range(1,N+1):
    # Hodgkin-Huxley
    H,currents = HH_residual(w0, i*dt)
    solve(lhs(H) == rhs(H), w0, bcs, solver_parameters={'linear_solver':'mumps'})

    # Fitzhugh-Nagumo
    F,currents_fn = FN_residual(w0fn, i*dt)
    solve(lhs(F) == rhs(F), w0fn, bcs_fn, solver_parameters={'linear_solver':'mumps'})

    v_vals = w0.vector().get_local().copy()[df[0]][ix]
    vfn_vals = w0fn.vector().get_local().copy()[df_fn[0]][ixfn]

    peak = max(peak, max(v_vals))
    peak_fn = max(peak_fn, max(vfn_vals))
    if i*dt > T0:
        currs    += kappa*currents
        currs_fn += kappa*currents_fn

    if i%10 == 0:
        print(assemble(v0*v0*dx), assemble(v0fn*v0fn*dx))
        v_file << v0
        vfn_file << v0fn
        Vout.append(v_vals)
        Vfnout.append(vfn_vals)

print(peak, peak_fn, currs, currs_fn)

Vout   = np.hstack(Vout)
Vfnout = np.hstack(Vfnout)
np.savez('./results/voltage_data.npz', v = Vout, vfn = Vfnout, x = X, t = t[::10]) 
