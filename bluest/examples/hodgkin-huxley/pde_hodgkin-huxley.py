from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

set_log_level(LogLevel.ERROR)

mpi_comm = MPI.comm_world

T = 10 # ms
dt = 0.001
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
I = 0*0.005*gna*vna
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

an = lambda v : 0.01*(10-v)/(exp(1-v/10)-1)
am = lambda v : 0.1*(25-v)/(exp(2.5-v/10)-1)
ah = lambda v : 0.07*exp(-v/20)
bn = lambda v : exp(-v/80)/8
bm = lambda v : 4*exp(-v/18)
bh = lambda v : 1/(1 + exp(3-v/10))

veq = (vk*gk*ninf(0)**4 + gna*vna*minf(0)**3*hinf(0) + gl*vl)/(gk*ninf(0)**4 + gna*minf(0)**3*hinf(0) + gl)

V0 = veq
n0 = ninf(V0 - veq)
m0 = minf(V0 - veq)
h0 = hinf(V0 - veq)

Vfn0 = veq
nfn0 = n0
hbar = ninf(V0 - veq) + hinf(V0 - veq)

mesh = UnitIntervalMesh(mpi_comm, 32)
fe = FiniteElement('CG', mesh.ufl_cell(), 1)
me = MixedElement([fe, fe, fe, fe])
mefn = MixedElement([fe, fe])

W = FunctionSpace(mesh, me)
Wfn = FunctionSpace(mesh, mefn)

w0 = Function(W)

v,n,m,h  = TrialFunctions(W)
vt,nt,mt,ht = TestFunctions(W)

dfv = np.array(W.sub(0).dofmap().dofs())
dfn = np.array(W.sub(1).dofmap().dofs())
dfm = np.array(W.sub(2).dofmap().dofs())
dfh = np.array(W.sub(3).dofmap().dofs())
dfs = np.sort(np.concatenate([dfn,dfm,dfh]))

w0.vector()[dfv] = V0
w0.vector()[dfn] = n0
w0.vector()[dfm] = m0
w0.vector()[dfh] = h0

left  = CompiledSubDomain('x[0] < DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)
right = CompiledSubDomain('x[0] > 1-DOLFIN_EPS && on_boundary', mpi_comm=mpi_comm)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
ds = Measure("ds")(subdomain_data=boundaries)

bcs = []
bc_expr = Expression('-c*(5.e3*pow(t,3)*exp(-15*t))', c=R/np.pi/a**2, t = 0, degree = 0, mpi_comm=mpi_comm)
#bc_expr = Expression('veq + 5.e4*pow(t,3)*exp(-15*t)', veq=veq, t = 0, degree = 0, mpi_comm=mpi_comm)
#bcs = [DirichletBC(W.sub(0), bc_expr, left), DirichletBC(W.sub(0), Constant(veq), right)]

#def get_veq():
#
#    V = FunctionSpace(mesh, fe)
#
#    u = Function(V)
#    v = TrialFunction(V)
#    vt = TestFunction(V)
#
#    Ina = Constant(gna*minf(0)**3*hinf(0))*(v - Constant(vna))
#    Ik  = Constant(gk*ninf(0)**4)*(v - Constant(vk))
#    Il  = Constant(gl)*(v - Constant(vl))
#
#    Dv = Constant(1/Cm)*(I - (Ina + Ik + Il))
#    F = (Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)
#    solve(lhs(F) == rhs(F), u, bcs, solver_parameters={'linear_solver':'mumps'})
#
#    w = Function(W)
#
#    w.split()[0].assign(u)
#    print(u.vector().get_local())
#
#    return w.vector()[dfv]
#
#print(get_veq())
#import sys; sys.exit(0)

def split_dofs(w):
    v = w.vector()[dfv]
    n = w.vector()[dfn]
    m = w.vector()[dfm]
    h = w.vector()[dfh]
    return v,n,m,h

def assign_dofs(w,v,n,m,h):
     w.vector()[dfv] = v
     w.vector()[dfn] = n
     w.vector()[dfm] = m
     w.vector()[dfh] = h
     return w

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

    Dw = assign_dofs(Function(W), Dv, Dn, Dm, Dh)
    Dv,Dn,Dm,Dh = Dw.split()

    return Dv,Dn,Dm,Dh

Dv,Dn,Dm,Dh = get_deltas(w0)

v0,n0,m0,h0 = w0.split()

n_form = (n-n0)*nt*dx + Constant(dt)*Dn*nt*dx
m_form = (m-m0)*mt*dx + Constant(dt)*Dm*mt*dx
h_form = (h-h0)*ht*dx + Constant(dt)*Dh*ht*dx
#v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx)
v_form = (v-v0)*vt*dx + Constant(dt)*(Dv*vt*dx + Constant(eps)*inner(grad(v),grad(vt))*dx - bc_expr*vt*ds(1))

F = v_form + n_form + m_form + h_form

v_file = File('./results/voltage.pvd')
v_file << w0.sub(0)
for i in range(1,N+1):
    # Hodgkin-Huxley
    bc_expr.t = i*dt
    solve(lhs(F) == rhs(F), w0, bcs, solver_parameters={'linear_solver':'mumps'})

    if i%10 == 0:
        v_file << w0.sub(0)

    ## Fitzhugh-Nagumo
    #Dnfn = alphan(Vfn[i]-veq)*(1-nfn[i]) - betan(Vfn[i]-veq)*nfn[i]
    #nfn[i+1] = nfn[i] + dt*Dnfn

    #Ina_fn = gna*minf(Vfn[i]-veq)**3*(hbar - nfn[i])*(Vfn[i] - vna)
    #Ik_fn  = gk*nfn[i]**4*(Vfn[i] - vk)
    #Il_fn  = gl*(Vfn[i] - vl)

    #DVfn = (1/3/Cm)*(I - (Ina_fn + Ik_fn + Il_fn))
    #Vfn[i+1] = Vfn[i] + dt*DVfn

