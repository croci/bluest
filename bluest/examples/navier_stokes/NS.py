# as in https://fenics-handson.readthedocs.io/en/latest/navierstokes/doc.html#steady-navier-stokes-flow
# problem taken from: http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html

from dolfin import *
import matplotlib.pyplot as plt
from mesh_generator import MPI_generate_NS_mesh
from mpi4py import MPI

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

set_log_level(LogLevel.ERROR)

worldcomm = MPI.COMM_WORLD
mpiRank = worldcomm.Get_rank()
mpiSize = worldcomm.Get_size()

def build_space(N_circle1, N_circle2, N_bulk, comm=worldcomm):
    """Prepare data for DGF benchmark. Return function
    space, list of boundary conditions and surface measure
    on the cylinder."""

    filestring = MPI_generate_NS_mesh(N_circle1, N_circle2, N_bulk)

    mesh = Mesh(comm)
    with XDMFFile(comm, filestring) as f:
        f.read(mesh)

    center1 = Point(0.2, 0.2)
    center2 = Point(1.0, 0.2)
    radius1 = 0.05
    radius2 = 0.08
    L = 2.2
    W = 0.41

    # Construct facet markers
    bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    for f in facets(mesh):
        mp = f.midpoint()
        if near(mp[0], 0.0):  # inflow
            bndry[f] = 1
        elif near(mp[0], L):  # outflow
            bndry[f] = 2
        elif near(mp[1], 0.0) or near(mp[1], W):  # walls
            bndry[f] = 3
        elif mp.distance(center1) <= radius1:  # first cylinder
            bndry[f] = 5
        elif mp.distance(center2) <= radius2:  # second cylinder
            bndry[f] = 6

    # Build function spaces (Taylor-Hood)
    P2 = VectorElement("P", mesh.ufl_cell(), 2)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH)

    # Prepare surface measure on cylinder
    ds_circle1 = Measure("ds", subdomain_data=bndry, subdomain_id=5)
    ds_circle2 = Measure("ds", subdomain_data=bndry, subdomain_id=6)

    return W, bndry, ds_circle1, ds_circle2

def get_bcs(W, U, bndry, comm=worldcomm):
    u_in = Expression(("4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"), degree=2, U=U, mpi_comm=comm)

    # Prepare Dirichlet boundary conditions
    bc_walls = DirichletBC(W.sub(0), (0, 0), bndry, 3)
    bc_cylinder1 = DirichletBC(W.sub(0), (0, 0), bndry, 5)
    bc_cylinder2 = DirichletBC(W.sub(0), (0, 0), bndry, 6)
    bc_in = DirichletBC(W.sub(0), u_in, bndry, 1)
    bcs = [bc_cylinder1, bc_cylinder2, bc_walls, bc_in]

    return bcs

def solve_stokes(W, nu, U, bndry, comm=worldcomm):
    """Solve steady Stokes and return the solution"""

    bcs = get_bcs(W, U, bndry, comm=comm)

    # Define variational forms
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = Constant(nu)*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx
    L = inner(Constant((0, 0)), v)*dx

    # Solve the problem
    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

    return w

def solve_navier_stokes(W, nu, U, bndry, comm=worldcomm):
    """Solve steady Navier-Stokes and return the solution"""

    bcs = get_bcs(W, U, bndry, comm=comm)

    # Define variational forms
    v, q = TestFunctions(W)

    if nu >= 0.0005:
        w = Function(W)
    else:
        nu_new = (10000*nu+1)*nu
        w = solve_navier_stokes(W, nu_new, U, bndry, comm=comm)

    u, p = split(w)
    F = Constant(nu)*inner(grad(u), grad(v))*dx + dot(dot(grad(u), u), v)*dx \
        - p*div(v)*dx - q*div(u)*dx

    # Solve the problem
    solve(F == 0, w, bcs, solver_parameters={"newton_solver": {'linear_solver': 'mumps', "absolute_tolerance": 1e-7, "relative_tolerance": 1e-6}})

    return w

def save_and_plot(w, name):
    """Saves and plots provided solution using the given
    name"""

    u, p = w.split()

    # Store to file
    with XDMFFile("results_{}/u.xdmf".format(name)) as f:
        f.write(u)
    with XDMFFile("results_{}/p.xdmf".format(name)) as f:
        f.write(p)

    # Plot
    plt.figure()
    pl = plot(u, title='velocity {}'.format(name))
    plt.colorbar(pl)
    plt.figure()
    pl = plot(p, mode='warp', title='pressure {}'.format(name))
    plt.colorbar(pl)


def postprocess(w, nu, U, ds_circle, which):
    """Return lift, drag and the pressure difference"""

    u, p = w.split()

    # Report drag and lift
    n = FacetNormal(w.function_space().mesh())
    force = -p*n + nu*dot(grad(u), n)
    F_D = assemble(-force[0]*ds_circle)
    F_L = assemble(-force[1]*ds_circle)

    U_mean = 2/3*U
    L = 0.1
    C_D = 2/(U_mean**2*L)*F_D
    C_L = 2/(U_mean**2*L)*F_L

    # Report pressure difference
    if which == 1:
        a_1 = Point(0.15, 0.2)
        a_2 = Point(0.25, 0.2)
    else:
        a_1 = Point(0.72, 0.2)
        a_2 = Point(0.88, 0.2)
    try:
        p_diff = p(a_1) - p(a_2)
    except RuntimeError:
        p_diff = 0

    return C_D, C_L, p_diff

def tasks_1_2_3_4():
    """Solve and plot alongside Stokes and Navier-Stokes"""

    # Problem data
    u_in = Expression(("4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"), degree=2, U=0.3)
    U = 0.3
    nu = 0.0005 # Re = 2/3*U*d/nu = 2/3*0.3*0.1/nu = 0.02/nu. For nu = 0.001, Re = 20

    # Discretization parameters
    N_circle = 32
    N_bulk = 64

    # Prepare function space, BCs and measure on circle
    W, bndry, ds_circle1, ds_circle2 = build_space(N_circle, N_circle, N_bulk)

    # Solve Stokes
    w = solve_stokes(W, nu, U, bndry)
    save_and_plot(w, 'stokes')

    # Solve Navier-Stokes
    w = solve_navier_stokes(W, nu, U, bndry)
    save_and_plot(w, 'navier-stokes')
    print(postprocess(w, nu, U, ds_circle1, 1))
    print(postprocess(w, nu, U, ds_circle2, 2))

    # Open and hold plot windows
    plt.show()

def tasks_5_6():
    """Run convergence analysis of drag and lift"""

    # Problem data
    U  = 0.3
    nu = 0.001

    # Push log levelo to silence DOLFIN
    old_level = get_log_level()
    warning = LogLevel.WARNING if cpp.__version__ > '2017.2.0' else WARNING
    set_log_level(warning)

    fmt_header = "{:10s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s}"
    fmt_row = "{:10d} | {:10d} | {:10d} | {:10d} | {:10.4f} | {:10.4f} | {:10.6f}"

    file1 = open("log1.txt", "w")
    file2 = open("log2.txt", "w")

    # Print table header
    file1.write(fmt_header.format("N_bulk", "N_circle1", "N_circle2", "#dofs", "C_D", "C_L", "p_diff")+"\n")
    file2.write(fmt_header.format("N_bulk", "N_circle1", "N_circle2", "#dofs", "C_D", "C_L", "p_diff")+"\n")

    # Solve on series of meshes
    for N_bulk in [16, 32, 64, 128]:
        for N_circle1 in [max(16,N_bulk//2), 4*N_bulk]:
            for N_circle2 in [max(16,N_bulk//2), 4*N_bulk]:

                # Prepare function space, BCs and measure on circle
                W, bndry, ds_circle1, ds_circle2 = build_space(N_circle1, N_circle2, N_bulk)

                # Solve Navier-Stokes
                w = solve_navier_stokes(W, nu, U, bndry)

                # Compute drag, lift
                C_D, C_L, p_diff = postprocess(w, nu, U, ds_circle1, 1)
                file1.write(fmt_row.format(N_bulk, N_circle1, N_circle2, W.dim(), C_D, C_L, p_diff)+"\n")
                C_D, C_L, p_diff = postprocess(w, nu, U, ds_circle2, 2)
                file2.write(fmt_row.format(N_bulk, N_circle1, N_circle2, W.dim(), C_D, C_L, p_diff)+"\n")

    # Pop log level
    set_log_level(old_level)

def cost_estimation(Nlist):
    from numpy import savez,array
    from time import time

    nu = 0.0005
    U  = 0.5

    times = []
    space_dims = []
    Nvals = []
    for N_bulk in Nlist:
        for N_circle1 in [4*N_bulk, max(16,N_bulk//2)]:
            for N_circle2 in [4*N_bulk, max(16,N_bulk//2)]:
                print("\n", N_bulk, N_circle1, N_circle2, "\n")
                W, bndry, ds_circle1, ds_circle2 = build_space(N_circle1, N_circle2, N_bulk)

                tic = time()
                w = solve_navier_stokes(W, nu, U, bndry)
                C_D, C_L, p_diff = postprocess(w, nu, U, ds_circle1, 1)
                C_D, C_L, p_diff = postprocess(w, nu, U, ds_circle2, 2)
                toc = time()

                times.append(toc-tic)
                space_dims.append(W.dim())
                Nvals.append([N_bulk, N_circle1, N_circle2])

    savez("NS_costs.npz", times=array(times), space_dims=array(space_dims), Nvals=array(Nvals))
    print(times, "\n\n", space_dims)

if __name__ == "__main__":

    build_space(16, 16, 32, comm=MPI.COMM_SELF)

    #cost_estimation([64,32,16])

    tasks_1_2_3_4()
    tasks_5_6()

