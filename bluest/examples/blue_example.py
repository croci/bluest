from dolfin import *

set_log_level(30)

mpiRank = MPI.rank(MPI.comm_world)

class TestFunctional(OutputFunctional):
    def evaluate(self, V, sample = None, rv_sample = None, level_info = (None,None)):

        u = TrialFunction(V)
        v = TestFunction(V)

        sol = Function(V)

        lhs = inner(exp(sample)*grad(u), grad(v))*dx
        rhs = Constant(1.0)*v*dx

        bcs = DirichletBC(u.function_space(), Constant(0.0), 'on_boundary')

        solve(lhs == rhs, sol, bcs)

        E1 = assemble(sol*sol*dx)
        E2 = assemble(inner(grad(sol),grad(sol))*dx)

        # NOTE: the output of an OutputFunctional must always be a list!!!
        return [E1, E2]

    def get_n_outputs(self):
        return 2

if __name__ == '__main__':

    from numpy.random import RandomState
    import sys

    dim = 2 # spatial dimension
    n_levels  = 6

    try: N = int(sys.argv[1])
    except IndexError: 
        raise ValueError("Must specify the number of samples for MLMC convergence tests!")

    RNG = RandomState(mpiRank)

    outer_meshes = [RectangleMesh(MPI.comm_self, Point(-1,-1), Point(1,1), 2**(l+1), 2**(l+1)) for l in range(1, n_levels+1)]
    inner_meshes = [RectangleMesh(MPI.comm_self, Point(-0.5,-0.5), Point(0.5,0.5), 2**l, 2**l) for l in range(1, n_levels+1)]

    outer_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in outer_meshes]
    inner_spaces = [FunctionSpace(mesh, 'CG', 1) for mesh in inner_meshes]

    matern_field = MaternField(inner_spaces, outer_spaces = outer_spaces, parameters = matern_parameters, nested_inner_outer = True, nested_hierarchy = False)
    mlmc_sampler = MLMCSampler(inner_spaces, stochastic_field = matern_field, output_functional = TestFunctional(), richardson_alpha = None)

    Eps = [5.0e-4/2**i for i in range(7)]

    #run
