from dolfin import *
from petsc4py import PETSc
import numpy as np

from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.spatial import cKDTree

def gammaratio(x,y): # computes the ratio between \Gamma(x) and \Gamma(y)
    return np.exp(gammaln(x) - gammaln(y))

def make_nested_mapping(outer_space, inner_space):
    # Maps the dofs between nested meshes
    outer_dof_coor = outer_space.tabulate_dof_coordinates()
    inner_dof_coor = inner_space.tabulate_dof_coordinates()

    tree = cKDTree(outer_dof_coor)
    _,mapping = tree.query(inner_dof_coor, k=1)
    return mapping

RNG = np.random.RandomState(1234567890)

class WhiteNoiseField(object):
    def __init__(self, V, RNG=RNG):
        self.RNG = RNG

        mesh = V.mesh()
        u = TrialFunction(V)
        v = TestFunction(V)

        n_cells = mesh.num_cells()
        eldim = V.element().space_dimension()

        self.V = V
        self.eldim = eldim
        self.n_cells = n_cells

        self.cell_volumes = np.sqrt(np.array([c.volume() for c in cells(mesh)]))
        c = Cell(mesh,0)
        self.H = np.linalg.cholesky(assemble_local(u*v*dx,c)/c.volume())

        dofmap = V.dofmap()
        n = eldim*n_cells
        cell_to_global_dofs = np.concatenate([dofmap.cell_dofs(i) for i in range(n_cells)])
        self.L = csr_matrix((np.ones((n,),dtype='int8'), (cell_to_global_dofs, np.arange(n))), shape = (V.dim(), n), dtype='int8')

    def sample(self):
        out = Function(self.V)
        r = self.RNG.randn(self.eldim, self.n_cells)
        z = (self.H@r)*self.cell_volumes
        out.vector()[:] = self.L@(z.T.flatten())
        return out

class MaternField(object):
    def __init__(self, V, inner_V, parameters, RNG=RNG):
        self.V = V
        self.inner_V = inner_V
        self.dim = V.mesh().geometry().dim()
        self.n_solves = int(np.round(parameters["nu"]/2. + self.dim/4.))

        self.WN = WhiteNoiseField(V,RNG)

        self.parameters = self.compute_scalings(parameters)
        self.ksp = self.solver_setup()
        self.mapping = make_nested_mapping(V, inner_V)

    def compute_scalings(self, params):
        # constant in the self-adjoint elliptic equation
        params["const"] = (np.sqrt(8.*params["nu"])/params['lmbda'])**-2
        # this correction gives an output of unit variance
        if self.dim != 2:
            sigma_correction = np.sqrt(gammaratio(params["nu"], params["nu"] + self.dim/2.)*params["nu"]**(self.dim/2.))*(2./np.pi)**(self.dim/4.)*params['lmbda']**(-self.dim/2.)
        else:
            sigma_correction = np.sqrt(2./np.pi)/params['lmbda']

        # scale std. dev and mean if lognormal scaling used
        avg   = params["avg"]
        sigma = params["sigma"]
        if params["lognormal_scaling"] == True:
            sigma_factor = np.sqrt(np.log(1. + (sigma/avg)**2.))
            mu = np.log(avg**2./np.sqrt(sigma**2. + avg**2.0))
        else:
            sigma_factor = sigma
            mu           = avg

        params["scaling"] = sigma_factor/sigma_correction
        params["mu"] = mu

        return params

    def solver_setup(self):
        prefix = "matern_"
        opts   = {"ksp_type" : "cg",
                  "ksp_atol" : 1.0e-10,
                  "ksp_rtol" : 1.0e-12,
                  "ksp_norm_type" : "unpreconditioned",
                  "ksp_diagonal_scale" : True,
                  "ksp_diagonal_scale_fix" : True,
                  "ksp_reuse_preconditioner" : True,
                  "ksp_max_it" : 1000,
                  "pc_factor_mat_solver_type" : "mumps",

                  #"ksp_converged_reason" : None,
                  #"ksp_monitor_true_residual" : None,
                  #"ksp_view" : None,

                  "pc_type" : "hypre",
                  "pc_hypre_boomeramg_strong_threshold" : [0.25, 0.25, 0.6][self.dim-1],
                  "pc_hypre_type" : "boomeramg",

                  }

        petsc_opts = PETSc.Options() 
        for key in opts:
            petsc_opts[prefix + key] = opts[key]

        V = self.V

        u = TrialFunction(V)
        v = TestFunction(V)

        bc = DirichletBC(V, Constant(0.0), "on_boundary")
        L = Constant(self.parameters["const"])*inner(grad(u), grad(v))*dx + u*v*dx + Constant(0.0)*v*dx
        L = as_backend_type(assemble_system(*system(L), bcs=bc)[0]).mat()

        ksp = PETSc.KSP().create(V.mesh().mpi_comm())
        ksp.setOperators(L)
        ksp.setOptionsPrefix(prefix)
        ksp.setFromOptions()
        return ksp

    def sample(self):
        parameters = self.parameters
        v = TestFunction(self.V)

        out = Function(self.V)
        b = self.WN.sample()
        b = as_backend_type(b.vector()).vec()
        x = b.duplicate()
        for i in range(self.n_solves):
            self.ksp.solve(b, x)
            reason = self.ksp.getConvergedReason()
            if reason < 0:
                resnorm = self.ksp.getResidualNorm()
                raise RuntimeError("ERROR in the Matern field sampler! Linear KSP solver did not converge with reason: %d. Residual norm: %f" % (reason, resnorm))

            if i < self.n_solves-1:
                out.vector()[:] = x.getArray()
                b = as_backend_type(assemble(out*v*dx)).vec()

        out.vector()[:] = parameters["scaling"]*x.getArray() + parameters["mu"]

        # nested interpolation
        inner_out = Function(self.inner_V)
        inner_out.vector()[:] = out.vector()[self.mapping]
        return inner_out

if __name__ == "__main__":
    # Matern field parameters
    # the Matern field smoothness parameter is give by 2*k - dim/2. Here k must be an integer
    # IMPORTANT: to sample the field the algorithm solves k linear systems so k must be kept small
    # IMPORTANT: for good accuracy the distance between the boundaries of the inner and outer domain (see below) must be at least lmbda and lmbda must be larger than the mesh size
    k = 1
    dim = 2 # geometric dimension (1D, 2D, 3D)
    parameters = {"lmbda"    : 0.2, # correlation length 
                  "avg"      : 1.0, # mean
                  "sigma"    : 0.2, # standard dev.
                  "lognormal_scaling" : False,
                  "nu"       : 2*k-dim/2} # smoothness parameter

    l = 7
    # NOTE: the way the algorithm works is that it samples the Matern field by solving an SPDE on an outer domain and then
    #       transferring the result onto the domain on which we actually need the sample. This code assumes
    #       that the inner domain mesh is nested within the outer domain mesh.
    if dim == 1:
        outer_mesh = IntervalMesh(MPI.comm_self, 2**(l+1), -1, 1)
        inner_mesh = IntervalMesh(MPI.comm_self, 2**l, -0.5, 0.5)
    elif dim == 2:
        outer_mesh = RectangleMesh(MPI.comm_self, Point(-1,-1), Point(1,1), 2**(l+1), 2**(l+1)) # auxiliary mesh
        inner_mesh = RectangleMesh(MPI.comm_self, Point(-0.5,-0.5), Point(0.5,0.5), 2**l, 2**l) # mesh on which we actually need the sample
    assert outer_mesh.geometry().dim() == dim

    outer_V = FunctionSpace(outer_mesh, "CG", 1)
    inner_V = FunctionSpace(inner_mesh, "CG", 1)

    WN = WhiteNoiseField(inner_V)
    wn = WN.sample()

    matern = MaternField(outer_V, inner_V, parameters)

    m = matern.sample()
    if parameters["lognormal_scaling"]:
        m = project(exp(m), inner_V)
    print(assemble(m*dx), np.sqrt(assemble((m-parameters["avg"])**2*dx))) # should match avg and sigma for fine enough grids
