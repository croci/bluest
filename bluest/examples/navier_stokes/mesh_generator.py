import pygmsh
import gmsh
import meshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

def generate_NS_mesh(Nc1, Nc2, N):

    filestring = "./meshes/NS_%d_%d_%d" % (N, Nc1, Nc2)

    cc1 = [0.2, 0.2, 0]
    cc2 = [1.0, 0.2, 0]
    r1 = 0.05
    r2 = 0.08
    L = 2.2
    W = 0.41

    geom = pygmsh.geo.Geometry()
    model = geom.__enter__()

    c1 = model.add_circle(cc1, r1, mesh_size=1/Nc1)
    c2 = model.add_circle(cc2, r2, mesh_size=1/Nc2)

    # Add points with finer resolution on left side
    points = [model.add_point((0,         0, 0), mesh_size=1./N),
              #model.add_point((cc1[0]-r1, 0, 0), mesh_size=1./Nc1),
              ##model.add_point((cc1[0],    0, 0), mesh_size=1./Nc1),
              ##model.add_point((cc1[0]+r1, 0, 0), mesh_size=1./Nc1),
              #model.add_point((cc2[0]-r2, 0, 0), mesh_size=1./Nc2),
              ##model.add_point((cc2[0],    0, 0), mesh_size=1./Nc2),
              ##model.add_point((cc2[0]+r2, 0, 0), mesh_size=1./Nc2),
              model.add_point((L,         0, 0), mesh_size=1./N),
              model.add_point((L,         W, 0), mesh_size=1./N),
              ##model.add_point((cc2[0]+r2, W, 0), mesh_size=1./Nc2),
              ##model.add_point((cc2[0],    W, 0), mesh_size=1./Nc2),
              #model.add_point((cc2[0]-r2, W, 0), mesh_size=1./Nc2),
              ##model.add_point((cc1[0]+r1, W, 0), mesh_size=1./Nc1),
              ##model.add_point((cc1[0],    W, 0), mesh_size=1./Nc1),
              #model.add_point((cc1[0]-r1, W, 0), mesh_size=1./Nc1),
              model.add_point((0,         W, 0), mesh_size=1./N)]

    # Add lines between all points creating the rectangle
    channel_lines = [model.add_line(points[i], points[i+1]) for i in range(-1, len(points)-1)]

    # Create a line loop and plane surface for meshing
    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop, holes=[c1.curve_loop, c2.curve_loop])

    # Call gmsh kernel before add physical entities
    model.synchronize()

    volume_marker = 6
    model.add_physical([plane_surface], "Volume")
    model.add_physical([channel_lines[0]], "Inflow")
    model.add_physical([channel_lines[2]], "Outflow")
    model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
    model.add_physical(c1.curve_loop.curves, "Obstacle1")
    model.add_physical(c2.curve_loop.curves, "Obstacle2")

    #geom.add_rectangle(0,L,0,W,0, mesh_size=1./N, holes=[c1,c2])
    mesh = geom.generate_mesh(dim=2)
    gmsh.write(filestring + ".msh")
    gmsh.clear()
    geom.__exit__()

    mesh_from_file = meshio.read(filestring + ".msh")

    #line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    #meshio.write(filestring + "_facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write(filestring + ".xdmf", triangle_mesh)

    return filestring + ".xdmf"

if __name__ == "__main__":
    Nc1 = 16; Nc2 = 64; N = 32

    import dolfin as do

    comm = do.MPI.comm_world
    mpiRank = do.MPI.rank(comm)
    mpiSize = do.MPI.size(comm)

    if mpiRank == 0:
        filestring = generate_NS_mesh(Nc1,Nc2,N)
    else:
        filestring = None

    filestring = comm.bcast(filestring, root=0)

    dmesh = do.Mesh(do.MPI.comm_self)

    with do.XDMFFile(dmesh.mpi_comm(), filestring) as f:
        f.read(dmesh)
