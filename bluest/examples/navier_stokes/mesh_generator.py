import pygmsh
import meshio
import sys

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

    with pygmsh.geo.Geometry() as geom:
        c1 = geom.add_circle(cc1, r1, mesh_size=1/Nc1)
        c2 = geom.add_circle(cc2, r2, mesh_size=1/Nc2)

        # Add points with finer resolution on left side
        points = [geom.add_point((0,         0, 0), mesh_size=1./N),
                  #geom.add_point((cc1[0]-r1, 0, 0), mesh_size=1./Nc1),
                  ##geom.add_point((cc1[0],    0, 0), mesh_size=1./Nc1),
                  ##geom.add_point((cc1[0]+r1, 0, 0), mesh_size=1./Nc1),
                  #geom.add_point((cc2[0]-r2, 0, 0), mesh_size=1./Nc2),
                  ##geom.add_point((cc2[0],    0, 0), mesh_size=1./Nc2),
                  ##geom.add_point((cc2[0]+r2, 0, 0), mesh_size=1./Nc2),
                  geom.add_point((L,         0, 0), mesh_size=1./N),
                  geom.add_point((L,         W, 0), mesh_size=1./N),
                  ##geom.add_point((cc2[0]+r2, W, 0), mesh_size=1./Nc2),
                  ##geom.add_point((cc2[0],    W, 0), mesh_size=1./Nc2),
                  #geom.add_point((cc2[0]-r2, W, 0), mesh_size=1./Nc2),
                  ##geom.add_point((cc1[0]+r1, W, 0), mesh_size=1./Nc1),
                  ##geom.add_point((cc1[0],    W, 0), mesh_size=1./Nc1),
                  #geom.add_point((cc1[0]-r1, W, 0), mesh_size=1./Nc1),
                  geom.add_point((0,         W, 0), mesh_size=1./N)]

        # Add lines between all points creating the rectangle
        channel_lines = [geom.add_line(points[i], points[i+1]) for i in range(-1, len(points)-1)]

        # Create a line loop and plane surface for meshing
        channel_loop = geom.add_curve_loop(channel_lines)
        plane_surface = geom.add_plane_surface(channel_loop, holes=[c1.curve_loop, c2.curve_loop])

        # Call gmsh kernel before add physical entities
        geom.synchronize()

        volume_marker = 6
        geom.add_physical([plane_surface], "Volume")
        geom.add_physical([channel_lines[0]], "Inflow")
        geom.add_physical([channel_lines[2]], "Outflow")
        geom.add_physical([channel_lines[1], channel_lines[3]], "Walls")
        geom.add_physical(c1.curve_loop.curves, "Obstacle1")
        geom.add_physical(c2.curve_loop.curves, "Obstacle2")

        mesh = geom.generate_mesh(dim=2)

        pygmsh.write(filestring + ".msh")

    mesh_from_file = meshio.read(filestring + ".msh")

    #line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    #meshio.write(filestring + "_facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write(filestring + ".xdmf", triangle_mesh)

    return filestring + ".xdmf"

def MPI_generate_NS_mesh(Nc1, Nc2, N):
    from mpi4py.MPI import COMM_WORLD, COMM_SELF
    mpi_comm = COMM_WORLD
    mpiRank = mpi_comm.Get_rank()
    mpiSize = mpi_comm.Get_size()
    if mpiSize > 1:
        if mpiRank == 0:
            comm = COMM_SELF.Spawn(sys.executable, args=['mesh_generator.py', str(Nc1), str(Nc2), str(N), str(True)], maxprocs=1)
            comm.Disconnect()

        mpi_comm.barrier()
    else:
        generate_NS_mesh(Nc1, Nc2, N)

    filestring = "./meshes/NS_%d_%d_%d.xdmf" % (N, Nc1, Nc2)
    return filestring

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("Nc1", action="store", nargs='?', default=16, type=int)
    parser.add_argument("Nc2", action="store", nargs='?', default=64, type=int)
    parser.add_argument("N",   action="store", nargs='?', default=32, type=int)
    parser.add_argument("ischild", action="store", nargs='?', default=False, type=bool)

    args = parser.parse_args()

    default = len(sys.argv) < 4

    Nc1 = args.Nc1
    Nc2 = args.Nc2
    N   = args.N
    ischild = args.ischild

    if not default:
        generate_NS_mesh(Nc1, Nc2, N)
        if ischild:
            from mpi4py import MPI
            comm = MPI.Comm.Get_parent()
            comm.Disconnect()

    else:
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
