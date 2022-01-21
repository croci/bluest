from dolfin import *
import pygmsh
import sys
from mesh_generator import MPI_generate_NS_mesh

worldComm = MPI.comm_world
mpiRank = worldComm.Get_rank()

Nc1 = 16; Nc2 = 16; N = 32

#if mpiRank == 0:
#    comm = MPI.comm_self.Spawn(sys.executable, args=['mesh_generator.py', str(Nc1), str(Nc2), str(N), str(True)], maxprocs=1)
#    comm.Disconnect()
#
#MPI.comm_world.barrier()
#
#filestring = "./meshes/NS_%d_%d_%d.xdmf" % (N, Nc1, Nc2)

filestring = MPI_generate_NS_mesh(Nc1, Nc2, N)

mesh = Mesh(MPI.comm_self)
with XDMFFile(mesh.mpi_comm(), filestring) as f:
    f.read(mesh)

print("Rank: ", mpiRank, "Done!")
