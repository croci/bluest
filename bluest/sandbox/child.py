from mpi4py import MPI
import numpy as np


comm = MPI.Comm.Get_parent()

print(f'Hi from {comm.Get_rank()}/{comm.Get_size()}')
A = np.random.randn(3000,3000); A = A.T@A
N = np.array(comm.Get_rank(), dtype='i')

comm.Reduce([N, MPI.INT], None, op=MPI.SUM, root=0)
