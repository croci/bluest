from mpi4py import MPI
from mpi4py.MPI import COMM_SELF,COMM_WORLD
import numpy as np

selfComm = COMM_SELF
worldComm = COMM_WORLD

rank = selfComm.Get_rank()
size = selfComm.Get_size()

print(size, rank, flush=True)

worldComm.barrier()

rank = worldComm.Get_rank()
size = worldComm.Get_size()

print(size, rank, flush=True)

a = np.arange(rank*9, (rank+1)*9).reshape((3,3))

print(rank, a)

receive = np.zeros_like(a)
receive = worldComm.allreduce(a, op=MPI.SUM)

print(rank, receive)
