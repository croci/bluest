import sys
import numpy as np
from mpi4py import MPI

def parallel_fun():
    comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args = ['child.py'],
        maxprocs=4)

    N = np.array(0, dtype='i')

    comm.Reduce(None, [N, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

    print(f'We got the magic number {N}')
