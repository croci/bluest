import numpy as np
from pyevtk.hl import pointsToVTK, gridToVTK

def vtk_export(input_filename, output_filename):
    a = dict(np.load(input_filename, allow_pickle=True))
    v = a['v']
    vfn = a['vfn']
    t = a['t']
    nt = len(t)
    nx = len(v)//nt
    x = np.linspace(0,max(t)//10,nx)
    v = v.reshape((nt,nx)); vfn = vfn.reshape((nt,nx))
    v = v[...,np.newaxis];  vfn = vfn[...,np.newaxis]
    z = np.array([0.0])
    coords = [item for item in np.meshgrid(x,t,z)]

    x,y,z = coords
    #pointsToVTK(output_filename, x,y,z, data=v)
    gridToVTK(output_filename, x,y,z, pointData={'voltage' : v, 'voltage_FN' : vfn})

if __name__ == '__main__':
    input_filename = './results/voltage_data.npz'
    output_filename = './results/voltage_plot'
    vtk_export(input_filename, output_filename)
