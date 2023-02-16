import numpy as np


def SCparticlesIn3D(R, NPART):
    nZ = R.shape[0]
    NELEM = R.shape[1] / NPART
    M = np.transpose(np.reshape(R, (nZ, NPART, NELEM)), (0, 2, 1))
    return M
