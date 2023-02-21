import numpy as np


def SCparticlesIn3D(R, NPART):  # TODO temporarily kept for reference
    # If the output of lattice_pass is 4D array it may be good to keep it due to ease of resolving elements and turns
    assert R.shape[1] == NPART # Most likely NPART is no longer needed
    return np.transpose(R.reshape(R.shape[0], R.shape[1], R.shape[2]*R.shape[3]), axes=(0, 2, 1))
    #  should give the same result
    #  return np.transpose(R, axes=(0, 2, 3, 1)).reshape(R.shape[0], R.shape[2]*R.shape[3], R.shape[1])

    # Stay with at output which is 4D