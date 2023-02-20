import numpy as np
from numpy import ndarray


def SCrandnc(cut_off: float = 2, shape: tuple = (1, )) -> ndarray:
    """
    Generates an array of random number(s) from normal distribution with a cut-off.

    Parameters
    ----------
    cut_off : float
        The cut-off value.
    shape : tuple
        The shape of the output array.

    Returns
    -------
    out : ndarray
        The output array.
    """
    out_shape = (1,) if np.sum(shape) < 1 else shape
    out = np.random.randn(np.prod(out_shape))
    outindex = np.abs(out) > np.abs(cut_off)
    while np.sum(outindex):
        out[outindex] = np.random.randn(np.sum(outindex))
        outindex = np.abs(out) > np.abs(cut_off)
    return out.reshape(out_shape)
