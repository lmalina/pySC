import numpy as np


def SCrandnc(c, shape=(1, 1)):
    if np.sum(shape) < 1:
        shape = (1,)
    out = np.random.randn(*shape)
    outindex = np.where(np.abs(out) > np.abs(c))
    while np.sum(outindex):
        out[outindex] = np.random.randn(np.sum(outindex))
        outindex = np.where(np.abs(out) > np.abs(c))
    return out
