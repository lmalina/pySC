import numpy as np


def SCrandnc(c, m=1, n=1, normalize=False):  # TODO a case with second argument [m,n]
    out = np.random.randn(m, n)
    outindex = np.where(np.abs(out) > np.abs(c))
    while len(outindex[0]) > 0:
        out[outindex] = np.random.randn(len(outindex[0]))
        outindex = np.where(np.abs(out) > np.abs(c))
    if normalize:
        print('Not yet implemented.')
    return out
