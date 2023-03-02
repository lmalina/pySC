import numpy as np
from pySC.utils.sc_tools import SCrandnc


def SCsetMultipoles(RING, ords, AB, method='rnd', order=[], type=[]):  # TODO nowhere used? calculate AB in place
    allowed_methods = ("sys", "rnd")
    if method not in allowed_methods:
        raise ValueError(f'Unsupported multipole method {method}. Allowed are {allowed_methods}.')
    if method == 'sys':
        for ord in ords:
            RING = applySysMultipoles(RING, ord, AB, order, type)
    else:  # method = 'rnd'
        for ord in ords:
            addAB = SCrandnc(2, AB.shape) * AB
            RING = applyRndMultipoles(RING, ord, addAB)
    return RING


def applySysMultipoles(RING, ord, AB, order, type):
    AB[order, type] = 0
    if type == 1:
        RING[ord]['SysPolAFromA'][order] = AB[:, 0]
        RING[ord]['SysPolBFromA'][order] = AB[:, 1]
    else:
        RING[ord]['SysPolAFromB'][order] = AB[:, 0]
        RING[ord]['SysPolBFromB'][order] = AB[:, 1]
    return RING


def applyRndMultipoles(RING, ord, AB):
    if hasattr(RING[ord], 'PolynomAOffset'):
        RING[ord]['PolynomAOffset'] = add_padded(RING[ord]['PolynomAOffset'], AB[:, 0])
    else:
        RING[ord]['PolynomAOffset'] = AB[:, 0]
    if hasattr(RING[ord], 'PolynomBOffset'):
        RING[ord]['PolynomBOffset'] = add_padded(RING[ord]['PolynomBOffset'], AB[:, 1])
    else:
        RING[ord]['PolynomBOffset'] = AB[:, 1]
    return RING


def add_padded(v1, v2):
    if v1.ndim != v2.ndim:
        raise ValueError(f'Unmatched number of dimensions {v1.ndim} and {v2.ndim}.')
    max_dims = np.array([max(d1, d2) for d1, d2 in zip(v1.shape, v2.shape)])
    if np.sum(max_dims > 1) > 1:
        raise ValueError(f'Wrong or mismatching dimensions {v1.shape} and {v2.shape}.')
    vsum = np.zeros(np.prod(max_dims))
    vsum[:np.max(v1.shape)] += v1
    vsum[:np.max(v2.shape)] += v2
    return vsum
