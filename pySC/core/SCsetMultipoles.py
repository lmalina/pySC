import numpy as np
from pySC.core.SCrandnc import SCrandnc


def SCsetMultipoles(RING, ords, AB, method='rnd', order=[], type=[]):
    if method == 'sys':
        for ord in ords:
            RING = applySysMultipoles(RING, ord, AB, order, type)
    elif method == 'rnd':
        for ord in ords:
            addAB = SCrandnc(2, AB.shape) * AB
            RING = applyRndMultipoles(RING, ord, addAB)
    else:
        raise ValueError('Unsupported multipole method. Allowed are ''sys'' or ''rnd''.')
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
    if 'PolynomAOffset' in RING[ord]:
        RING[ord]['PolynomAOffset'] = addPadded(RING[ord]['PolynomAOffset'], AB[:, 0])
    else:
        RING[ord]['PolynomAOffset'] = AB[:, 0]
    if 'PolynomBOffset' in RING[ord]:
        RING[ord]['PolynomBOffset'] = addPadded(RING[ord]['PolynomBOffset'], AB[:, 1])
    else:
        RING[ord]['PolynomBOffset'] = AB[:, 1]
    return RING

def addPadded(v1, v2):
    if not ((v1.ndim == 1 and v2.ndim == 1) or (v1.ndim == 2 and v2.ndim == 2)):
        raise ValueError('Wrong dimensions.')
    l1 = v1.shape[0]
    l2 = v2.shape[0]
    if l2 > l1:
        v1 = np.pad(v1, (0, l2 - l1), 'constant')
    if l1 > l2:
        v2 = np.pad(v2, (0, l1 - l2), 'constant')
    return v1 + v2
