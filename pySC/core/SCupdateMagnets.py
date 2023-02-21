import numpy as np


def SCupdateMagnets(SC, ords=None):
    for ord in (SC.ORD.Magnet if ords is None else ords):
        SC = _updateMagnets(SC, ord, ord)
        if hasattr(SC.RING[ord], 'MasterOf'):
            for childOrd in SC.RING[ord].MasterOf:
                SC = _updateMagnets(SC, ord, childOrd)
    return SC


def _updateMagnets(SC, source, target):  # TODO simplify AB calculated in place
    SC.RING[target].PolynomB = SC.RING[source].SetPointB * add_padded(np.ones(len(SC.RING[source].SetPointB)),
                                                                      SC.RING[source].CalErrorB)
    SC.RING[target].PolynomA = SC.RING[source].SetPointA * add_padded(np.ones(len(SC.RING[source].SetPointA)),
                                                                      SC.RING[source].CalErrorA)
    sysPolynomB = []
    sysPolynomA = []
    if hasattr(SC.RING[target], 'SysPolBFromB'):
        for n in range(len(SC.RING[target].SysPolBFromB)):
            if SC.RING[target].SysPolBFromB[n] is not None:
                sysPolynomB.append(SC.RING[target].PolynomB[n] * SC.RING[target].SysPolBFromB[n])
    if hasattr(SC.RING[target], 'SysPolBFromA'):
        for n in range(len(SC.RING[target].SysPolBFromA)):
            if SC.RING[target].SysPolBFromA[n] is not None:
                sysPolynomB.append(SC.RING[target].PolynomA[n] * SC.RING[target].SysPolBFromA[n])
    if hasattr(SC.RING[target], 'SysPolAFromB'):
        for n in range(len(SC.RING[target].SysPolAFromB)):
            if SC.RING[target].SysPolAFromB[n] is not None:
                sysPolynomA.append(SC.RING[target].PolynomB[n] * SC.RING[target].SysPolAFromB[n])
    if hasattr(SC.RING[target], 'SysPolAFromA'):
        for n in range(len(SC.RING[target].SysPolAFromA)):
            if SC.RING[target].SysPolAFromA[n] is not None:
                sysPolynomA.append(SC.RING[target].PolynomA[n] * SC.RING[target].SysPolAFromA[n])
    if len(sysPolynomA) > 0:
        for n in range(len(sysPolynomA) - 1):
            sysPolynomA[n + 1] = add_padded(sysPolynomA[n + 1], sysPolynomA[n])
        SC.RING[target].PolynomA = add_padded(SC.RING[target].PolynomA, sysPolynomA[-1])
    if len(sysPolynomB) > 0:
        for n in range(len(sysPolynomB) - 1):
            sysPolynomB[n + 1] = add_padded(sysPolynomB[n + 1], sysPolynomB[n])
        SC.RING[target].PolynomB = add_padded(SC.RING[target].PolynomB, sysPolynomB[-1])
    if hasattr(SC.RING[target], 'PolynomBOffset'):
        SC.RING[target].PolynomB = add_padded(SC.RING[target].PolynomB, SC.RING[target].PolynomBOffset)
        SC.RING[target].PolynomA = add_padded(SC.RING[target].PolynomA, SC.RING[target].PolynomAOffset)
    if hasattr(SC.RING[source], 'BendingAngleError'):
        SC.RING[target].PolynomB[0] = SC.RING[target].PolynomB[0] + SC.RING[source].BendingAngleError * SC.RING[
            target].BendingAngle / SC.RING[target].Length
    if hasattr(SC.RING[source], 'BendingAngle'):
        if hasattr(SC.RING[source], 'CombinedFunction') and SC.RING[source].CombinedFunction == 1:
            alpha_act = SC.RING[source].SetPointB[1] * (1 + SC.RING[source].CalErrorB[1]) / SC.RING[source].NomPolynomB[
                1]
            effBendingAngle = alpha_act * SC.RING[target].BendingAngle
            SC.RING[target].PolynomB[0] = SC.RING[target].PolynomB[0] + (
                    effBendingAngle - SC.RING[target].BendingAngle) / SC.RING[target].Length
    if SC.RING[source].PassMethod == 'CorrectorPass':
        SC.RING[target].KickAngle[0] = SC.RING[target].PolynomB[0]
        SC.RING[target].KickAngle[1] = SC.RING[target].PolynomA[0]
    SC.RING[target].MaxOrder = len(SC.RING[target].PolynomB) - 1
    return SC


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
