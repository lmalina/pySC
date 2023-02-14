import numpy as np


def SCupdateMagnets(SC, ords=None):
    if ords is None:
        ords = SC.ORD.Magnet
    for ord in ords:
        SC = updateMagnets(SC, ord, ord)
        if hasattr(SC.RING[ord], 'MasterOf'):
            for childOrd in SC.RING[ord].MasterOf:
                SC = updateMagnets(SC, ord, childOrd)
    return SC


def updateMagnets(SC, source, target):
    SC.RING[target].PolynomB = SC.RING[source].SetPointB * addPadded(np.ones(len(SC.RING[source].SetPointB)),
                                                                     SC.RING[source].CalErrorB)
    SC.RING[target].PolynomA = SC.RING[source].SetPointA * addPadded(np.ones(len(SC.RING[source].SetPointA)),
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
            sysPolynomA[n + 1] = addPadded(sysPolynomA[n + 1], sysPolynomA[n])
        SC.RING[target].PolynomA = addPadded(SC.RING[target].PolynomA, sysPolynomA[-1])
    if len(sysPolynomB) > 0:
        for n in range(len(sysPolynomB) - 1):
            sysPolynomB[n + 1] = addPadded(sysPolynomB[n + 1], sysPolynomB[n])
        SC.RING[target].PolynomB = addPadded(SC.RING[target].PolynomB, sysPolynomB[-1])
    if hasattr(SC.RING[target], 'PolynomBOffset'):
        SC.RING[target].PolynomB = addPadded(SC.RING[target].PolynomB, SC.RING[target].PolynomBOffset)
        SC.RING[target].PolynomA = addPadded(SC.RING[target].PolynomA, SC.RING[target].PolynomAOffset)
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


def addPadded(v1, v2):  # TODO this is probably wrong
    if not ((v1.ndim == 1 and v2.ndim == 1) or (v1.ndim == 2 and v2.ndim == 2)):
        raise ValueError('Wrong dimensions.')
    l1 = len(v1)
    l2 = len(v2)
    if l2 > l1: v1[l2 - 1] = 0
    if l1 > l2: v2[l1 - 1] = 0
    return v1 + v2
