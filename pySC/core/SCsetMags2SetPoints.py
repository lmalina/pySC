import numpy as np

from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints
from pySC.core.SCupdateMagnets import SCupdateMagnets


def SCsetMags2SetPoints(SC, MAGords, type, order, setpoints, method='abs', dipCompensation=False):
    # TODO correct accessing SC.RING.attr.subattr/elements
    valid_methods = ("abs", "rel", "add")
    if method not in valid_methods:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {valid_methods}.')
    if len(setpoints) == 1:
        setpoints = np.repeat(setpoints, len(MAGords))
    for i, ord in enumerate(MAGords):
        nomAB = np.array([SC.RING[ord].NomPolynomA, SC.RING[ord].NomPolynomB])
        curAB = np.array([SC.RING[ord].SetPointA, SC.RING[ord].SetPointB])
        if method == 'abs':
            setpoints[i] = setpoints[i]
        elif method == 'rel':
            setpoints[i] = setpoints[i] * nomAB[order, type]
        elif method == 'add':
            setpoints[i] = setpoints[i] + curAB[order, type]
        setpoints[i] = _checkClipping(SC, ord, type, order, setpoints[i])
        if dipCompensation and order == 2:
            SC = _dipCompensation(SC, ord, setpoints[i])
        if type == 1:
            SC.RING[ord].SetPointA[order] = setpoints[i]
        else:
            SC.RING[ord].SetPointB[order] = setpoints[i]
        SC = SCupdateMagnets(SC, ord)
    return SC


def _dipCompensation(SC, ord, setpoint):
    if not (hasattr(SC.RING[ord], 'BendingAngle') and SC.RING[ord].BendingAngle != 0 and ord in SC.ORD.CM[0]):
        return SC
    idealKickDifference = ((setpoint - (SC.RING[ord].SetPointB[2] - SC.RING[ord].NomPolynomB[2])) /
                           SC.RING[ord].NomPolynomB[2] - 1) * SC.RING[ord].BendingAngle / SC.RING[ord].Length
    SC, _ = SCsetCMs2SetPoints(SC, ord, idealKickDifference * SC.RING[ord].Length, 1, method='add')
    return SC


def _checkClipping(SC, ord, type, order, setpoint):
    if not (type == 1 and order == 2):
        return setpoint
    if hasattr(SC.RING[ord], 'SkewQuadLimit') and abs(setpoint) > abs(SC.RING[ord].SkewQuadLimit):
        print('SC:SkewLim', 'Skew quadrupole (ord: %d) is clipping' % ord)
        setpoint = np.sign(setpoint) * SC.RING[ord].SkewQuadLimit
    return setpoint
