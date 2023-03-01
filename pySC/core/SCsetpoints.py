import numpy as np

from pySC.core.SCregisterUpdate import SCupdateCAVs, SCupdateMagnets
from typing import Tuple
from numpy import ndarray
from pySC.classes import SimulatedComissioning
VALID_METHODS = ("abs", "rel", "add")


def SCsetCavs2SetPoints(SC: SimulatedComissioning, CAVords: ndarray, type: str, setpoints: ndarray, method: str ='abs') -> SimulatedComissioning:
    setpoints = _check_input_and_setpoints(method, CAVords, setpoints)
    setpoint_str = f"{type}SetPoint"
    for i, ord in enumerate(CAVords):
        new_setpoint = setpoints[i]
        if method == 'rel':
            new_setpoint *= getattr(SC.RING[ord], setpoint_str)
        if method == 'add':
            new_setpoint += getattr(SC.RING[ord], setpoint_str)
        setattr(SC.RING[ord], setpoint_str, new_setpoint)
    SC = SCupdateCAVs(SC, CAVords)
    return SC


def SCsetCMs2SetPoints(SC: SimulatedComissioning, CMords: ndarray, setpoints: ndarray, nDim: int, method: str = 'abs') -> Tuple[SimulatedComissioning, ndarray]:
    # TODO correct accessing SC.RING.attr.subattr/elements
    setpoints = _check_input_and_setpoints(method, CMords, setpoints)
    for i, ord in enumerate(CMords):
        curAB = np.array([SC.RING[ord].SetPointB, SC.RING[ord].SetPointA]).T
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            normBy = np.array([1, 1])
        else:
            normBy = np.array([-1, 1]) * SC.RING[ord].Length  # positive setpoint -> positive kick -> negative horizontal field
        if method == 'rel':
            setpoints[i] *= curAB[0, nDim] * normBy[nDim]
        if method == 'add':
            setpoints[i] += curAB[0, nDim] * normBy[nDim]

        if hasattr(SC.RING[ord], 'CMlimit') and abs(setpoints[i]) > abs(SC.RING[ord].CMlimit[nDim]):
            print(f'CM (ord: {ord} / dim: {nDim}) is clipping')
            setpoints[i] = np.sign(setpoints[i]) * SC.RING[ord].CMlimit[nDim]
        if nDim == 1:
            SC.RING[ord].SetPointB[0] = setpoints[i] / normBy[nDim]
        else:
            SC.RING[ord].SetPointA[0] = setpoints[i] / normBy[nDim]
    SC = SCupdateMagnets(SC, CMords)
    return SC, setpoints


def SCsetMags2SetPoints(SC: SimulatedComissioning, MAGords: ndarray, type: int, order: int, setpoints: ndarray, method: str = 'abs', dipCompensation: bool = False) -> SimulatedComissioning:
    # TODO type to 0 and 1
    setpoints = _check_input_and_setpoints(method, MAGords, setpoints)
    for i, ord in enumerate(MAGords):
        nomAB = np.array([SC.RING[ord].NomPolynomA, SC.RING[ord].NomPolynomB])
        curAB = np.array([SC.RING[ord].SetPointA, SC.RING[ord].SetPointB])
        if method == 'rel':
            setpoints[i] *= nomAB[order, type]
        if method == 'add':
            setpoints[i] += curAB[order, type]

        setpoints[i] = _checkClipping(SC, ord, type, order, setpoints[i])
        # TODO should check CF magnets
        if dipCompensation and order == 2:  # quad
            SC = _dipCompensation(SC, ord, setpoints[i])
        if type == 1:
            SC.RING[ord].SetPointA[order] = setpoints[i]
        else:
            SC.RING[ord].SetPointB[order] = setpoints[i]
        SC = SCupdateMagnets(SC, np.array([ord]))
    return SC


def _dipCompensation(SC, ord, setpoint):
    if not (hasattr(SC.RING[ord], 'BendingAngle') and SC.RING[ord].BendingAngle != 0 and ord in SC.ORD.CM[0]):
        return SC
    idealKickDifference = ((setpoint - (SC.RING[ord].SetPointB[2] - SC.RING[ord].NomPolynomB[2])) /
                           SC.RING[ord].NomPolynomB[2] - 1) * SC.RING[ord].BendingAngle / SC.RING[ord].Length
    SC, _ = SCsetCMs2SetPoints(SC, ord, idealKickDifference * SC.RING[ord].Length, 1, method='add')
    return SC


def _checkClipping(SC, ord, type, order, setpoint):
    if not (type == 1 and order == 2): # not a skew quad
        return setpoint
    if hasattr(SC.RING[ord], 'SkewQuadLimit') and abs(setpoint) > abs(SC.RING[ord].SkewQuadLimit):
        print('SC:SkewLim', 'Skew quadrupole (ord: %d) is clipping' % ord)
        setpoint = np.sign(setpoint) * SC.RING[ord].SkewQuadLimit
    return setpoint


def _check_input_and_setpoints(method, ords, setpoints):
    if method not in VALID_METHODS:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {VALID_METHODS}.')
    if len(setpoints) not in (1, len(ords)):
        raise ValueError(f'Setpoints have to have length of 1 or matching to the length or ordinates.')
    if len(setpoints) == 1:
        return np.repeat(setpoints, len(ords))
    return setpoints