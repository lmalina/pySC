import numpy as np
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
    SC.update_cavities(CAVords)
    return SC


def SCgetCMSetPoints(SC: SimulatedComissioning, CMords: ndarray, skewness: bool) -> ndarray:
    setpoints = np.nan*np.ones(len(CMords))
    order = 0
    ndim = 1 if skewness else 0
    for i, ord in enumerate(CMords):
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            normBy = np.array([1, 1])
        else:
            normBy = np.array([-1, 1]) * SC.RING[ord].Length  # positive setpoint -> positive kick -> negative horizontal field
        if skewness:
            setpoints[i] = SC.RING[ord].SetPointA[order] * normBy[ndim]
        else:
            setpoints[i] = SC.RING[ord].SetPointB[order] * normBy[ndim]
    return setpoints

def SCsetCMs2SetPoints(SC: SimulatedComissioning, CMords: ndarray, setpoints: ndarray, skewness: bool, method: str = 'abs') -> Tuple[SimulatedComissioning, ndarray]:
    # TODO correct accessing SC.RING.attr.subattr/elements
    # TODO skewness old 2 -> True, 1 -> False
    setpoints = _check_input_and_setpoints(method, CMords, setpoints)
    order = 0
    ndim = 1 if skewness else 0
    for i, ord in enumerate(CMords):
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            normBy = np.array([1, 1])
        else:
            normBy = np.array([-1, 1]) * SC.RING[ord].Length  # positive setpoint -> positive kick -> negative horizontal field
        if method == 'rel':
            setpoints[i] *= (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * normBy[ndim]
        if method == 'add':
            setpoints[i] += (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * normBy[ndim]

        if hasattr(SC.RING[ord], 'CMlimit') and abs(setpoints[i]) > abs(SC.RING[ord].CMlimit[ndim]):
            print(f'CM (ord: {ord} / dim: {ndim}) is clipping')
            setpoints[i] = np.sign(setpoints[i]) * SC.RING[ord].CMlimit[ndim]
        if skewness:
            SC.RING[ord].SetPointA[order] = setpoints[i] / normBy[ndim]
        else:
            SC.RING[ord].SetPointB[order] = setpoints[i] / normBy[ndim]
    SC.update_magnets(CMords)
    return SC, setpoints


def SCsetMags2SetPoints(SC: SimulatedComissioning, MAGords: ndarray, skewness: bool, order: int, setpoints: ndarray, method: str = 'abs', dipCompensation: bool = False) -> SimulatedComissioning:
    # TODO skewness old 2 -> False, 1 -> True
    # TODO order decreases by 1
    setpoints = _check_input_and_setpoints(method, MAGords, setpoints)
    for i, ord in enumerate(MAGords):
        if method == 'rel':
            setpoints[i] *= SC.RING[ord].NomPolynomA[order] if skewness else SC.RING[ord].NomPolynomB[order]
        if method == 'add':
            setpoints[i] += SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]
        if skewness and order == 1:  # skew quad
            if hasattr(SC.RING[ord], 'SkewQuadLimit') and abs(setpoints[i]) > abs(SC.RING[ord].SkewQuadLimit):
                print(f'SC:SkewLim \n Skew quadrupole (ord: {ord}) is clipping')
                setpoints[i] = np.sign(setpoints[i]) * SC.RING[ord].SkewQuadLimit
        # TODO should check CF magnets
        if dipCompensation and order == 1:  # quad  # TODO check also skewness?
            SC = _dipole_compensation(SC, ord, setpoints[i])
        if skewness:
            SC.RING[ord].SetPointA[order] = setpoints[i]
        else:
            SC.RING[ord].SetPointB[order] = setpoints[i]
    SC.update_magnets(MAGords)
    return SC


def _dipole_compensation(SC, ord, setpoint):
    if not (hasattr(SC.RING[ord], 'BendingAngle') and SC.RING[ord].BendingAngle != 0 and ord in SC.ORD.CM[0]):
        return SC
    idealKickDifference = ((setpoint - (SC.RING[ord].SetPointB[1] - SC.RING[ord].NomPolynomB[1])) /
                           SC.RING[ord].NomPolynomB[1] - 1) * SC.RING[ord].BendingAngle / SC.RING[ord].Length
    SC, _ = SCsetCMs2SetPoints(SC, ord, idealKickDifference * SC.RING[ord].Length, skewness=False, method='add')
    return SC


def _check_input_and_setpoints(method, ords, setpoints):
    if method not in VALID_METHODS:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {VALID_METHODS}.')
    if len(setpoints) not in (1, len(ords)):
        raise ValueError(f'Setpoints have to have length of 1 or matching to the length or ordinates.')
    if len(setpoints) == 1:
        return np.repeat(setpoints, len(ords))
    return setpoints