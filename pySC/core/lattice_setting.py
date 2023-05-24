import numpy as np
from typing import Tuple

from at import Lattice
from numpy import ndarray
from pySC.core.classes import SimulatedComissioning, DotDict
from pySC.utils.classdef_tools import add_padded
from pySC.utils.sc_tools import SCrandnc

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
    # skewness: old 2 -> True, 1 -> False
    new_setpoints = _check_input_and_setpoints(method, CMords, setpoints)
    order = 0
    ndim = 1 if skewness else 0
    for i, ord in enumerate(CMords):
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            normBy = np.array([1, 1])
        else:
            normBy = np.array([-1, 1]) * SC.RING[ord].Length  # positive setpoint -> positive kick -> negative horizontal field
        if method == 'rel':
            new_setpoints[i] *= (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * normBy[ndim]
        if method == 'add':
            new_setpoints[i] += (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * normBy[ndim]

        if hasattr(SC.RING[ord], 'CMlimit') and abs(new_setpoints[i]) > abs(SC.RING[ord].CMlimit[ndim]):
            print(f'CM (ord: {ord} / dim: {ndim}) is clipping')
            new_setpoints[i] = np.sign(new_setpoints[i]) * SC.RING[ord].CMlimit[ndim]
        if skewness:
            SC.RING[ord].SetPointA[order] = new_setpoints[i] / normBy[ndim]
        else:
            SC.RING[ord].SetPointB[order] = new_setpoints[i] / normBy[ndim]
    SC.update_magnets(CMords)
    return SC, new_setpoints


def SCsetMags2SetPoints(SC: SimulatedComissioning, MAGords: ndarray, skewness: bool, order: int, setpoints: ndarray, method: str = 'abs', dipCompensation: bool = False) -> SimulatedComissioning:
    # skewness: old 2 -> False, 1 -> True , order decresed by 1
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


def SCsetMultipoles(RING, ords: ndarray, BA, method: str = 'rnd', order: int = None, skewness: bool = None):
    allowed_methods = ("sys", "rnd")
    if method not in allowed_methods:
        raise ValueError(f'Unsupported multipole method {method}. Allowed are {allowed_methods}.')
    if BA.ndim != 2 or BA.shape[1] != 2:
        raise ValueError("BA has to  be numpy.array of shape N x 2.")
    if method == 'rnd':
        for ord in ords:
            randBA = SCrandnc(2, BA.shape) * BA  # TODO this should be registered in SC.SIG
            for ind, target in enumerate(("B", "A")):
                attr_name = f"Polynom{target}Offset"
                setattr(RING[ord], attr_name,
                        add_padded(getattr(RING[ord], attr_name), randBA[:, ind])
                        if hasattr(RING[ord], attr_name) else randBA[:, ind])
        return RING
    # Systematic multipole errors
    if order is None or skewness is None:
        raise ValueError(f'Order and skewness have to be defined with method "sys".')
    ind, source = (1, "A") if skewness else (0, "B")
    newBA = BA[:, :]
    newBA[order, ind] = 0
    for ord in ords:
        for target in ("A", "B"):
            attr_name = f'SysPol{target}From{source}'
            syspol = getattr(RING[ord], attr_name) if hasattr(RING[ord], attr_name) else DotDict()
            syspol[order] = newBA[:, ind]
            setattr(RING[ord], attr_name, syspol)
    return RING


def SCcronoff(ring: Lattice, *args: str) -> Lattice:  # TODO some at methods do that?
    valid_args = ('radiationoff', 'radiationon', 'cavityoff', 'cavityon')
    if invalid_args := [arg for arg in args if arg not in valid_args]:
        raise ValueError(f"Unknown arguments found: {invalid_args}"
                         f"Available options are: {valid_args}")
    for mode in args:
        if mode == 'radiationoff':
            for ind in range(len(ring)):
                if ring[ind].PassMethod == 'BndMPoleSymplectic4RadPass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4Pass'
                elif ring[ind].PassMethod == 'BndMPoleSymplectic4E2RadPass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4E2Pass'
                elif ring[ind].PassMethod == 'StrMPoleSymplectic4RadPass':
                    ring[ind].PassMethod = 'StrMPoleSymplectic4Pass'
        elif mode == 'radiationon':
            for ind in range(len(ring)):
                if ring[ind].PassMethod == 'BndMPoleSymplectic4Pass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4RadPass'
                elif ring[ind].PassMethod == 'BndMPoleSymplectic4E2Pass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4E2RadPass'
                elif ring[ind].PassMethod == 'StrMPoleSymplectic4Pass':
                    ring[ind].PassMethod = 'StrMPoleSymplectic4RadPass'
        elif mode == 'cavityoff':
            for ind in range(len(ring)):
                if hasattr(ring[ind], 'Frequency'):
                    ring[ind].PassMethod = 'IdentityPass'
        elif mode == 'cavityon':
            for ind in range(len(ring)):
                if hasattr(ring[ind], 'Frequency'):
                    ring[ind].PassMethod = 'RFCavityPass'
    return ring


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
    if len(setpoints) not in (1, len(ords)) or np.prod(setpoints.shape) > len(ords):
        raise ValueError(f'Setpoints have to have length of 1 or matching to the length or ordinates.')
    if len(setpoints) == 1:
        return np.repeat(setpoints, len(ords))
    return setpoints[:]
