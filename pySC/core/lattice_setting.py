import numpy as np
from typing import Tuple

from at import Lattice
from numpy import ndarray

from pySC.core.constants import SETTING_METHODS, SETTING_ABS, SETTING_REL, SETTING_ADD
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def set_cavity_setpoints(SC: SimulatedCommissioning, ords: ndarray, type: str, setpoints: ndarray,
                         method: str = SETTING_ABS) -> SimulatedCommissioning:
    new_setpoints = _check_input_and_setpoints(method, ords, setpoints)
    setpoint_str = f"{type}SetPoint"
    for i, ord in enumerate(ords):
        new_setpoint = new_setpoints[i]
        if method == SETTING_REL:
            new_setpoint *= getattr(SC.RING[ord], setpoint_str)
        if method == SETTING_ADD:
            new_setpoint += getattr(SC.RING[ord], setpoint_str)
        setattr(SC.RING[ord], setpoint_str, new_setpoint)
    SC.update_cavities(ords)
    return SC


def get_cm_setpoints(SC: SimulatedCommissioning, ords: ndarray, skewness: bool) -> ndarray:
    setpoints = np.nan*np.ones(len(ords))
    order = 0
    ndim = 1 if skewness else 0
    for i, ord in enumerate(ords):
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            norm_by = np.array([1, 1])
        else:
            # positive setpoint -> positive kick -> negative horizontal field
            norm_by = np.array([-1, 1]) * SC.RING[ord].Length
        if skewness:
            setpoints[i] = SC.RING[ord].SetPointA[order] * norm_by[ndim]
        else:
            setpoints[i] = SC.RING[ord].SetPointB[order] * norm_by[ndim]
    return setpoints


def set_cm_setpoints(SC: SimulatedCommissioning, ords: ndarray, setpoints: ndarray, skewness: bool,
                     method: str = SETTING_ABS) -> Tuple[SimulatedCommissioning, ndarray]:
    new_setpoints = _check_input_and_setpoints(method, ords, setpoints)
    order = 0
    ndim = 1 if skewness else 0
    for i, ord in enumerate(ords):
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            norm_by = np.array([1, 1])
        else:
            # positive setpoint -> positive kick -> negative horizontal field
            norm_by = np.array([-1, 1]) * SC.RING[ord].Length
        if method == SETTING_REL:
            new_setpoints[i] *= (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * norm_by[ndim]
        if method == SETTING_ADD:
            new_setpoints[i] += (SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]) * norm_by[ndim]

        if hasattr(SC.RING[ord], 'CMlimit') and abs(new_setpoints[i]) > abs(SC.RING[ord].CMlimit[ndim]):
            LOGGER.info(f'CM (ord: {ord} / dim: {ndim}) is clipping')
            new_setpoints[i] = np.sign(new_setpoints[i]) * SC.RING[ord].CMlimit[ndim]
        if skewness:
            SC.RING[ord].SetPointA[order] = new_setpoints[i] / norm_by[ndim]
        else:
            SC.RING[ord].SetPointB[order] = new_setpoints[i] / norm_by[ndim]
    SC.update_magnets(ords)
    return SC, new_setpoints


def set_magnet_setpoints(SC: SimulatedCommissioning, ords: ndarray, skewness: bool, order: int, setpoints: ndarray,
                         method: str = SETTING_ABS, dipole_compensation: bool = False) -> SimulatedCommissioning:
    new_setpoints = _check_input_and_setpoints(method, ords, setpoints)
    for i, ord in enumerate(ords):
        if method == SETTING_REL:
            new_setpoints[i] *= SC.RING[ord].NomPolynomA[order] if skewness else SC.RING[ord].NomPolynomB[order]
        if method == SETTING_ADD:
            new_setpoints[i] += SC.RING[ord].SetPointA[order] if skewness else SC.RING[ord].SetPointB[order]
        if skewness and order == 1:  # skew quad
            if hasattr(SC.RING[ord], 'SkewQuadLimit') and abs(new_setpoints[i]) > abs(SC.RING[ord].SkewQuadLimit):
                LOGGER.info(f'SC:SkewLim \n Skew quadrupole (ord: {ord}) is clipping')
                new_setpoints[i] = np.sign(new_setpoints[i]) * SC.RING[ord].SkewQuadLimit
        # TODO should check CF magnets
        if dipole_compensation and order == 1:  # quad  # TODO check also skewness?
            SC = _dipole_compensation(SC, ord, new_setpoints[i])
        if skewness:
            SC.RING[ord].SetPointA[order] = new_setpoints[i]
        else:
            SC.RING[ord].SetPointB[order] = new_setpoints[i]
    SC.update_magnets(ords)
    return SC


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
    ideal_kick_difference = ((setpoint - (SC.RING[ord].SetPointB[1] - SC.RING[ord].NomPolynomB[1])) /
                             SC.RING[ord].NomPolynomB[1] - 1) * SC.RING[ord].BendingAngle / SC.RING[ord].Length
    SC, _ = set_cm_setpoints(SC, ord, ideal_kick_difference * SC.RING[ord].Length, skewness=False, method=SETTING_ADD)
    return SC


def _check_input_and_setpoints(method, ords, setpoints):
    if method not in SETTING_METHODS:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {SETTING_METHODS}.')
    if len(setpoints) not in (1, len(ords)) or np.prod(setpoints.shape) > len(ords):
        raise ValueError(f'Setpoints have to have length of 1 or matching to the length or ordinates.')
    if len(setpoints) == 1:
        return np.repeat(setpoints, len(ords))
    return setpoints.copy()
