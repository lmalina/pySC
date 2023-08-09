"""
Lattice setting
-------------

This module contains the 'machine-based' functions to interact with lattice under study.
"""
import numpy as np
from typing import Union, List

from at import Lattice
from numpy import ndarray

from pySC.core.constants import SETTING_METHODS, SETTING_ABS, SETTING_REL, SETTING_ADD, NUM_TO_AB
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.at_wrapper import atgetfieldvalues
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
SETPOINT = "SetPoint"


def set_cavity_setpoints(SC: SimulatedCommissioning,
                         ords: Union[int, List[int], ndarray],
                         setpoints: Union[float, List[float], ndarray],
                         param: str, method: str = SETTING_ABS) -> SimulatedCommissioning:
    ords_1d, setpoints_1d = _check_input_and_setpoints(method, ords, setpoints)
    setpoint_str = f"{param}{SETPOINT}"
    if method == SETTING_REL:
        setpoints_1d *= atgetfieldvalues(SC.RING, ords_1d, setpoint_str)
    if method == SETTING_ADD:
        setpoints_1d += atgetfieldvalues(SC.RING, ords_1d, setpoint_str)
    for i, ord in enumerate(ords_1d):
        setattr(SC.RING[ord], setpoint_str, setpoints_1d[i])
    SC.update_cavities(ords_1d)
    return SC


def switch_rf(ring: Lattice, ords: ndarray, state: bool) -> Lattice:
    cavs = [i for i in np.ravel(np.array([ords], dtype=int)) if hasattr(ring[i], 'Frequency')]
    for ind in cavs:
        ring[ind].PassMethod = 'RFCavityPass' if state else 'IdentityPass'
    return ring


def get_cm_setpoints(SC: SimulatedCommissioning, ords: Union[int, List[int], ndarray], skewness: bool) -> ndarray:
    ords_1d = np.ravel(np.array([ords], dtype=int))
    order = 0
    ndim = int(skewness)
    letter = NUM_TO_AB[ndim]
    setpoints = atgetfieldvalues(SC.RING, ords_1d, f"{SETPOINT}{letter}", order)
    for i, ord1d in enumerate(ords_1d):
        if SC.RING[ord1d].PassMethod != 'CorrectorPass':
            # positive setpoint -> positive kick -> negative horizontal field
            setpoints[i] *= (-1) ** (ndim + 1) * SC.RING[ord1d].Length
    return setpoints


def set_cm_setpoints(SC: SimulatedCommissioning,
                     ords: Union[int, List[int], ndarray],
                     setpoints: Union[float, List[float], ndarray],
                     skewness: bool, method: str = SETTING_ABS) -> SimulatedCommissioning:
    # TODO corrector does not have PolynomA/B in at?
    ords_1d, setpoints_1d = _check_input_and_setpoints(method, ords, setpoints)
    order = 0
    ndim = int(skewness)
    letter = NUM_TO_AB[ndim]
    for i, ord in enumerate(ords_1d):
        # positive setpoint -> positive kick -> negative horizontal field
        norm_by = (-1) ** (ndim + 1) * SC.RING[ord].Length if SC.RING[ord].PassMethod != 'CorrectorPass' else 1
        if method == SETTING_REL:
            setpoints_1d[i] *= getattr(SC.RING[ord], f"{SETPOINT}{letter}")[order] * norm_by
        if method == SETTING_ADD:
            setpoints_1d[i] += getattr(SC.RING[ord], f"{SETPOINT}{letter}")[order] * norm_by
        if hasattr(SC.RING[ord], 'CMlimit') and abs(setpoints_1d[i]) > abs(SC.RING[ord].CMlimit[ndim]):
            LOGGER.info(f'CM (ord: {ord} / dim: {ndim}) is clipping')
            setpoints_1d[i] = np.sign(setpoints_1d[i]) * SC.RING[ord].CMlimit[ndim]
        getattr(SC.RING[ord], f"{SETPOINT}{letter}")[order] = setpoints_1d[i] / norm_by

    SC.update_magnets(ords_1d)
    return SC


def set_magnet_setpoints(SC: SimulatedCommissioning,
                         ords: Union[int, List[int], ndarray],
                         setpoints: Union[float, List[float], ndarray],
                         skewness: bool, order: int, method: str = SETTING_ABS,
                         dipole_compensation: bool = False) -> SimulatedCommissioning:
    ords_1d, setpoints_1d = _check_input_and_setpoints(method, ords, setpoints)
    letter = NUM_TO_AB[int(skewness)]
    if method == SETTING_REL:
        setpoints_1d *= atgetfieldvalues(SC.RING, ords_1d, f"NomPolynom{letter}", order)
    if method == SETTING_ADD:
        setpoints_1d += atgetfieldvalues(SC.RING, ords_1d, f"{SETPOINT}{letter}", order)
    for i, ord in enumerate(ords_1d):
        if skewness and order == 1 and getattr(SC.RING[ord], 'SkewQuadLimit', np.inf) < np.abs(setpoints_1d[i]):
            LOGGER.info(f'SkewLim \n Skew quadrupole (ord: {ord}) is clipping')
            setpoints_1d[i] = np.sign(setpoints_1d[i]) * SC.RING[ord].SkewQuadLimit
        # TODO should check CF magnets
        if dipole_compensation and order == 1:  # quad  # TODO check also skewness?
            SC = _dipole_compensation(SC, ord, setpoints_1d[i])
        getattr(SC.RING[ord], f"{SETPOINT}{letter}")[order] = setpoints_1d[i]

    SC.update_magnets(ords_1d)
    return SC


def switch_cavity_and_radiation(ring: Lattice, *args: str) -> Lattice:  # TODO some at methods do that?
    valid_args = ('radiationoff', 'radiationon', 'cavityoff', 'cavityon')
    if invalid_args := [arg for arg in args if arg not in valid_args]:
        raise ValueError(f"Unknown arguments found: {invalid_args}"
                         f"Available options are: {valid_args}")
    non_rad_pass_methods = ['BndMPoleSymplectic4Pass', 'BndMPoleSymplectic4E2Pass', 'StrMPoleSymplectic4Pass']
    rad_pass_methods = [method.replace("Pass", "RadPass") for method in non_rad_pass_methods]

    if 'radiationoff' in args:
        for ind in range(len(ring)):
            if ring[ind].PassMethod in rad_pass_methods:
                ring[ind].PassMethod = ring[ind].PassMethod.replace("Rad", "")
    elif 'radiationon' in args:
        for ind in range(len(ring)):
            if ring[ind].PassMethod in non_rad_pass_methods:
                ring[ind].PassMethod = ring[ind].PassMethod.replace("Pass", "RadPass")
    if 'cavityoff' in args:
        return switch_rf(ring, np.arange(len(ring)), False)
    elif 'cavityon' in args:
        return switch_rf(ring, np.arange(len(ring)), True)
    return ring


def _dipole_compensation(SC, ord, setpoint):
    if getattr(SC.RING[ord], 'BendingAngle', 0) != 0 and ord in SC.ORD.HCM:
        return set_cm_setpoints(
            SC, ord, (setpoint - SC.RING[ord].SetPointB[1]) / SC.RING[ord].NomPolynomB[1] * SC.RING[ord].BendingAngle,
            skewness=False, method=SETTING_ADD)
    return SC


def _check_input_and_setpoints(method, ords, setpoints):
    if method not in SETTING_METHODS:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {SETTING_METHODS}.')
    ords_1d = np.ravel(np.array([ords], dtype=int))
    setpoints_1d = np.ravel(np.array([setpoints]))
    if len(setpoints_1d) not in (1, len(ords_1d)):
        raise ValueError(f'Setpoints have to have length of 1 or matching to the length or ordinates.')
    return ords_1d, (np.repeat(setpoints_1d, len(ords_1d)) if len(setpoints_1d) == 1 else setpoints_1d)
