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
    """
    Set RF properties to setpoints
    
    Set the setpoints of `Voltage`, `Frequency` or `TimeLag` as specified in "param" of the rf
    cavities specified in `ords`. If only a single setpoint is given for multiple cavities,
    the setpoint is applied to all cavities.

    Args:
        SC: SimulatedCommissioning class instance
        ords: Array of cavity ordinates in the lattice structure (SC.ORD.RF)
        setpoints: Setpoints (array or single value for all cavities)
        param: String ('Voltage', 'Frequency' or 'TimeLag') specifying which cavity field should be set.
        method: 'abs' (default), Use absolute setpoint
                'rel', Use relative setpoint to nominal value
                'add', Add setpoints to current value
        
    Returns:
        The modified SC structure.
    
    Examples:
        Sets the time lag of all cavities registered in SC to zero::
            
            SC = set_cavity_setpoints(SC, ords=SC.ORD.Cavity,
                                      setpoints=0, param='TimeLag')
        
        Adds 1kHz to the frequency of the first cavity::
            
            SC = set_cavity_setpoints(SC, ords=SC.ORD.Cavity(1),
                                      setpoints=1E3, param='Frequency',
                                      method='add')
    
    """
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
    """

    Return current dipole Corrector Magnets (CM) setpoints

    Reads the setpoints of the CMs specified in `ords` in the dimension `skewness`.

    Args:
        SC: SimulatedCommissioning class instance
        ords: Array of CM ordinates in the lattice structure (ex: SC.ORD.CM[0])
        skewness: boolean specifying CM dimension ([False|True] -> [hor|ver])

    Returns:
        CM setpoints [rad]
    """
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
    """
    Sets dipole corrector magnets to different setpoints

    Sets horizontal or vertical CMs as specified in `ords` and `skewness`, respectively, to `setpoints`
    [rad] and updates the magnetic fields. If the corresponding setpoint exceeds the CM limit
    specified in the corresponding lattice field `CMlimit`, the CM is clipped to that value
    and a warning is being printed (to switch off, use `warning('off','SC:CM1'))`. Positive setpoints
    will results in kicks in the positive horizontal or vertical direction.

    Args:
        SC: SimulatedCommissioning class instance
        ords:  Array of CM ordinates in the lattice structure (ex: SC.ORD.CM[0])
        setpoints:  CM setpoints (array or single value for all CMs) [rad]
        skewness: boolean specifying CM dimension ([False|True] -> [hor|ver])
        method: 'abs' (default), Use absolute setpoint
                'rel', Use relative setpoint to current value
                'add', Add setpoints to current value

    Returns:

        The lattice structure with modified and applied setpoints

    Examples:
        Set all registered horizontal CMs to zero::

            SC = set_cm_setpoints(SC, ords=SC.ORD.HCM,
                                    skewness=False, setpoints=0)

        Add 10urad to the fourth registered vertical CM::

            SC = set_cm_setpoints(SC, ords=SC.ORD.VCM[4],
                                    setpoints=1E-5, skewness=True,
                                    method='add')

    """
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
    """
    Sets magnets to setpoints

    Sets magnets (except CMs) as specified in `ords` to `setpoints` while `order` and `skewness` defines
    which field entry should be used (see below). The setpoints may be given relative to their nominal
    value or in absolute terms. If the considered quadrupole is a combined function magnet with
    non-zero bending angle and the kick compensation flag 'dipole_compensation'=True, the appropriate bending
    angle difference is calculated and the horizontal CM setpoint is changed accordingly to compensate
    for that dipole kick difference.
    If the setpoint of a skew quadrupole exceeds the limit specified in the corresponding lattice
    field `SkewQuadLimit`, the setpoint is clipped to that value and a warning is being printed.

    Args:
        SC: SimulatedCommissioning class instance
        ords:  Array of magnets ordinates in the lattice structure (ex: SC.ORD.HCM) (numpy.array() or list of int [int,int,..])
        setpoints:  magnets setpoints (array or single value for all magnets).
                    setpoints are assigned to the given order and skewness, i.e. once updated through
                    SimulatedCommissioning.apply_errors, they correspond to a single element of PolynomA or PolynomB
        skewness: boolean specifying magnet plane ([False|True] -> [PolynomB|PolynomA])
        method: 'abs' (default), Use absolute setpoint
                'rel', Use relative setpoint to nominal value
                'add', Add setpoints to current value
        order: Numeric value defining the order of the considered magnet: [0,1,2,...] => [dip,quad,sext,...]
        dipole_compensation: (default = False) Used for combined function magnets. If this flag is set and if there is a horizontal CM
                            registered in the considered magnet, the CM is used to compensate the bending angle difference
                            if the applied quadrupole setpoints differs from the design value.

    Returns:
        The SimulatedCommissioning class instance containing lattice with modified and applied setpoints.

    Examples:
        Identify the ordinates of all elements named `'SF'` and switch their sextupole component off::

            ords = SCgetOrds(SC.RING,'SF')
            SC.register_magnets(ords)
            SC = set_magnet_setpoints(SC, ords=ords,
                                      skewness=False, order=2, setpoints=0.0,
                                      method='abs')

        Identify the ordinates of all elements named `QF` and `QD` and set their quadrupole component to 99% of their design value::

            ords = SCgetOrds(SC.RING,'QF|QD')
            SC.register_magnets(ords)
            SC = set_magnet_setpoints(SC, ords=ords,
                                      skewness=False, order=1, setpoints=0.99,
                                      method='rel')

    """
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
    """
    switch cavity / radiation to on / off

    Depending on `mode` switch_cavity_and_radiation switches the cavities / the radiation in `RING` on or off.
    Possible `mode`s are "radiationoff", "radiationon", "cavityoff", "cavityon".
    Multiple modes can be specified.

    Args:
        ring:
            AT lattice, for example SC.RING
        "radiationoff": turn OFF radiation.  Priority is given to the *OFF modes.
        "radiationon": turn ON radiation.  Priority is given to the *OFF modes.
        "cavityoff": turn OFF cavity.  Priority is given to the *OFF modes.
        "cavityon" : turn ON cavity.  Priority is given to the *OFF modes.

    Returns:
        ring:
            The modified base AT structure lattice.

    Examples:
        Switch cavities and radiation in `SC.RING` off.::

            SC.RING = switch_cavity_and_radiation(SC.RING, 'cavityoff', 'radiationoff')
    """

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
