"""
Lattice setting
-------------

This module contains the 'machine-based' functions to interact with lattice under study.
"""
import numpy as np
from at import Lattice
from numpy import ndarray
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def switch_rf(ring: Lattice, ords: ndarray, state: bool) -> Lattice:
    cavs = [i for i in np.ravel(np.array([ords], dtype=int)) if hasattr(ring[i], 'Frequency')]
    for ind in cavs:
        ring[ind].PassMethod = 'RFCavityPass' if state else 'IdentityPass'
    return ring


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

