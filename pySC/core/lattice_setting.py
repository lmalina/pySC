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

    # present radiation state
    radstate = ring.radiation

    # present cavity state
    ords = np.arange(len(ring))
    cavs = [i for i in np.ravel(np.array([ords], dtype=int)) if hasattr(ring[i], 'Frequency')]

    cavstate = False
    pm=[]
    for ind in cavs:
        pm.append(ring[ind].PassMethod)
    cavpm=np.unique(pm)
    if len(cavpm) > 1:
        print(f'cavity pass methods are inconsistent. Assume cavities are off.')
        cavstate=False
    elif len(cavpm) == 1:
        if cavpm == 'RFCavityPass':
            cavstate = True
        else:
            cavstate = False

    if 'radiationoff' in args:
        if cavstate:  # rad off, cav on
            ring.disable_6d(cavity_pass='RFCavityPass')
        else:  # rad off, cav off
            ring.disable_6d()

    elif 'radiationon' in args:
        if cavstate:  # rad on, cav on
            ring.enable_6d()
        else:  # rad on, cav off
            ring.enable_6d(cavity_pass='IdentityPass')

    if 'cavityoff' in args:
        if radstate:  # rad on, cav off
            ring.enable_6d(cavity_pass='IdentityPass')
        else:  # rad off, cav off
            ring.disable_6d()

    elif 'cavityon' in args:
        if radstate:  # rad on, cav on
            ring.enable_6d()
        else:  # rad off, cav on
            ring.disable_6d(cavity_pass='RFCavityPass')

    return ring

