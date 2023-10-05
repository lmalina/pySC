"""
Chromaticity
-------------

This module contains functions to fit the chromaticity of 'SC.RING'.
"""

import numpy as np
from scipy.optimize import fmin

from pySC.utils.at_wrapper import atlinopt
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def fit_chroma(SC, s_ords, target_chroma=None, init_step_size=np.array([2, 2]), xtol=1E-4, ftol=1E-3):
    """
    Applies a chromaticity correction using two sextupole families.

    Args:
        SC: SimulatedCommissioning instance
        s_ords: [2xN] array or list [[1 x NS1],[1 x NS2], [1 x NS3], ...] of sextupole ordinates
        target_chroma ([1x2] array, optional): Target chromaticity for correction. Default: chromaticity of 'SC.IDEALRING'
        init_step_size ([1x2] array, optional): Initial step size for the solver. Default: [2,2]
        xtol(float, optional): Step tolerance for solver. Default: 1e-4
        ftol(float, optional): Merit tolerance for solver. Default: 1e-4

    Returns:
        SC: SimulatedCommissioning instance with corrected chromaticity.
    Example:
        SC = fit_chroma(SC, s_ords=[SCgetOrds(sc.RING, 'SF'), SCgetOrds(sc.RING, 'SD')], target_chroma=numpy.array([1,1]))
    """
    if target_chroma is None:
        target_chroma = SC.IDEALRING.get_chrom()
    if np.sum(np.isnan(target_chroma)):
        LOGGER.error('Target chromaticity must not contain NaN. Aborting.')
        return SC

    LOGGER.debug(f'Fitting chromaticities from {SC.RING.get_chrom()} to {target_chroma}.')  # first two elements
    for n in range(len(s_ords)):
        SP0.append(np.zeros_like(s_ords[n]))
    for nFam in range(len(s_ords)):
        for n in range(len(s_ords[nFam])):
            SP0[nFam][n] = SC.RING[s_ords[nFam][n]].SetPointB[2]
    fun = lambda x: _fit_chroma_fun(SC, s_ords, x, SP0, target_chroma)
    sol = fmin(fun, init_step_size, xtol=xtol, ftol=ftol)
    LOGGER.debug(f'        Final chromaticity: {SC.RING.get_chrom()}\n          Setpoints change: {sol}.')  # first two elements
    return SC


def _fit_chroma_fun(SC, s_ords, setpoints, init_setpoints, target):
    for n in range(len(s_ords)):
        SC.set_magnet_setpoints(s_ords[n], setpoints[n] + init_setpoints[n], False, 2, method='abs', dipole_compensation=True)
    nu = SC.RING.get_chrom()
    nu = nu[0:2]
    return np.sqrt(np.mean((nu - target) ** 2))
