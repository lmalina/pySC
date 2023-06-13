import matplotlib.pyplot as plt
import numpy as np
from pySC.core.beam import SCgetBPMreading
from pySC.utils.at_wrapper import atpass, findspos
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def SCfitInjectionZ(SC, mode, nDims=np.array([0, 1]), nBPMs=np.array([0, 1, 2]), plotFlag=False):
    allowed_modes = ('fitTrajectory', 'injectionDrift')
    if mode not in allowed_modes:
        raise ValueError(f"Unknown {mode=} , {allowed_modes=}")
    # uses SC.INJ.nShots
    deltaZ0 = np.zeros(6)
    s_pos = findspos(SC.RING)
    B = SCgetBPMreading(SC)
    if plotFlag:
        fig, ax = plt.subplots(nrows=2, num=342)
        titleStr = ['Horizontal', 'Vertical']
    if mode == 'fitTrajectory':
        Bref = B[:, nBPMs]
        deltaZ0[0:4] = -fminsearch(merritFunction, np.zeros(4))  # TODO pass variables?
        if plotFlag:
            SC.INJ.Z0 = SC.INJ.Z0 + deltaZ0
            B1 = SCgetBPMreading(SC)
            sBPM = s_pos[SC.ORD.BPM[nBPMs]]
            titleStr = ['Horizontal', 'Vertical']
            for nDim in range(2):
                ax[nDim].plot(sBPM, 1E3 * Bref[nDim, :], 'O--')
                ax[nDim].plot(sBPM, 1E3 * B1[nDim, nBPMs], 'X--')
                ax[nDim].set_xlabel('s [m]')
                ax[nDim].set_ylabel('BPM reading [mm]')
                ax[nDim].set_title(titleStr[nDim])
                ax[nDim].legend(['Initial', 'After correction'])
            plt.show()
    else:
        tmpS = s_pos[SC.ORD.BPM]
        sBPM = [tmpS[-1] - s_pos[-1], tmpS[0]]
        Bref = [B[:, len(SC.ORD.BPM) - 1], B[:, len(SC.ORD.BPM)]]
        sol = np.zeros((len(nDims),2))
        for nDim in nDims:
            sol[nDim] = np.polyfit(sBPM, Bref[nDim, :], 1)
            deltaZ0[2 * nDim] = - sol[nDim][1]
            deltaZ0[2 * nDim + 1] = - sol[nDim][0]
        if plotFlag:
            for nDim in nDims:
                ax[nDim].plot(sBPM, 1E6 * Bref[nDim, :], 'o', )
                ax[nDim].plot(sBPM, 1E6 * (sol[nDim][0] * sBPM + sol[nDim][1]), '--')
                ax[nDim].plot(sBPM, 1E6 * (SC.INJ.Z0[2 * nDim] * sBPM + SC.INJ.Z0[2 * nDim - 1]), 'k-')
                ax[nDim].plot(sBPM, [0, 0], 'k--')
                ax[nDim].legend(['BPM reading', 'Fitted trajectory', 'Real trajectory'])
                ax[nDim].set_xlabel('s [m]')
                ax[nDim].set_ylabel('Beam offset [mm]')
                ax[nDim].set_title(titleStr[nDim])
            plt.show()

    if np.isnan(deltaZ0).any():
        raise RuntimeError("Failed ")
    LOGGER.debug(
        '\nInjection trajectory corrected from \n x:  %.0fum -> %.0fum \n x'': %.0furad -> %.0furad \n y:  %.0fum -> %.0fum \n y'': %.0furad -> %.0furad\n' % (
        1E6 * SC.INJ.Z0[0], 1E6 * (SC.INJ.Z0[0] + deltaZ0[0]), 1E6 * SC.INJ.Z0[1],
        1E6 * (SC.INJ.Z0[1] + deltaZ0[1]), 1E6 * SC.INJ.Z0[2], 1E6 * (SC.INJ.Z0[2] + deltaZ0[2]),
        1E6 * SC.INJ.Z0[3], 1E6 * (SC.INJ.Z0[3] + deltaZ0[3])))
    return deltaZ0


def merritFunction(SC, Bref, ordsUsed, x):
    Ta = atpass(SC.IDEALRING, [x, 0, 0], 1, ordsUsed)
    T = Ta[[0, 2], :]
    out = np.sqrt(np.mean((Bref[:] - T[:]) ** 2))
    return out
