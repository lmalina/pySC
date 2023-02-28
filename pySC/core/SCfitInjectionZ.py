import matplotlib.pyplot as plt
import numpy as np
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.at_wrapper import atpass, findspos

def SCfitInjectionZ(SC, mode, nDims=np.array([0,1]), nBPMs=np.array([0,1,2]), nShots=None, verbose=0, plotFlag=False):
    if nShots is None:
        nShots = SC.INJ.nShots
    else:
        SC.INJ.nShots = nShots
    ERROR = 0
    deltaZ0 = np.zeros(6)

    B = SCgetBPMreading(SC)
    if mode == 'fitTrajectory':
        ordsUsed = SC.ORD.BPM[nBPMs]
        Bref = B[:, nBPMs]
        deltaZ0[0:4] = -fminsearch(merritFunction, np.zeros(4)) # TODO pass variables?
        if plotFlag:
            SC.INJ.Z0 = SC.INJ.Z0 + deltaZ0
            B1 = SCgetBPMreading(SC)
            sBPM = findspos(SC.RING, SC.ORD.BPM[nBPMs])
            plt.figure(342)
            plt.clf()
            titleStr = ['Horizontal', 'Vertical']
            for nDim in range(2):
                plt.subplot(1, 2, nDim + 1)
                plt.plot(sBPM, 1E3 * Bref[nDim, :], 'O--', sBPM, 1E3 * B1[nDim, nBPMs], 'X--')
                plt.xlabel('s [m]')
                plt.ylabel('BPM reading [mm]')
                plt.title(titleStr[nDim])
                plt.legend(['Initial', 'After correction'])
            plt.show()
    elif mode == 'injectionDrift':
        tmpS = findspos(SC.RING, SC.ORD.BPM)
        sBPM = [tmpS[-1] - findspos(SC.RING, len(SC.RING) + 1)[0], tmpS[0]]
        Bref = [B[:, len(SC.ORD.BPM) - 1], B[:, len(SC.ORD.BPM)]]
        for nDim in nDims:
            sol[nDim] = np.polyfit(sBPM, Bref[nDim, :], 1)
            deltaZ0[2 * nDim] = - sol[nDim][1]
            deltaZ0[2 * nDim + 1] = - sol[nDim][0]
        if plotFlag:
            plt.figure(342)
            plt.clf()
            titleStr = ['Horizontal', 'Vertical']
            for nDim in nDims:
                plt.subplot(1, 2, nDim + 1)
                plt.plot(sBPM, 1E6 * Bref[nDim, :], 'o', sBPM, 1E6 * (sol[nDim][0] * sBPM + sol[nDim][1]), '--', sBPM,
                         1E6 * (SC.INJ.Z0[2 * nDim] * sBPM + SC.INJ.Z0[2 * nDim - 1]), 'k-', sBPM, [0, 0], 'k--')
                plt.legend(['BPM reading', 'Fitted trajectory', 'Real trajectory'])
                plt.xlabel('s [m]')
                plt.ylabel('Beam offset [mm]')
                plt.title(titleStr[nDim])
            plt.show()
    else:
        print('Unsupported mode: ' + mode)
    if verbose:
        print(
            '\nInjection trajectory corrected from \n x:  %.0fum -> %.0fum \n x'': %.0furad -> %.0furad \n y:  %.0fum -> %.0fum \n y'': %.0furad -> %.0furad\n' % (
            1E6 * SC.INJ.Z0[0], 1E6 * (SC.INJ.Z0[0] + deltaZ0[0]), 1E6 * SC.INJ.Z0[1],
            1E6 * (SC.INJ.Z0[1] + deltaZ0[1]), 1E6 * SC.INJ.Z0[2], 1E6 * (SC.INJ.Z0[2] + deltaZ0[2]),
            1E6 * SC.INJ.Z0[3], 1E6 * (SC.INJ.Z0[3] + deltaZ0[3])))
    if np.isnan(deltaZ0).any():
        ERROR = 1
    return deltaZ0


def merritFunction(SC, Bref, ordsUsed, x):
    Ta = atpass(SC.IDEALRING, [x, 0, 0], 1, 1, ordsUsed)
    T = Ta[[0, 2], :]
    out = np.sqrt(np.mean((Bref[:] - T[:]) ** 2))
    return out

# End
