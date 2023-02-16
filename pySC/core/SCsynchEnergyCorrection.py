import matplotlib.pyplot as plt
import numpy as np

from pySC import findorbit6
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCavs2SetPoints import SCsetCavs2SetPoints


def SCsynchEnergyCorrection(SC, cavOrd=None, range=(-1E3, 1E3), nSteps=15, nTurns=150, minTurns=0, plotResults=0,
                            plotProgress=0, verbose=0):
    if cavOrd is None:
        cavOrd = SC.ORD.Cavity
    ERROR = 0
    deltaF = 0
    fTestVec = np.linspace(range[0], range[1], nSteps)
    BPMshift = np.nan * np.ones(len(fTestVec))
    SC.INJ.nTurns = nTurns
    if verbose:
        print(f'Correct energy error with: \n {SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns \n '
              f'{SC.INJ.nShots} Shots \n {nSteps} Frequency steps '
              f'between [{1E-3 * range[0]}.1f {1E-3 * range[1]}.1f] kHz.\n\n')
    for nE in range(len(fTestVec)):
        tmpSC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', fTestVec[nE], 'add')
        [BPMshift[nE], TBTdE] = getTbTEnergyShift(tmpSC, minTurns)
        if plotProgress:
            _plotProgress(TBTdE, BPMshift, fTestVec, nE)
    x = fTestVec
    y = BPMshift
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) == 0:
        ERROR = 1
        print('No transmission.')
        return deltaF, ERROR
    p = np.polyfit(x, y, 1)
    deltaF = -p[1] / p[0]
    if plotResults:
        _plotFunction(fTestVec, BPMshift, deltaF, p)
    if np.isnan(deltaF):
        ERROR = 2
        print('NaN energy correction step.')
        return deltaF, ERROR
    if verbose:
        XCO = findorbit6(SC.RING)
        tmpSC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', deltaF, 'add')
        XCOfinal = findorbit6(tmpSC.RING)
        print('Frequency correction step: %.2fkHz' % (1E-3 * deltaF))
        print('>> Energy error corrected from %.2f%% to %.2f%%' % (
        1E2 * (SC.INJ.Z0[5] - XCO[5]), 1E2 * (SC.INJ.Z0[5] - XCOfinal[5])))
    return deltaF, ERROR


def getTbTEnergyShift(SC, minTurns):
    B = SCgetBPMreading(SC)
    BB = B[0, :].reshape([], SC.INJ.nTurns)
    TBTdE = np.mean(BB - np.tile(BB[:, 0], (SC.INJ.nTurns, 1)).T, axis=1)
    x = np.arange(SC.INJ.nTurns)
    y = TBTdE
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) < minTurns:
        BPMshift = np.nan
    else:
        BPMshift = np.linalg.lstsq(x[:, np.newaxis], y[:, np.newaxis], rcond=None)[0][0]
    return BPMshift, TBTdE


def _plotProgress(TBTdE, BPMshift, fTestVec, nE):
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(TBTdE, 'o')
    plt.plot(np.arange(len(TBTdE)) * BPMshift[nE], '--')
    plt.xlabel('Number of turns')
    plt.ylabel('$<\Delta x_\mathrm{TBT}>$ [m]')
    plt.subplot(2, 1, 2)
    plt.plot(1E-3 * fTestVec[:nE], 1E6 * BPMshift[:nE], 'o')
    plt.xlabel('$\Delta f$ [kHz]')
    plt.ylabel('$<\Delta x>$ [$\mu$m/turn]')
    plt.show()


def _plotFunction(fTestVec, BPMshift, deltaF, p):
    plt.figure(88)
    plt.clf()
    plt.hold(True)
    plt.plot(1E-3 * fTestVec, 1E6 * BPMshift, 'o')
    plt.plot(1E-3 * fTestVec, 1E6 * (fTestVec * p[0] + p[1]), '--')
    plt.plot(1E-3 * deltaF, 0, 'kX', MarkerSize=16)
    plt.xlabel('$\Delta f$ [$kHz$]')
    plt.ylabel('$<\Delta x>$ [$\mu$m/turn]')
    plt.legend({'Measurement', 'Fit', 'dE correction'})  # ,'Closed orbit'})
    plt.show()
