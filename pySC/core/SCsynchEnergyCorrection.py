import matplotlib.pyplot as plt
import numpy as np

from pySC.at_wrapper import findorbit6
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints

# TODO currently not needed
def SCsynchEnergyCorrection(SC, cavOrd=None, f_range=(-1E3, 1E3), nSteps=15, nTurns=150, minTurns=0, plotResults=0,
                            plotProgress=0, verbose=0):
    if cavOrd is None:
        cavOrd = SC.ORD.RF
    ERROR = 0
    deltaF = 0
    fTestVec = np.linspace(f_range[0], f_range[1], nSteps)
    BPMshift = np.nan * np.ones(len(fTestVec))
    SC.INJ.nTurns = nTurns
    if verbose:
        print(f'Correct energy error with: \n {SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns \n '
              f'{SC.INJ.nShots} Shots \n {nSteps} Frequency steps '
              f'between [{1E-3 * f_range[0]}.1f {1E-3 * f_range[1]}.1f] kHz.\n\n')
    for nE in range(len(fTestVec)):
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', np.array([fTestVec[nE]]), 'add')
        [BPMshift[nE], TBTdE] = getTbTEnergyShift(SC, minTurns)
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', -np.array([fTestVec[nE]]), 'add')
        if plotProgress:
            funPlotProgress(TBTdE, BPMshift, fTestVec, nE)

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
        plt.plot(1E-3 * fTestVec, 1E6 * BPMshift, 'o')
        plt.plot(1E-3 * fTestVec, 1E6 * (fTestVec * p[0] + p[1]), '--')
        plt.plot(1E-3 * deltaF, 0, 'kX', markersize=16)
        plt.xlabel('$\Delta f$ [$kHz$]')
        plt.ylabel('$<\Delta x>$ [$\mu$m/turn]')
        plt.legend({'Measurement', 'Fit', 'dE correction'})  # ,'Closed orbit'})
        plt.show()
        




    if np.isnan(deltaF):
        ERROR = 2
        print('NaN energy correction step.')
        return deltaF, ERROR
    if verbose:
        XCO = findorbit6(SC.RING)[0]
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', np.array([deltaF]), 'add')
        XCOfinal = findorbit6(SC.RING)[0]
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', -np.array([deltaF]), 'add')

        print('Frequency correction step: %.2fkHz' % (1E-3 * deltaF))
        print('>> Energy error corrected from %.2f%% to %.2f%%' % (
        1E2 * (SC.INJ.Z0[4] - XCO[4]), 1E2 * (SC.INJ.Z0[4] - XCOfinal[4])))
    return deltaF, ERROR


def getTbTEnergyShift(SC, minTurns):
    B = SCgetBPMreading(SC,do_plot=SC.plot)
    BB = np.transpose(np.reshape(B[0, :], (SC.INJ.nTurns, len(SC.ORD.BPM))))
    dE = np.mean(BB - np.transpose(np.tile(BB[:, 0], (SC.INJ.nTurns, 1))), axis=0)
    x = np.linspace(1, SC.INJ.nTurns-1, SC.INJ.nTurns-1)
    y = dE[1:]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) < minTurns:
        BPMshift = np.nan
    else:
        A = np.vstack([x, np.zeros(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        BPMshift = m
    return BPMshift, dE

def funPlotProgress(TBTdE, BPMshift, fTestVec, nE):
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 1, 1)
    # plt.hold(True)
    plt.plot(TBTdE, 'o')
    plt.plot(np.arange(len(TBTdE)) *BPMshift[nE], '--')
    plt.xlabel('Number of turns')
    plt.ylabel('$<\Delta x_\mathrm{TBT}>$ [m]')
    plt.subplot(2, 1, 2)
    plt.plot(fTestVec[0:nE], BPMshift[0:nE], 'o')
    plt.xlabel('$\Delta f$ [Hz]')
    plt.ylabel('$<\Delta x>$ [m/turn]')
    plt.show()



