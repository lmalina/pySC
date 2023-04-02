import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, root, minimize
from scipy.optimize import fsolve

from pySC.at_wrapper import findorbit6
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints

def SCsynchPhaseCorrection(SC, cavOrd=None, nSteps=15, nTurns=20, plotResults=0, plotProgress=0, verbose=0):
    if cavOrd is None:
        cavOrd = SC.ORD.RF
    ERROR = 0
    deltaPhi = 0
    BPMshift = np.full((nSteps), np.nan)
    lamb = 299792458 / SC.RING[cavOrd[0]].Frequency
    timeLagVec = 1 / 2 * lamb * np.linspace(-1, 1, nSteps)
    SC.INJ.nTurns = nTurns
    if verbose:
        print('Calibrate RF phase with: \n %d Particles \n %d Turns \n %d Shots \n %d Phase steps.\n\n' % (
            SC.INJ.nParticles, SC.INJ.nTurns, SC.INJ.nShots, nSteps))
    for nL in range(len(timeLagVec)):
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', np.array([timeLagVec[nL]]), 'add')
        BPMshift[nL], TBTdE = getTbTEnergyShift(SC)
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', -np.array([timeLagVec[nL]]), 'add')
        if plotProgress:
            funPlotProgress(TBTdE, BPMshift, timeLagVec, nL)
    if not (max(BPMshift) > 0 > min(BPMshift)):
        print('Zero crossing not within data set.\n')
        ERROR = 1
        return deltaPhi, ERROR

    def fit_fun(x, a, b, c):
        return a * np.sin(np.pi * x + b) + c

    x = timeLagVec
    y = BPMshift
    p0 = [max(BPMshift)-np.mean(BPMshift), 3.14, np.mean(BPMshift)]
    param, param_cov = curve_fit(fit_fun, x, y, p0=p0)
    sol = lambda x: fit_fun(x, param[0], param[1], param[2])

    if not (max(sol(x)) > 0 > min(sol(x))):
        print('Zero crossing not within fit function\n')
        ERROR = 2 # TODO: proper error raise

    deltaPhi = fsolve(sol, x[np.argmax(sol(x))] - abs(timeLagVec[0]) / 2)

    if deltaPhi > lamb / 2:
        deltaPhi = deltaPhi - lamb
    elif deltaPhi < - lamb / 2:
        deltaPhi = deltaPhi + lamb
    if plotResults:
        plt.plot(x / lamb * 360, y, 'o', color='red', label="data")
        plt.plot(x / lamb * 360, sol(x), '--', color='blue', label="fit")
        plt.plot(SC.INJ.Z0[5] / lamb * 360, sol(SC.INJ.Z0[5]), 'rD', markersize=12, label="Initial time lag")
        plt.plot((deltaPhi) / lamb * 360, 0, 'kX', markersize=12, label="Final time lag")
        plt.xlabel("RF phase [deg]")
        plt.ylabel("BPM change [m]")
        plt.legend()
        plt.show()
    if np.isnan(deltaPhi):
        ERROR = 3
        print('SCrfCommissioning: ERROR (NaN phase)\n')
        return deltaPhi, ERROR
    if verbose:
        XCO = findorbit6(SC.RING)[0]
        tmpSC = copy.deepcopy(SC)
        tmpSC = SCsetCavs2SetPoints(tmpSC, cavOrd, 'TimeLag', deltaPhi, 'add')
        XCOfinal = findorbit6(tmpSC.RING)[0]
        initial = np.fmod((XCO[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        final = np.fmod((XCOfinal[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        print('>> Time lag correction step: %.3fm' % deltaPhi[0])
        print('>> Static phase error corrected from %.0fdeg to %.1fdeg' % (initial, final))
        return deltaPhi, ERROR


def getTbTEnergyShift(SC):
    B = SCgetBPMreading(SC)
    BB = np.transpose(np.reshape(B[0, :], (SC.INJ.nTurns, len(SC.ORD.BPM))))
    dE = np.mean(BB - np.transpose(np.tile(BB[:, 0], (SC.INJ.nTurns, 1))), axis=0)
    x = np.linspace(1, SC.INJ.nTurns-1, SC.INJ.nTurns-1)
    y = dE[1:]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
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
    plt.xlabel('$\Delta \phi$ [m]')
    plt.ylabel('$<\Delta x>$ [m/turn]')
    plt.show()


# TODO: implement input check

#            def inputCheck(SC, par):
#     if SC.INJ.trackMode != 'TBT':
#         raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
#     if nSteps < 2 or nTurns < 2:
#         raise ValueError('Number of steps and number of turns must be larger than 2.')
#     if not hasattr(SC.RING[cavOrd], 'Frequency'):
#         raise ValueError('This is not a cavity (ord: %d)' % cavOrd)
#     if not any(SC.RING[cavOrd].PassMethod in ['CavityPass', 'RFCavityPass']):
#         print('Cavity (ord: %d) seemed to be switched off.' % cavOrd)
#
# # End
