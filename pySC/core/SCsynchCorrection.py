import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, fsolve

from pySC.at_wrapper import findorbit6
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def SCsynchPhaseCorrection(SC, cavOrd=None, nSteps=15, nTurns=20, plotResults=False, plotProgress=False, verbose=False):
    if cavOrd is None:
        cavOrd = SC.ORD.RF

    SC.INJ.nTurns = nTurns  # TODO this is not nice
    BPMshift = np.full(nSteps, np.nan)
    lamb = 299792458 / SC.RING[cavOrd[0]].Frequency
    timeLagVec = 1 / 2 * lamb * np.linspace(-1, 1, nSteps)

    LOGGER.debug(f'Calibrate RF phase with: \n {SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns '
                 f'\n {SC.INJ.nShots} Shots \n {nSteps} Phase steps.\n\n')

    for nL in range(len(timeLagVec)):
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', np.array([timeLagVec[nL]]), 'add')
        BPMshift[nL], TBTdE = _get_tbt_energy_shift(SC)
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', -np.array([timeLagVec[nL]]), 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, BPMshift, timeLagVec, nL, phase=True)

    if not (max(BPMshift) > 0 > min(BPMshift)):
        raise RuntimeError('Zero crossing not within data set.\n')

    def fit_fun(x, a, b, c):
        return a * np.sin(np.pi * x + b) + c

    p0 = [max(BPMshift)-np.mean(BPMshift), 3.14, np.mean(BPMshift)]
    param, param_cov = curve_fit(fit_fun, timeLagVec, BPMshift, p0=p0)
    sol = lambda x: fit_fun(x, param[0], param[1], param[2])

    if not (max(sol(timeLagVec)) > 0 > min(sol(timeLagVec))):
        raise RuntimeError('Zero crossing not within fit function\n')

    deltaPhi = fsolve(sol, timeLagVec[np.argmax(sol(timeLagVec))] - abs(timeLagVec[0]) / 2)
    deltaPhi = _fold_phase(deltaPhi, lamb)
    if np.isnan(deltaPhi):
        raise RuntimeError('SCrfCommissioning: ERROR (NaN phase)\n')

    if plotResults:
        plt.plot(timeLagVec / lamb * 360, BPMshift, 'o', color='red', label="data")
        plt.plot(timeLagVec / lamb * 360, sol(timeLagVec), '--', color='blue', label="fit")
        plt.plot(SC.INJ.Z0[5] / lamb * 360, sol(SC.INJ.Z0[5]), 'rD', markersize=12, label="Initial time lag")
        plt.plot(deltaPhi / lamb * 360, 0, 'kX', markersize=12, label="Final time lag")
        plt.xlabel("RF phase [deg]")
        plt.ylabel("BPM change [m]")
        plt.legend()
        plt.show()

    if verbose:
        XCO = findorbit6(SC.RING)[0]
        tmpSC = copy.deepcopy(SC)
        tmpSC = SCsetCavs2SetPoints(tmpSC, cavOrd, 'TimeLag', deltaPhi, 'add')
        XCOfinal = findorbit6(tmpSC.RING)[0]
        initial = np.fmod((XCO[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        final = np.fmod((XCOfinal[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        LOGGER.debug(f'Time lag correction step: {deltaPhi[0]:.3f} m\n')
        LOGGER.debug(f'Static phase error corrected from {initial:.0f} deg to {final:.1f} deg')

    return deltaPhi


def SCsynchEnergyCorrection(SC, cavOrd=None, f_range=(-1E3, 1E3), nSteps=15, nTurns=150, minTurns=0, plotResults=False,
                            plotProgress=False, verbose=False):
    if cavOrd is None:
        cavOrd = SC.ORD.RF

    SC.INJ.nTurns = nTurns  # TODO this is not nice
    BPMshift = np.full(nSteps, np.nan)
    fTestVec = np.linspace(f_range[0], f_range[1], nSteps)
    LOGGER.debug(f'Correct energy error with: \n '
                 f'{SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns \n {SC.INJ.nShots} Shots \n {nSteps} '
                 f'Frequency steps between {1E-3 * f_range[0]:.1f} and {1E-3 * f_range[1]:.1f} kHz.\n\n')

    for nE in range(len(fTestVec)):
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', np.array([fTestVec[nE]]), 'add')
        [BPMshift[nE], TBTdE] = _get_tbt_energy_shift(SC, minTurns)
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', -np.array([fTestVec[nE]]), 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, BPMshift, fTestVec, nE, phase=False)

    fTestVec = fTestVec[~np.isnan(BPMshift)]
    BPMshift = BPMshift[~np.isnan(BPMshift)]
    if len(BPMshift) < 2:
        raise RuntimeError('No transmission.')

    p = np.polyfit(fTestVec, BPMshift, 1)
    deltaF = -p[1] / p[0]
    if np.isnan(deltaF):
        raise RuntimeError('NaN energy correction step.')

    if plotResults:
        plt.plot(1E-3 * fTestVec, 1E6 * BPMshift, 'o')
        plt.plot(1E-3 * fTestVec, 1E6 * (fTestVec * p[0] + p[1]), '--')
        plt.plot(1E-3 * deltaF, 0, 'kX', markersize=16)
        plt.xlabel(r'$\Delta f$ [$kHz$]')
        plt.ylabel(r'$<\Delta x>$ [$\mu$m/turn]')
        plt.legend({'Measurement', 'Fit', 'dE correction'})  # ,'Closed orbit'})
        plt.show()

    if verbose:
        XCO = findorbit6(SC.RING)[0]
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', np.array([deltaF]), 'add')
        XCOfinal = findorbit6(SC.RING)[0]
        SC = SCsetCavs2SetPoints(SC, cavOrd, 'Frequency', -np.array([deltaF]), 'add')
        LOGGER.debug(f'Frequency correction step: {1E-3 * deltaF:.2f} kHz')
        LOGGER.debug(f'Energy error corrected from {1E2 * (SC.INJ.Z0[4] - XCO[4]):.2f}% '
                     f'to {1E2 * (SC.INJ.Z0[4] - XCOfinal[4]):.2f}%')
    return deltaF


def _get_tbt_energy_shift(SC, min_turns=0):  # TODO should be simplified
    B = SCgetBPMreading(SC)
    x_reading = np.reshape(B[0, :], (SC.INJ.nTurns, len(SC.ORD.BPM)))
    dE = np.mean(x_reading - x_reading[0, :], axis=1)
    # TODO technically the first point should remain in for the fit, the subtraction only makes `c` invalid,
    #  but does not decrease the number of degrees of freedom
    x = np.linspace(1, SC.INJ.nTurns-1, SC.INJ.nTurns-1)
    y = dE[1:]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) < min_turns:
        return np.nan, dE
    A = np.vstack([x, np.zeros(len(x))]).T
    BPMshift, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return BPMshift, dE


def _fun_plot_progress(TBTdE, BPMshift, fTestVec, nE, phase=True):
    f, ax = plt.subplots(nrows=2, num=2)
    f.clf()
    ax[0].plot(TBTdE, 'o')
    ax[0].plot(np.arange(len(TBTdE)) * BPMshift[nE], '--')
    ax[0].xlabel('Number of turns')
    ax[0].ylabel(r'$<\Delta x_\mathrm{TBT}>$ [m]')
    ax[1].plot(fTestVec[:nE], BPMshift[:nE], 'o')
    ax[1].xlabel(r'$\Delta \phi$ [m]' if phase else r'$\Delta f$ [m]')
    ax[1].ylabel(r'$<\Delta x>$ [m/turn]')
    plt.show()


def _fold_phase(delta_phi, lamb):
    if np.abs(delta_phi) > lamb / 2:
        return delta_phi - lamb * np.sign(delta_phi)
    return delta_phi


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

