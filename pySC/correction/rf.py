import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

from pySC.utils.at_wrapper import findorbit6
from pySC.core.beam import bpm_reading
from pySC.core.lattice_setting import set_cavity_setpoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def SCsynchPhaseCorrection(SC, cavOrd=None, nSteps=15, plotResults=False, plotProgress=False):
    # TODO minimum number of turns as in energy correction?
    LOGGER.debug(f'Calibrate RF phase with: \n {SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns '
                 f'\n {SC.INJ.nShots} Shots \n {nSteps} Phase steps.\n\n')
    if cavOrd is None:
        cavOrd = SC.ORD.RF
    if turns_required := 2 > SC.INJ.nTurns:
        raise ValueError(f"Not enough turns for the for the tracking: {turns_required=}  set it to SC.INJ.nTurns")
    if SC.INJ.trackMode != 'TBT':
        raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
    bpm_shift = np.full(nSteps, np.nan)
    lamb = 299792458 / SC.RING[cavOrd[0]].Frequency
    l_test_vec = 1 / 2 * lamb * np.linspace(-1, 1, nSteps)

    # Main loop
    for nL in range(len(l_test_vec)):
        SC = set_cavity_setpoints(SC, cavOrd, 'TimeLag', np.array([l_test_vec[nL]]), 'add')
        bpm_shift[nL], TBTdE = _get_tbt_energy_shift(SC)
        SC = set_cavity_setpoints(SC, cavOrd, 'TimeLag', -np.array([l_test_vec[nL]]), 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, bpm_shift, l_test_vec, nL, phase=True)

    # Check data
    l_test_vec = l_test_vec[~np.isnan(bpm_shift)]
    bpm_shift = bpm_shift[~np.isnan(bpm_shift)]
    if len(bpm_shift) < 3:
        raise RuntimeError('Not enough data points for fit.')
    if not (max(bpm_shift) > 0 > min(bpm_shift)):
        raise RuntimeError('Zero crossing not within data set.\n')

    # Fit sinusoidal function to data
    param, param_cov = curve_fit(_sin_fit_fun, l_test_vec/lamb, bpm_shift, p0=np.array([max(bpm_shift)-np.mean(bpm_shift), 3.14, np.mean(bpm_shift)]))
    sol = lambda x: _sin_fit_fun(x/lamb, param[0], param[1], param[2])

    if not (max(sol(l_test_vec)) > 0 > min(sol(l_test_vec))):
        raise RuntimeError('Zero crossing not within fit function\n')

    # Find zero crossing of fitted function
    delta_phi = fsolve(sol, l_test_vec[np.argmax(sol(l_test_vec))] - abs(l_test_vec[0]) / 2)
    delta_phi = _fold_phase(delta_phi, lamb)
    if np.isnan(delta_phi):
        raise RuntimeError('SCsynchPhaseCorrection: ERROR (NaN phase)')

    initial = _fold_phase((findorbit6(SC.RING)[0][5] - SC.INJ.Z0[5]) / lamb, lamb)
    SC = set_cavity_setpoints(SC, cavOrd, 'TimeLag', delta_phi, 'add')
    final = _fold_phase((findorbit6(SC.RING)[0][5] - SC.INJ.Z0[5]) / lamb, lamb)
    LOGGER.debug(f'Time lag correction step: {delta_phi[0]:.3f} m\n')
    LOGGER.debug(f'Static phase error corrected from {initial*360:.0f} deg to {final*360:.1f} deg')

    if plotResults:
        plt.plot(l_test_vec / lamb * 360, bpm_shift, 'o', color='red', label="data")
        plt.plot(l_test_vec / lamb * 360, sol(l_test_vec), '--', color='blue', label="fit")
        plt.plot(initial / lamb * 360, sol(initial), 'rD', markersize=12, label="Initial time lag")
        plt.plot(delta_phi / lamb * 360, 0, 'kX', markersize=12, label="Final time lag")
        plt.xlabel("RF phase [deg]")
        plt.ylabel("BPM change [m]")
        plt.legend()
        plt.show()

    return SC


def SCsynchEnergyCorrection(SC, cavOrd=None, f_range=(-1E3, 1E3), nSteps=15, minTurns=0, plotResults=False,
                            plotProgress=False):
    LOGGER.debug(f'Correct energy error with: \n '
                 f'{SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns \n {SC.INJ.nShots} Shots \n {nSteps} '
                 f'Frequency steps between {1E-3 * f_range[0]:.1f} and {1E-3 * f_range[1]:.1f} kHz.\n\n')
    if turns_required := max(minTurns, 2) > SC.INJ.nTurns:
        raise ValueError(f"Not enough turns for the for the tracking: {turns_required=}  set it to SC.INJ.nTurns")
    if SC.INJ.trackMode != 'TBT':
        raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
    if cavOrd is None:
        cavOrd = SC.ORD.RF
    BPM_shift = np.full(nSteps, np.nan)
    f_test_vec = np.linspace(f_range[0], f_range[1], nSteps)

    # Main loop
    for nE in range(len(f_test_vec)):  # TODO later this does not have to set value back and forth in each step
        SC = set_cavity_setpoints(SC, cavOrd, 'Frequency', np.array([f_test_vec[nE]]), 'add')
        [BPM_shift[nE], TBTdE] = _get_tbt_energy_shift(SC, minTurns)
        SC = set_cavity_setpoints(SC, cavOrd, 'Frequency', -np.array([f_test_vec[nE]]), 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, BPM_shift, f_test_vec, nE, phase=False)

    # Check data
    f_test_vec = f_test_vec[~np.isnan(BPM_shift)]
    BPM_shift = BPM_shift[~np.isnan(BPM_shift)]
    if len(BPM_shift) < 2:
        raise RuntimeError('Not enough data points for fit.')

    # Fit linear function to data
    p = np.polyfit(f_test_vec, BPM_shift, 1)
    delta_f = -p[1] / p[0]
    if np.isnan(delta_f):
        raise RuntimeError('SCsynchEnergyCorrection: ERROR (NaN frequency)')

    initial = SC.INJ.Z0[4] - findorbit6(SC.RING)[0][4]
    SC = set_cavity_setpoints(SC, cavOrd, 'Frequency', np.array([delta_f]), 'add')
    final = SC.INJ.Z0[4] - findorbit6(SC.RING)[0][4]
    LOGGER.debug(f'Frequency correction step: {1E-3 * delta_f:.2f} kHz')
    LOGGER.debug(f'Energy error corrected from {1E2*initial:.2f}% to {1E2*final:.2f}%')

    if plotResults:
        plt.plot(1E-3 * f_test_vec, 1E6 * BPM_shift, 'o', label='Measurement')
        plt.plot(1E-3 * f_test_vec, 1E6 * (f_test_vec * p[0] + p[1]), '--', label='Fit')
        plt.plot(1E-3 * delta_f, 0, 'kX', markersize=16, label='dE correction')
        plt.xlabel(r'$\Delta f$ [$kHz$]')
        plt.ylabel(r'$<\Delta x>$ [$\mu$m/turn]')
        plt.show()

    return SC


def _sin_fit_fun(x, a, b, c):
    return a * np.sin(2 * np.pi * x + b) + c


def _get_tbt_energy_shift(SC, min_turns=2):
    bpm_readings = bpm_reading(SC)
    x_reading = np.reshape(bpm_readings[0, :], (SC.INJ.nTurns, len(SC.ORD.BPM)))
    TBTdE = np.mean(x_reading - x_reading[0, :], axis=1)
    if len(TBTdE[~np.isnan(TBTdE)]) < min_turns:  # not enough data for linear fit
        return np.nan, TBTdE
    nan_mask = ~np.isnan(TBTdE)
    return np.polyfit(np.arange(SC.INJ.nTurns)[nan_mask], TBTdE[nan_mask], 1)[0], TBTdE


def _fun_plot_progress(TBTdE, BPM_shift, f_test_vec, nE, phase=True):
    f, ax = plt.subplots(nrows=2, num=2)
    ax[0].plot(TBTdE, 'o')
    ax[0].plot(np.arange(len(TBTdE)) * BPM_shift[nE], '--')
    ax[0].set_xlabel('Number of turns')
    ax[0].set_ylabel(r'$<\Delta x_\mathrm{TBT}>$ [m]')
    ax[1].plot(f_test_vec[:nE], BPM_shift[:nE], 'o')
    ax[1].set_xlabel(r'$\Delta \phi$ [m]' if phase else r'$\Delta f$ [m]')
    ax[1].set_ylabel(r'$<\Delta x>$ [m/turn]')
    plt.show()


def _fold_phase(delta_phi, lamb):
    if np.abs(delta_phi) > lamb / 2:
        return delta_phi - lamb * np.sign(delta_phi)
    return delta_phi


# def inputCheck(SC, par):
#     if SC.INJ.trackMode != 'TBT':
#         raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
#     if nSteps < 2 or nTurns < 2:
#         raise ValueError('Number of steps and number of turns must be larger than 2.')
#     if not hasattr(SC.RING[cavOrd], 'Frequency'):
#         raise ValueError('This is not a cavity (ord: %d)' % cavOrd)
#     if not any(SC.RING[cavOrd].PassMethod in ['CavityPass', 'RFCavityPass']):
#         raise ValueError('Cavity (ord: %d) seemed to be switched off.' % cavOrd)
