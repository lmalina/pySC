"""
RF
-------------

This module contains functions to correct RF phase and frequency.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

from pySC.utils.at_wrapper import findorbit6
from pySC.core.beam import bpm_reading
from pySC.core.lattice_setting import set_cavity_setpoints
from pySC.utils import logging_tools
from pySC.core.constants import SPEED_OF_LIGHT

LOGGER = logging_tools.get_logger(__name__)
MIN_TURNS_FOR_LINEAR_FIT = 3


def correct_rf_phase(SC, cav_ords=None, bpm_ords=None, n_steps=15, plotResults=False, plotProgress=False):
    def _sin_fit_fun(x, a, b, c):
        return a * np.sin(2 * np.pi * x + b) + c

    LOGGER.debug(f'Calibrate RF phase with: \n {SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns '
                 f'\n {SC.INJ.nShots} Shots \n {n_steps} Phase steps.\n\n')
    cav_ords, bpm_ords = _input_check(SC, cav_ords, bpm_ords, n_steps)
    bpm_shift, bpm_shift_err = np.full(n_steps, np.nan), np.full(n_steps, np.nan)
    lamb = SPEED_OF_LIGHT / SC.RING[cav_ords[0]].FrequencySetPoint
    test_vec = 1 / 2 * lamb * np.linspace(-1, 1, n_steps)

    for step in range(n_steps):
        SC = set_cavity_setpoints(SC, cav_ords, test_vec[step], 'TimeLag', 'add')
        bpm_shift[step], bpm_shift_err[step], TBTdE = _get_tbt_energy_shift(SC, bpm_ords)
        SC = set_cavity_setpoints(SC, cav_ords, -test_vec[step], 'TimeLag', 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, bpm_shift, test_vec, phase=True)

    test_vec, bpm_shift, bpm_shift_err = _check_data(test_vec, bpm_shift, bpm_shift_err)

    # Fit sinusoidal function to data
    (param, param_cov) = curve_fit(_sin_fit_fun, test_vec / lamb, bpm_shift, sigma=bpm_shift_err, absolute_sigma=True,
                                   p0=np.array([max(bpm_shift) - np.mean(bpm_shift), np.pi, np.mean(bpm_shift)]))
    if np.abs(param[2]) > np.abs(param[0]):  # fitted sin_wave does not cross zero
        raise RuntimeError(f'Zero crossing not within fit function\n Consider increasing RF voltage to at least '
                           f'{SC.RING[cav_ords[0]].VoltageSetPoint *  np.abs(param[2] / param[0])} V.')

    sol = lambda x: _sin_fit_fun(x / lamb, param[0], param[1], param[2])

    # Find zero crossing of fitted function
    delta_phi = fsolve(sol, test_vec[np.argmax(sol(test_vec))] - abs(test_vec[0]) / 2)
    delta_phi = _fold_phase(delta_phi, lamb)
    if np.isnan(delta_phi):
        raise RuntimeError('SCsynchPhaseCorrection: ERROR (NaN phase)')

    SC = set_cavity_setpoints(SC, cav_ords, delta_phi, 'TimeLag', 'add')
    LOGGER.debug(f'Time lag correction step: {delta_phi[0]:.3f} m\n')
    if plotResults:
        _plot_results((test_vec, bpm_shift, bpm_shift_err), sol, delta_phi, x_scale=360 / lamb, phase=True)
    return SC


def correct_rf_frequency(SC, cav_ords=None, bpm_ords=None, f_range=(-1E3, 1E3), n_steps=15, plotResults=False,
                         plotProgress=False):
    LOGGER.debug(f'Correct energy error with: \n '
                 f'{SC.INJ.nParticles} Particles \n {SC.INJ.nTurns} Turns \n {SC.INJ.nShots} Shots \n {n_steps} '
                 f'Frequency steps between {1E-3 * f_range[0]:.1f} and {1E-3 * f_range[1]:.1f} kHz.\n\n')
    cav_ords, bpm_ords = _input_check(SC, cav_ords, bpm_ords, n_steps)
    bpm_shift, bpm_shift_err = np.full(n_steps, np.nan), np.full(n_steps, np.nan)
    test_vec = np.linspace(f_range[0], f_range[1], n_steps)

    # Main loop
    for step in range(n_steps):  # TODO later this does not have to set value back and forth in each step
        SC = set_cavity_setpoints(SC, cav_ords, test_vec[step], 'Frequency', 'add')
        bpm_shift[step], bpm_shift_err[step], TBTdE = _get_tbt_energy_shift(SC, bpm_ords)
        SC = set_cavity_setpoints(SC, cav_ords, -test_vec[step], 'Frequency', 'add')
        if plotProgress:
            _fun_plot_progress(TBTdE, bpm_shift, test_vec, phase=False)

    test_vec, bpm_shift, bpm_shift_err = _check_data(test_vec, bpm_shift, bpm_shift_err, min_num_points=3)
    # Fit linear function to data
    p, pcov = np.polyfit(test_vec, bpm_shift, 1, w=1 / bpm_shift_err, cov="unscaled")
    delta_f = -p[1] / p[0]
    if np.isnan(delta_f):
        raise RuntimeError('SCsynchEnergyCorrection: ERROR (NaN frequency)')

    SC = set_cavity_setpoints(SC, cav_ords, delta_f, 'Frequency', 'add')
    LOGGER.info(f'Frequency correction step: {1E-3 * delta_f:.2f} kHz')
    sol = lambda x: x * p[0] + p[1]
    if plotResults:
        _plot_results((test_vec, bpm_shift, bpm_shift_err), sol, delta_f, x_scale=1e-3, phase=False)
    return SC


def _plot_results(data, fit, corrected, x_scale, phase=True):
    test_vec, bpm_shift, bpm_shift_err = data
    f, ax = plt.subplots(nrows=1)
    y_scale = 1e6
    ax.errorbar(x_scale * test_vec, y_scale * bpm_shift, yerr=y_scale * bpm_shift_err, fmt='o', color='red',
                label="Measurement")
    ax.plot(x_scale * test_vec, y_scale * fit(test_vec), '--', color='blue', label='Fit')
    ax.plot(x_scale * 0, y_scale * fit(0), 'rD', markersize=12, label="Initial")
    ax.plot(x_scale * corrected, y_scale * 0, 'kX', markersize=16, label='Corrected')
    ax.set_xlabel(("RF phase [deg]" if phase else r'$\Delta f$ [$kHz$]'))
    ax.set_ylabel(r'$\overline{\Delta x}$ [$\mu$m / turn]')
    ax.legend()
    f.tight_layout()
    f.show()


def get_phase_and_energy_error(SC, cav_ords):
    lamb = SPEED_OF_LIGHT / SC.RING[cav_ords[0]].Frequency
    orbit6 = findorbit6(SC.RING)[0]
    phase_error = _fold_phase((orbit6[5] - SC.INJ.Z0[5]) / lamb, lamb)
    energy_error = SC.INJ.Z0[4] - orbit6[4]
    LOGGER.info(f'Static phase error {phase_error * 360:.0f} deg')
    LOGGER.info(f'Energy error {1E2 * energy_error:.2f}%')
    return phase_error, energy_error


def _check_data(test_vec, bpm_shift, bpm_shift_err, min_num_points=6):
    mask = ~np.isnan(bpm_shift)
    if np.sum(~np.isnan(bpm_shift)) < min_num_points:
        raise RuntimeError('Not enough data points for fit.')
    return test_vec[mask], bpm_shift[mask], bpm_shift_err[mask]


def _get_tbt_energy_shift(SC, bpm_ords):
    bpm_readings = bpm_reading(SC, bpm_ords)
    x_reading = np.reshape(bpm_readings[0, :], (SC.INJ.nTurns, len(bpm_ords)))
    mean_trajectory_diff_tbt = np.mean(x_reading - x_reading[0, :], axis=1)
    if len(mean_trajectory_diff_tbt[~np.isnan(mean_trajectory_diff_tbt)]) < MIN_TURNS_FOR_LINEAR_FIT:  # not enough data for linear fit
        return np.nan, np.nan, mean_trajectory_diff_tbt
    nan_mask = ~np.isnan(mean_trajectory_diff_tbt)
    fit_result = np.polyfit(np.arange(SC.INJ.nTurns)[nan_mask], mean_trajectory_diff_tbt[nan_mask], 1, cov=True)
    slope, err_slope = fit_result[0][0], np.sqrt(fit_result[1][0, 0])
    return slope, err_slope, mean_trajectory_diff_tbt


def _fun_plot_progress(mean_trajectory_diff_tbt, bpm_shift, test_vec, phase=True):
    f, ax = plt.subplots(nrows=2, num=2)
    ax[0].plot(mean_trajectory_diff_tbt, 'o')
    ax[0].plot(np.arange(len(mean_trajectory_diff_tbt)) * bpm_shift[~np.isnan(bpm_shift)][-1], '--')
    ax[0].set_xlabel('Number of turns')
    ax[0].set_ylabel(r'$<\Delta x_\mathrm{TBT}>$ [m]')
    ax[1].plot(test_vec, bpm_shift, 'o')
    ax[1].set_xlabel(r'$\Delta \phi$ [m]' if phase else r'$\Delta f$ [Hz]')
    ax[1].set_ylabel(r'$<\Delta x>$ [m/turn]')
    f.show()


def _fold_phase(delta_phi, lamb):
    dphi = np.remainder(delta_phi, lamb)
    if np.abs(dphi) > lamb / 2:
        return dphi - lamb * np.sign(dphi)
    return dphi


def _input_check(SC, cav_ords, bpm_ords, n_steps):
    if cav_ords is None:
        cav_ords = SC.ORD.RF
    if bpm_ords is None:
        bpm_ords = SC.ORD.BPM
    if turns_required := MIN_TURNS_FOR_LINEAR_FIT > SC.INJ.nTurns:
        raise ValueError(f"Not enough turns for the for the tracking: {turns_required=}  set it to SC.INJ.nTurns")
    if SC.INJ.trackMode != 'TBT':
        raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
    if n_steps < 3:
        raise ValueError('Number of steps must be larger than 3.')
    for cav_ord in cav_ords:
        if not hasattr(SC.RING[cav_ord], 'Frequency'):
            raise ValueError(f'This is not a cavity (ord: {cav_ord})')
        if SC.RING[cav_ord].PassMethod not in ('CavityPass', 'RFCavityPass'):
            raise ValueError(f'Cavity (ord: {cav_ord}) seems to be switched off.')
    return cav_ords, bpm_ords
