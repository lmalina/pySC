import numpy as np

from pySC.core.beam import bpm_reading
from pySC.utils import logging_tools
from pySC.utils.at_wrapper import atgetfieldvalues

LOGGER = logging_tools.get_logger(__name__)


def response_matrix(SC, amp, bpm_ords, cm_ords, mode='fixedKick', n_steps=2, fit_order=1):
    if ((not isinstance(amp, list) and not len(amp) == 1) or
            (isinstance(amp, list) and (len(amp[0]) != len(cm_ords[0]) or len(amp[1]) != len(cm_ords[1])))):
        raise ValueError('response_matrix amplitude must be defined as single value or '
                         'array matching the number of used HCM and VCM.')
    if not isinstance(amp, list):
        amp = [np.ones(len(cm_ords[0])) * amp, np.ones(len(cm_ords[1])) * amp]
    LOGGER.debug(f'Calculate {SC.INJ.nTurns}-turn trajectory response matrix for {len(bpm_ords)} BPMs and '
                 f'{len(cm_ords[0])}|{len(cm_ords[1])} CMs with {mode=} and '
                 f'amplitudes {np.mean(amp[0])}|{np.mean(amp[1])} using {n_steps} steps ...')
    n_hcm, n_vcm = cm_ords[0].shape[0], cm_ords[1].shape[0]
    rm = np.full((2 * SC.INJ.nTurns * bpm_ords.shape[0], n_hcm + n_vcm), np.nan)
    error = np.full((2 * SC.INJ.nTurns * bpm_ords.shape[0], n_hcm + n_vcm), np.nan)
    cm_steps = [np.zeros((n_steps, n_hcm)), np.zeros((n_steps, n_vcm))]
    bref = np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0])
    if SC.INJ.trackMode == 'ORB' and np.sum(np.isnan(bref)):
        raise ValueError('No closed orbit found.')
    i = 0
    for n_dim in range(2):
        cmstart = SC.get_cm_setpoints(cm_ords[n_dim], bool(n_dim))
        for nCM in range(len(cm_ords[n_dim])):
            max_step, gradient = _kick_amplitude(SC, bref, bpm_ords, cm_ords[n_dim][nCM], amp[n_dim][nCM], bool(n_dim), mode)
            cm_step_vec = np.linspace(-max_step, max_step, n_steps)
            if n_steps != 2:
                real_cm_setpoint = cmstart[nCM] + cm_step_vec
                gradient = np.vstack((np.zeros((n_steps - 1, len(bref))), gradient.T))
                for n_step in range(n_steps):
                    if cm_step_vec[n_step] != 0 and cm_step_vec[n_step] != max_step:
                        SC.set_cm_setpoints(cm_ords[n_dim][nCM], cmstart[nCM] + cm_step_vec[n_step], skewness=bool(n_dim))
                        real_cm_setpoint[n_step] = SC.get_cm_setpoints(cm_ords[n_dim][nCM], skewness=bool(n_dim))
                        gradient[n_step, :] = np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0]) - bref
                dCM = real_cm_setpoint - cmstart[nCM]
                cm_steps[n_dim][:, nCM] = dCM
                rm[:, i] = gradient / dCM
            else:
                dCM = max_step
                cm_steps[n_dim][:, nCM] = dCM
                for nBPM in range(gradient.shape[1]):
                    x = dCM[~np.isnan(gradient[:, nBPM])]
                    y = gradient[~np.isnan(gradient[:, nBPM]), nBPM]
                    rm[nBPM, i] = np.polyfit(x, y, fit_order)[fit_order - 1]
                    error[nBPM, i] = np.sqrt(np.mean((rm[nBPM, i] * x - y).T ** 2))
            i = i + 1
            SC.set_cm_setpoints(cm_ords[n_dim][nCM], cmstart[nCM], skewness=bool(n_dim))
    rm[np.isnan(rm)] = 0
    LOGGER.debug(' done.')
    return rm, error, cm_steps


def dispersion(SC, rf_step, bpm_ords=None, cav_ords=None, n_steps=2):
    bpm_ords, cav_ords = _check_ords(SC, bpm_ords, cav_ords)
    bref = np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0])
    if n_steps == 2:
        SC.set_cavity_setpoints(cav_ords, rf_step, 'Frequency', 'add')
        B = np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0])
        SC.set_cavity_setpoints(cav_ords, -rf_step, 'Frequency', 'add')
        return (B - bref) / rf_step
    rf_steps = np.zeros((len(cav_ords), n_steps))
    for n_cav, cav_ord in enumerate(cav_ords):
        rf_steps[n_cav, :] = SC.RING[cav_ord].FrequencySetPoint + np.linspace(-rf_step, rf_step, n_steps)
    dB = np.zeros((n_steps, *np.shape(bref)))
    rf0 = atgetfieldvalues(SC.RING, cav_ords, "FrequencySetPoint")
    for nStep in range(n_steps):
        SC.set_cavity_setpoints(cav_ords, rf_steps[:, nStep], 'Frequency', 'abs')
        dB[nStep, :] = np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0]) - bref
    SC.set_cavity_setpoints(cav_ords, rf0, 'Frequency', 'abs')
    return np.linalg.lstsq(np.linspace(-rf_step, rf_step, n_steps), dB)[0]


def _check_ords(SC, bpm_ords, cav_ords):
    if bpm_ords is None:
        bpm_ords = SC.ORD.BPM
    if cav_ords is None:
        cav_ords = SC.ORD.RF
    return bpm_ords, cav_ords


def _kick_amplitude(SC, bref, bpm_ords, cm_ord, amp, skewness: bool, mode):
    cmstart = SC.get_cm_setpoints(cm_ord, skewness)
    max_step = amp
    params = dict(fixedOffset=(4, 0.5), fixedKick=(20, 0.9))
    n_iter, decrease_factor = params[mode]
    for n in range(n_iter):
        SC, max_step, bpm_readings_ravel = _try_setpoint(SC, bpm_ords, cm_ord, cmstart, max_step, skewness)
        max_pos, max_pos_ref = np.sum(~np.isnan(bpm_readings_ravel)) / 2, np.sum(~np.isnan(bref)) / 2
        if max_pos_ref <= max_pos:
            if mode == "fixedOffset":
                max_step = max_step * amp / np.max(np.abs(bpm_readings_ravel - bref))
            else:
                break
        else:
            max_step *= decrease_factor
            LOGGER.debug(
                f'Insufficient beam reach ({max_pos:d}/{max_pos_ref:d}). '
                f'cm_step reduced to {1E6 * max_step:.1f}urad.')
    return max_step, np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0]) - bref


def _try_setpoint(SC, bpm_ords, cm_ord, cmstart, max_step, skewness):
    SC.set_cm_setpoints(cm_ord, cmstart + max_step, skewness)
    real_cm_setpoint = SC.get_cm_setpoints(cm_ord, skewness)
    if real_cm_setpoint != (cmstart + max_step):
        LOGGER.debug('CM  clipped. Using different CM direction.')
        max_step *= -1
        SC.set_cm_setpoints(cm_ord, cmstart + max_step, skewness)
    return SC, max_step, np.ravel(bpm_reading(SC, bpm_ords=bpm_ords)[0])
