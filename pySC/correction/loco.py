import at
import numpy as np
import multiprocessing
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.core.constants import SETTING_ADD, TRACK_ORB
from pySC.core.beam import bpm_reading
from pySC.utils.sc_tools import SCgetPinv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pySC.utils import logging_tools
LOGGER = logging_tools.get_logger(__name__)


def calculate_jacobian(SC, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, trackMode=TRACK_ORB,
                       useIdealRing=True, skewness=False, order=1, method=SETTING_ADD, includeDispersion=False, rf_step=1E3,
                       cav_ords=None, full_jacobian=True):
    pool = multiprocessing.Pool()
    quad_args = [(quad_index, SC, C_model, dkick, used_cor_ind, bpm_indexes, dk, trackMode, useIdealRing,
                  skewness, order, method, includeDispersion, rf_step, cav_ords) for quad_index in quads_ind]
    # results = []
    # for quad_arg in quad_args:
    #     results.append(generating_quads_response_matrices(quad_arg))
    results = pool.map(generating_quads_response_matrices, quad_args)
    pool.close()
    pool.join()

    results = [result / dk for result in results]
    if full_jacobian:  # # Construct the complete Jacobian matrix for the LOCO
        # TODO modify for calibration errors of given size
        n_correctors = len(np.concatenate(used_cor_ind))
        n_bpms = len(bpm_indexes) * 2  # in both planes
        return np.concatenate((results, np.tile(C_model, (n_correctors + n_bpms, 1, 1))))
    return results


def generating_quads_response_matrices(args):
    (quad_index, SC, C_model, correctors_kick, used_cor_indexes, used_bpm_indexes, dk, trackMode, useIdealRing,
     skewness, order, method, includeDispersion, rf_step, cav_ords) = args
    LOGGER.debug('generating response to quad of index', quad_index)
    if not includeDispersion:
        SC.set_magnet_setpoints(quad_index, dk, skewness, order, method)
        C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick=correctors_kick,
                                  useIdealRing=useIdealRing,
                                  trackMode=trackMode)
        SC.set_magnet_setpoints(quad_index, -dk, skewness, order, method)
        return C_measured - C_model

    dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords)
    SC.set_magnet_setpoints(quad_index, dk, skewness, order, method)
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick=correctors_kick, useIdealRing=useIdealRing,
                              trackMode=trackMode)
    dispersion_meas = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords, rfStep=rf_step)
    SC.set_magnet_setpoints(quad_index, -dk, skewness, order, method)
    return np.hstack((C_measured - C_model, (dispersion_meas - dispersion_model).reshape(-1, 1)))


def measure_closed_orbit_response_matrix(SC, bpm_ords, cm_ords, dkick=1e-5):
    LOGGER.info('Calculating Measure response matrix')
    n_turns = 1
    n_bpms = len(bpm_ords)
    n_cms = len(cm_ords[0]) + len(cm_ords[1])
    response_matrix = np.full((2 * n_bpms * n_turns, n_cms), np.nan)
    SC.INJ.trackMode = 'ORB'  # TODO modifies SC (not a pure function)!

    closed_orbits0 = bpm_reading(SC, bpm_ords=bpm_ords)[0]
    cnt = 0
    for n_dim in range(2):
        for cm_ord in cm_ords[n_dim]:
            SC.set_cm_setpoints(cm_ord, dkick, skewness=bool(n_dim), method=SETTING_ADD)
            closed_orbits1 = bpm_reading(SC, bpm_ords=bpm_ords)[0]
            SC.set_cm_setpoints(cm_ord, -dkick, skewness=bool(n_dim), method=SETTING_ADD)
            response_matrix[:, cnt] = np.ravel((closed_orbits1 - closed_orbits0) / dkick)
            cnt = cnt + 1
    return response_matrix


def loco_correction_lm(initial_guess0, orm_model, orm_measured, Jn, lengths, including_fit_parameters, bounds=(-np.inf, np.inf), weights=1,
                       verbose=2):
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    result = least_squares(lambda delta_params: objective2(delta_params, orm_measured - orm_model, Jn[mask, :, :], weights),
                           initial_guess0[mask], #bounds=bounds,
                           method="lm",
                           verbose=verbose)  # , xtol= 1e-2)
    return result.x


def loco_correction_ng(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, weights=1,
                       max_iterations=1000, eps=1e-6):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    for _ in range(max_iterations):
        residuals = objective(initial_guess, orm_model, orm_measured, J, lengths, including_fit_parameters, 1)
        r = residuals.reshape(orm_model.shape)

        t2 = np.zeros([len(initial_guess), 1])
        for i in range(len(initial_guess)):
            t2[i] = np.sum(np.dot(np.dot(J[i], weights), r.T))

        t3 = (np.dot(Jt, t2)).reshape(-1)
        initial_guess1 = initial_guess + t3  # TODO check the sign
        if np.max(np.abs(t3)) <= eps:
            return initial_guess
        initial_guess = initial_guess1
    return initial_guess


def objective(delta_params, orm_model, orm_measured, J, lengths, including_fit_parameters, weights):
    # This function is already tested
    len_quads, len_corr, len_bpm = lengths
    mask = np.zeros(delta_params.shape)
    if 'quads' in including_fit_parameters:
        mask[:len_quads] = 1
    if 'cor' in including_fit_parameters:
        mask[len_quads:len_quads + len_corr] = 1
    if 'bpm' in including_fit_parameters:
        mask[len_quads + len_corr:] = 1
    residuals = orm_measured - orm_model - np.einsum("ijk,i->jk", J, delta_params * mask)
    residuals = np.dot(residuals, np.sqrt(weights))
    return residuals.ravel()


def _get_parameters_mask(including_fit_parameters, lengths):
    len_quads, len_corr, len_bpm = lengths
    mask = np.zeros(len_quads + len_corr + len_bpm, dtype=bool)
    mask[:len_quads] = 'quads' in including_fit_parameters
    mask[len_quads:len_quads + len_corr] = 'cor' in including_fit_parameters
    mask[len_quads + len_corr:] = 'bpm' in including_fit_parameters
    return mask


def objective2(masked_params, orm_residuals, masked_jacobian, weights):
    return np.dot(orm_residuals - np.einsum("ijk,i->jk", masked_jacobian, masked_params), np.sqrt(weights)).ravel()


def set_correction(SC, r, elem_ind, individuals=True, skewness=False, order=1, method=SETTING_ADD, dipole_compensation=True):
    if individuals:
        SC.set_magnet_setpoints(elem_ind, -r, skewness, order, method, dipole_compensation=dipole_compensation)
        return SC

    for fam_num, quad_fam in enumerate(elem_ind):
        SC.set_magnet_setpoints(quad_fam, -r[fam_num], skewness, order, method, dipole_compensation=dipole_compensation)
    return SC


def model_beta_beat(ring, twiss, elements_indexes, plot=False):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    bx = np.array(twiss_error.beta[:, 0] / twiss.beta[:, 0] - 1)
    by = np.array(twiss_error.beta[:, 1] / twiss.beta[:, 1] - 1)
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if plot:
        init_font = plt.rcParams["font.size"]
        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(nrows=2, sharex="all")
        betas = [bx, by]
        letters = ("x", "y")
        for i in range(2):
            ax[i].plot(s_pos, betas[i])
            ax[i].set_xlabel("s_pos [m]")
            ax[i].set_ylabel(rf'$\Delta\beta_{letters[i]}$ / $\beta_{letters[i]}$')
            ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[i].grid(True, which='both', linestyle=':', color='gray')

        fig.show()
        plt.rcParams.update({'font.size': init_font})

    return bx_rms, by_rms


def select_equally_spaced_elements(total_elements, num_elements):
    step = len(total_elements) // (num_elements - 1)
    return total_elements[::step]


def get_inverse(jacobian, s_cut, weights):
    n_resp_mats = len(jacobian)
    #matrix = np.zeros([n_resp_mats, n_resp_mats])
    #for i in range(n_resp_mats):
    #    for j in range(n_resp_mats):
    #        matrix[i, j] = np.sum(np.dot(np.dot(jacobian[i], weights), jacobian[j].T))
    sum_ = np.sum(jacobian, axis=1)          # Sum over i and j for all planes
    matrix = sum_ @ weights @ sum_.T
    return SCgetPinv(matrix, num_removed=n_resp_mats - min(n_resp_mats, s_cut), plot=True)
