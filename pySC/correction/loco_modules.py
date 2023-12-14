import at
import numpy as np
import multiprocessing
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.core.constants import SETTING_ADD
from pySC.core.beam import bpm_reading
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import linregress
from pySC.utils import logging_tools
LOGGER = logging_tools.get_logger(__name__)


def generating_jacobian(SC, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, trackMode='ORB',
                        useIdealRing=True, skewness=False, order=1, method='add', includeDispersion=False, rf_step=1E3,
                        cav_ords=None, full_jacobian=True):
    pool = multiprocessing.Pool()
    quad_args = [(quad_index, SC, C_model, dkick, used_cor_ind, bpm_indexes, dk, trackMode, useIdealRing,
                  skewness, order, method, includeDispersion, rf_step, cav_ords) for quad_index in quads_ind]
    results = pool.map(generating_quads_response_worker, quad_args)
    pool.close()
    pool.join()
    if full_jacobian:  # # Construct the complete Jacobian matrix for the LOCO
        length_corrections = len(np.concatenate(used_cor_ind))
        length_bpm = len(bpm_indexes) * 2
        return np.concatenate((results, np.tile(C_model, (length_corrections + length_bpm, 1, 1))))
    return results


def generating_quads_response_worker(args):
    return generating_quads_response_parallel(*args)


def generating_quads_response_parallel(quad_index, SC, C_model, correctrs_kick, used_cor_indexes, used_bpm_indexes, dk,
                                       useIdealRing, trackMode, skewness, order, method, includeDispersion, rf_step,
                                       cav_ords):
    LOGGER.debug('generating response to quad of index', quad_index)
    C0 = C_model
    if includeDispersion:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords)
        C0 = np.hstack((C0, dispersion_model.reshape(-1, 1)))

    C = quads_sensitivity_matrices(SC, correctrs_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk, trackMode,
                                   useIdealRing, skewness, order, method, includeDispersion, rf_step, cav_ords)
    return (C - C0) / dk


def quads_sensitivity_matrices(SC, correctors_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing,
                               trackMode, skewness, order, method, includeDispersion, rf_step, cav_ords):
    SC.set_magnet_setpoints(quad_index, dk, skewness, order, method)
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick=correctors_kick, useIdealRing=useIdealRing,
                              trackMode=trackMode)
    qx = C_measured
    if includeDispersion:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords, rfStep=rf_step)
        qx = np.hstack((qx, dispersion_model.reshape(-1, 1)))
    SC.set_magnet_setpoints(quad_index, -dk, skewness, order, method)
    return qx


def measure_closed_orbit_response_matrix(SC, bpm_ords, cm_ords, dkick=1e-5):
    LOGGER.info('Calculating Measure response matrix')
    n_turns = 1
    n_bpms = len(bpm_ords)
    n_cms = len(cm_ords[0]) + len(cm_ords[1])
    response_matrix = np.full((2 * n_bpms * n_turns, n_cms), np.nan)
    SC.INJ.trackMode = 'ORB'  # TODO modifies SC!

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


def loco_correction(objective_function, initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured,
                    J, Jt, lengths, including_fit_parameters, method='lm', verbose=2, max_iterations=1000, eps=1e-6,
                    W=1, show_plot=True):
    if method not in ("lm", "ng"):
        raise ValueError("Unsupported method only 'lm' or 'ng' are currently supported")
    if method == 'lm':
        result = least_squares(objective_function, initial_guess0, method=method, verbose=verbose)  # , xtol= 1e-2)
        return result.x

    for iter in range(max_iterations):
        model = orbit_response_matrix_model
        len_quads = lengths[0]
        len_corr = lengths[1]
        len_bpm = lengths[2]

        if 'quads' in including_fit_parameters:
            delta_g = initial_guess0[:len_quads]
            J1 = J[:len_quads]
            B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
            model += B

        if 'cor' in including_fit_parameters:
            delta_x = initial_guess0[len_quads:len_quads + len_corr]
            J2 = J[len_quads:len_quads + len_corr]
            # Co = orbit_response_matrix_model * delta_x
            Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
            model += Co

        if 'bpm' in including_fit_parameters:
            delta_y = initial_guess0[len_quads + len_corr:]
            J3 = J[len_quads + len_corr:]
            # G = orbit_response_matrix_model * delta_y[:, np.newaxis]
            G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)

            model += G

        r = orbit_response_matrix_measured - model

        t2 = np.zeros([len(initial_guess0), 1])
        for i in range(len(initial_guess0)):
            t2[i] = np.sum(np.dot(np.dot(J[i], W), r.T))

        t3 = (np.dot(Jt, t2)).reshape(-1)
        initial_guess1 = initial_guess0 + t3
        t4 = abs(initial_guess1 - initial_guess0)

        if max(t4) <= eps:
            break
        initial_guess0 = initial_guess1
        # e = np.dot(initial_guess0, J) - t2
    #  params_to_check = calculate_parameters(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,including_fit_parameters)
    return initial_guess0  # , params_to_check


def objective(delta_params, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,
              including_fit_parameters, W):
    D = orbit_response_matrix_measured - orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    residuals = D
    if 'quads' in including_fit_parameters:
        delta_g = delta_params[:len_quads]
        J1 = J[:len_quads]
        B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
        residuals -= B

    if 'cor' in including_fit_parameters:
        delta_x = delta_params[len_quads:len_quads + len_corr]
        J2 = J[len_quads:len_quads + len_corr]
        # Co = orbit_response_matrix_model * delta_x
        Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
        residuals -= Co

    if 'bpm' in including_fit_parameters:
        delta_y = delta_params[len_quads + len_corr:]
        J3 = J[len_quads + len_corr:]
        # G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)
        residuals -= G

    residuals = np.dot(residuals, np.sqrt(W))
    return residuals.ravel()


def calculate_parameters(parameters, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,
                         including_fit_parameters, W):
    model = orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        delta_g = parameters[:len_quads]
        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        model += B

    if 'cor' in including_fit_parameters:
        delta_x = parameters[len_quads:len_quads + len_corr]
        Co = orbit_response_matrix_model * delta_x
        model += Co

    if 'bpm' in including_fit_parameters:
        delta_y = parameters[len_quads + len_corr:]
        G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        model += G

    residuals = orbit_response_matrix_measured - model
    # Calculate R-squared
    r_squared = r2_score(orbit_response_matrix_measured, model)  # , sample_weight = 1)

    # Calculate RMSE
    rms = np.sqrt(mean_squared_error(orbit_response_matrix_measured, model))  # , model, sample_weight = 1)) #np.diag(W)

    params_to_check_ = {
        'residulas': residuals,
        'r_squared': r_squared,
        'rmse': rms,
    }
    return params_to_check_


def set_correction(SC, r, elem_ind, Individuals=True, skewness=False, order=1, method='add', dipole_compensation=True):
    if Individuals:
        for i in range(len(elem_ind)):
            field = elem_ind[i].SCFieldName
            # setpoint = fit_parameters.OrigValues[n_group] + damping * (
            #        fit_parameters.IdealValues[n_group] - fit_parameters.Values[n_group])
            if field == 'SetPointB':  # Normal quadrupole
                SC.set_magnet_setpoints(ord, -r[i], False, 1, dipole_compensation=dipole_compensation)
            elif field == 'SetPointA':  # Skew quadrupole
                SC.set_magnet_setpoints(ord, -r[i], True, 1)

            SC.set_magnet_setpoints(elem_ind[i], -r[i], skewness, order, method)
        return SC

    for quad_fam in range(len(elem_ind)):  # TODO this is strange
        for quad in quad_fam:
            field = elem_ind[quad].SCFieldName
            # setpoint = fit_parameters.OrigValues[n_group] + damping * (
            #        fit_parameters.IdealValues[n_group] - fit_parameters.Values[n_group])
            if field == 'SetPointB':  # Normal quadrupole
                SC.set_magnet_setpoints(ord, -r[quad], False, 1, dipole_compensation=dipole_compensation)
            elif field == 'SetPointA':  # Skew quadrupole
                SC.set_magnet_setpoints(ord, -r[quad], True, 1)

            SC.set_magnet_setpoints(elem_ind[quad], -r[quad], skewness, order, method)
    return SC


def set_correction_(SC, r, elem_ind, skewness=False, order=1, method='add', dipole_compensation=True):
    for i in range(len(elem_ind)):
        SC.set_magnet_setpoints(elem_ind[i], -r[i], skewness, order, method)
    return SC


def model_beta_beat(ring, twiss, elements_indexes, makeplot):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    bx = np.array((twiss_error.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0])
    by = np.array((twiss_error.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1])
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if makeplot:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, bx)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal beta')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, by)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical beta')
        plt.show()

    return bx_rms, by_rms


def select_equally_spaced_elements(total_elements, num_elements):
    step = len(total_elements) // (num_elements - 1)
    indexes = total_elements[::step]
    return indexes


def get_inverse(Jn, sCut, W):

    Nk = len(Jn)
    A = np.zeros([Nk, Nk])
    for i in range(Nk):
        for j in range(Nk):
            A[i, j] = np.sum(np.dot(np.dot(Jn[i], W), Jn[j].T))
    u, s, v = np.linalg.svd(A, full_matrices=True)
    plt.plot(np.log(s), 'd--')
    plt.title('singular value')
    plt.xlabel('singular values')
    plt.ylabel('$\log(\sigma_i)$')
    plt.show()

    smat = 0.0 * A
    si = s ** -1
    n_sv = sCut  # Cut off
    si[n_sv:] *= 0.0
    smat[:Nk, :Nk] = np.diag(si)
    return np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
