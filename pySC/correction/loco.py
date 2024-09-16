import at
import numpy as np
import multiprocessing
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.core.constants import SETTING_ADD, TRACK_ORB
from pySC.core.beam import bpm_reading
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pySC.utils import logging_tools, sc_tools, at_wrapper
from pySC.lattice_properties.response_model import SCgetModelRING,orbpass
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
    if full_jacobian:  # Construct the complete Jacobian matrix for the LOCO
        # assuming only linear scaling errors of BPMs and corrector magnets
        n_correctors = len(np.concatenate(used_cor_ind))
        n_bpms = len(bpm_indexes) * 2  # in both planes
        j_cor = np.zeros((n_correctors,) + C_model.shape)
        for i in range(n_correctors):
            j_cor[i, :, i] = C_model[:, i]   # a single column of response matrix corresponding to a corrector
        j_bpm = np.zeros((n_bpms,) + C_model.shape)
        for i in range(n_bpms):
            j_bpm[i, i, :] = C_model[i, :]  # a single row of response matrix corresponding to a given plane of BPM
        return np.concatenate((results, j_cor, j_bpm), axis=0)
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

    
    #dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords, rfStep=rf_step)
    _, _, twiss = at.get_optics(SC.IDEALRING, used_bpm_indexes)  ## ADD dispersion to the ORMs from AT get_optics
    dx = twiss.dispersion[:, 0]
    dy = twiss.dispersion[:, 2]
    dispersion_model=  np.column_stack((dx, dy))
    SC.set_magnet_setpoints(quad_index, dk, skewness, order, method)
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick=correctors_kick, useIdealRing=useIdealRing,
                              trackMode=trackMode)
    #dispersion_meas = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords, rfStep=rf_step, useIdealRing=False)
    _, _, twiss = at.get_optics(SC.RING, used_bpm_indexes)
    dx = twiss.dispersion[:, 0]
    dy = twiss.dispersion[:, 2]
    dispersion_meas =  np.column_stack((dx, dy))
    SC.set_magnet_setpoints(quad_index, -dk, skewness, order, method)
    return np.hstack((C_measured - C_model, ((dispersion_meas - dispersion_model)/correctors_kick).reshape(-1, 1)))


def measure_closed_orbit_response_matrix(SC, bpm_ords, cm_ords, dkick=1e-5, includeDispersion=False):
    LOGGER.info('Calculating Measure response matrix')
    n_turns = 1
    n_bpms = len(bpm_ords)
    n_cms = len(cm_ords[0]) + len(cm_ords[1])
    response_matrix = np.full((2 * n_bpms * n_turns, n_cms), np.nan)
    SC.INJ.trackMode = TRACK_ORB  # TODO may modify SC (not a pure function)!

    closed_orbits0 = bpm_reading(SC, bpm_ords=bpm_ords)[0]
    cnt = 0
    for n_dim in range(2):
        for cm_ord in cm_ords[n_dim]:
            SC.set_cm_setpoints(cm_ord, dkick, skewness=bool(n_dim), method=SETTING_ADD)
            closed_orbits1 = bpm_reading(SC, bpm_ords=bpm_ords)[0]
            SC.set_cm_setpoints(cm_ord, -dkick, skewness=bool(n_dim), method=SETTING_ADD)
            response_matrix[:, cnt] = np.ravel((closed_orbits1 - closed_orbits0) / dkick)
            cnt = cnt + 1
    if includeDispersion == True:
        _, _, twiss = at.get_optics(SC.RING, bpm_ords)
        dx = twiss.dispersion[:, 0]
        dy = twiss.dispersion[:, 2]
        dispersion_meas = np.column_stack((dx, dy))
        return np.hstack(
            (response_matrix, ((dispersion_meas) / dkick).reshape(-1, 1)))      
    return response_matrix


def loco_correction_lm(initial_guess0, orm_model, orm_measured, Jn, lengths, including_fit_parameters, bounds=(-np.inf, np.inf), weights=1,
                       verbose=2):
    mask = _get_parameters_mask2(including_fit_parameters, lengths)
    result = least_squares(lambda delta_params: objective(delta_params, orm_measured - orm_model, Jn[mask, :, :], weights),
                           initial_guess0[mask], #bounds=bounds,
                           method="lm",
                           verbose=verbose)
    return result.x


def loco_correction_ng(initial_guess0, orm_model, orm_measured, J, lengths, including_fit_parameters, s_cut, weights=1, includeDispersion=False):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask2(including_fit_parameters, lengths)
    residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
    r = residuals.reshape(np.transpose(orm_model).shape)
    t2 = np.zeros([len(initial_guess[mask]), 1])
    for i in range(len(initial_guess[mask])):
        t2[i] = np.sum(np.dot(np.dot(J[i].T, weights), r.T))
    return get_inverse(J[mask, :, :], t2, s_cut, weights)


def objective(masked_params, orm_residuals, masked_jacobian, weights):
    return np.dot(np.transpose(orm_residuals - np.einsum("ijk,i->jk", masked_jacobian, masked_params)),
                  np.sqrt(weights)).ravel()


def _get_parameters_mask(including_fit_parameters, lengths):
    len_quads, len_corr, len_bpm = lengths
    mask = np.zeros(len_quads + len_corr + len_bpm, dtype=bool)
    mask[:len_quads] = 'quads' in including_fit_parameters
    mask[len_quads:len_quads + len_corr] = 'cor' in including_fit_parameters
    mask[len_quads + len_corr:] = 'bpm' in including_fit_parameters
    return mask

def _get_parameters_mask2(including_fit_parameters, lengths):
    mask = np.zeros(sum(lengths), dtype=bool)
    current_index = 0
    for param, length in zip(including_fit_parameters, lengths):
        mask[current_index:current_index + length] = True
        current_index += length
    return mask


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


def get_inverse(jacobian, B, s_cut, weights, plot=False):
    n_resp_mats = len(jacobian)
    sum_corr = np.sum(jacobian, axis=2)          # Sum over i and j for all planes
    matrix = np.dot(np.dot(sum_corr, weights), sum_corr.T)
    inv_matrix = sc_tools.pinv(matrix, num_removed=n_resp_mats - min(n_resp_mats, s_cut), plot=plot)
    results = np.ravel(np.dot(inv_matrix, B))
    # e = np.ravel(np.dot(matrix, results)) - np.ravel(B)
    return results




'''
The code below is for optics calculations (based on AT get_optics) and generates plots for analyzing the lattice.
 Example usage:

 [-, -, twiss] = at.get_optics(SC.IDEALRING, SC.ORD.BPM)
 analyze_ring(SC, twiss, SC.ORD.BPM, makeplot=False)
'''

def analyze_ring(SC, twiss, bpm_indices, useIdealRing=True, makeplot=False):
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SC.RING #SCgetModelRING(SC)
    rmsx, rmsy = rms_orbits(ring, bpm_indices)
    bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices)
    dx_rms_err, dy_rms_err = getDispersionErr(ring, twiss, bpm_indices)

    print(f"RMS horizontal orbit: {rmsx * 1.e6:.2f} µm, RMS vertical orbit: {rmsy * 1.e6:.2f} µm")
    print(f"RMS horizontal beta beating: {bx_rms_err * 100:.2f}%, RMS vertical beta beating: {by_rms_err * 100:.2f}%")
    print(f"RMS relative horizontal dispersion: {dx_rms_err:.4f} mm, RMS relative vertical dispersion: {dy_rms_err:.4f} mm")
    print(f"Tune values: {at.get_tune(ring, get_integer=True)}, Chromaticity values: {at.get_chrom(ring)}")

    if makeplot:
        plot_orbits(ring, bpm_indices)
        plot_beta_beat(ring, twiss, bpm_indices)
        plot_dispersion_err(ring, twiss, bpm_indices)

def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2))

def plot_data(s_pos, data, xlabel, ylabel, title):
    plt.rc('font', size=13)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(s_pos, data)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, which='both', linestyle=':', color='gray')
    plt.title(title)
    plt.show()

def rms_orbits(ring, elements_indexes, trackMode='ORB', Z0=np.zeros(6)):
    track_methods = dict(TBT=at_wrapper.atpass, ORB=orbpass)
    if trackMode == 'ORB':
        nTurns = 1
    trackmethod = track_methods['ORB']
    Ta = trackmethod(ring, Z0=Z0, nTurns=nTurns, REFPTS=elements_indexes)

    closed_orbitx = np.ravel(np.transpose(Ta[0, :, :, :], axes=(2, 1, 0)))
    closed_orbity = np.ravel(np.transpose(Ta[2, :, :, :], axes=(2, 1, 0)))

    rmsx = calculate_rms(closed_orbitx)
    rmsy = calculate_rms(closed_orbity)

    return rmsx, rmsy

def getBetaBeat(ring, twiss, elements_indexes):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    bx = (twiss_error.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0]
    by = (twiss_error.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1]

    bx_rms = calculate_rms(bx)
    by_rms = calculate_rms(by)

    return bx_rms, by_rms

def getDispersionErr(ring, twiss, elements_indexes):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    dx = twiss_error.dispersion[:, 0] - twiss.dispersion[:, 0]
    dy = twiss_error.dispersion[:, 2] - twiss.dispersion[:, 2]

    dx_rms = calculate_rms(dx)
    dy_rms = calculate_rms(dy)

    return dx_rms, dy_rms

def plot_orbits(ring, elements_indexes, trackMode='ORB', Z0=np.zeros(6)):
    _, _, twiss = at.get_optics(ring, elements_indexes)
    track_methods = dict(TBT=at_wrapper.atpass, ORB=orbpass)
    if trackMode == 'ORB':
       nTurns = 1
    trackmethod = track_methods['ORB']
    Ta = trackmethod(ring, Z0=Z0, nTurns=nTurns, REFPTS=elements_indexes)
    closed_orbitx = np.ravel(np.transpose(Ta[0, :, :, :], axes=(2, 1, 0))) / 1.e-06
    closed_orbity = np.ravel(np.transpose(Ta[2, :, :, :], axes=(2, 1, 0))) / 1.e-06
    plot_data(twiss.s_pos, closed_orbitx, "s_pos [m]", r"closed_orbit x [$\mu$m]", "Horizontal closed orbit")
    plot_data(twiss.s_pos, closed_orbity, "s_pos [m]", r"closed_orbit y [$\mu$m]", "Vertical closed orbit")

def plot_beta_beat(ring, twiss, elements_indexes):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    bx = (twiss_error.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0]
    by = (twiss_error.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1]

    plot_data(twiss_error.s_pos, bx, "s_pos [m]", r'$\Delta \beta_x / \beta_x$', "Horizontal beta beating")
    plot_data(twiss_error.s_pos, by, "s_pos [m]", r'$\Delta \beta_y / \beta_y$', "Vertical beta beating")

def plot_dispersion_err(ring, twiss, elements_indexes):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    dx = twiss_error.dispersion[:, 0] - twiss.dispersion[:, 0]
    dy = twiss_error.dispersion[:, 2] - twiss.dispersion[:, 2]

    plot_data(twiss_error.s_pos, dx, "s_pos [m]", r'$\Delta \eta_x$ [m]', "Horizontal dispersion error")
    plot_data(twiss_error.s_pos, dy, "s_pos [m]", r'$\Delta  \eta_y$ [m]', "Vertical dispersion error")
