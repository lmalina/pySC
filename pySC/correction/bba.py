import matplotlib.pyplot as plt
import numpy as np
import copy
from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.correction.orbit_trajectory import SCfeedbackRun
from pySC.core.beam import bpm_reading, all_elements_reading
from pySC.utils.sc_tools import SCrandnc
from pySC.core.lattice_setting import set_cm_setpoints, set_magnet_setpoints, get_cm_setpoints
from pySC.utils import logging_tools
from pySC.core.classes import DotDict
from pySC.core.constants import TRACKING_MODES, TRACK_TBT, NUM_TO_AB

LOGGER = logging_tools.get_logger(__name__)

"""
Pieces of found function calls from ALS-U SR, i.e. nothing in PETRA IV

qOrd = SCgetOrds(SC.RING,'QF')
SC,errorFlags = SCBBA(SC, repmat(SC.ORD.BPM,2,1),QuadOrds, mode='TBT', fakeMeasForFailures=True, 
        quadOrdPhaseAdvance=qOrd[0], quadStrengthPhaseAdvance=[0.95 0.8 1.05], magSPvec=magSPvec, plotResults=SC.plot)

SC = SCBBA(SC, BPMordsQuadBBA, QuadOrds, mode='ORB',fakeMeasForFailures=True, outlierRejectionAt=200E-6, RMstruct=RMstruct,
        plotResults=SC.plot, magSPvec=magSPvec, minSlopeForFit=0.005, minBPMrangeAtBBABBPM=1*20E-6,
        BBABPMtarget=5*50E-6, minBPMrangeOtherBPM=0)

# Orbit correction (without weights)
SC = performOrbitCorr_ALSU_SR(SC,RMstruct,'weight=[])

# BBA on sextupoles (quad trim coils)
SC = SCBBA(SC, BPMordsSextBBA, sextOrds, mode='ORB',fakeMeasForFailures=True, outlierRejectionAt=200E-6,
    RMstruct=RMstruct, plotResults=SC.plot, switch_off_sextupoles=True, setpoint_method='abs', magSPvec=magSPvecSext, 
    minSlopeForFit=0.005, minBPMrangeAtBBABBPM=1*10E-6, BBABPMtarget=5*50E-6, minBPMrangeOtherBPM=0)
"""


def bba(SC, bpm_ords, mag_ords, **kwargs):
    par = DotDict(dict(mode=SC.INJ.trackMode, outlierRejectionAt=np.inf, nSteps=10, fit_order=1, magnet_order=1,
                       magSPvec=np.array([0.95, 1.05]), setpoint_method='rel', RMstruct=[], orbBumpWindow=5, BBABPMtarget=1E-3,
                       minBPMrangeAtBBABBPM=500E-6, minBPMrangeOtherBPM=100E-6, maxStdForFittedCenters=600E-6,
                       nXPointsNeededAtMeasBPM=3, maxNumOfDownstreamBPMs=len(SC.ORD.BPM), minSlopeForFit=0.03,
                       maxTrajChangeAtInjection=np.array([0.9E-3, 0.9E-3]), quadOrdPhaseAdvance=np.array([8]),
                       quadStrengthPhaseAdvance=np.array([0.95, 1.05]), fakeMeasForFailures=False, dipole_compensation=True,
                       skewness=False, switch_off_sextupoles=False, useBPMreadingsForOrbBumpRef=False, plotResults=False))
    par.update(**kwargs)
    if bpm_ords.shape != mag_ords.shape:  # both in shape 2 x N
        raise ValueError('Input arrays for BPMs and magnets must be same size.')
    if par.magSPvec.ndim < 2:
        par.magSPvec = np.tile(par.magSPvec, mag_ords.shape + (1,))
    if par.mode not in TRACKING_MODES:
        raise ValueError(f"Unknown mode {par.mode}.")
    if par.mode == TRACK_TBT and SC.INJ.nTurns != 2:
        raise ValueError('Beam-based alignment in TBT mode works with 2 turns. Please set: SC.INJ.nTurns = 2')
    init_ring = SC.RING.deepcopy()

    init_offset_errors = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
    error_flags = np.full(bpm_ords.shape, np.nan)
    offset_changes = np.full(bpm_ords.shape, np.nan)
    if par.mode == TRACK_TBT:
        quad_k, scalings, bpm_ranges = [], [], []
        for n_dim in range(bpm_ords.shape[0]):
            last_bpm_ind = np.where(bpm_ords[n_dim, -1] == SC.ORD.BPM)[0][0]
            quads_strengths, all_scalings, all_bpm_ranges = single_phase_advance_injection_scan(SC, n_dim, last_bpm_ind, par)
            quad_k.append(quads_strengths)
            scalings.append(all_scalings)
            bpm_ranges.append(all_bpm_ranges)
    kick_vec0 = par.maxTrajChangeAtInjection.reshape(2, 1) * np.linspace(-1, 1, par.nSteps)
    q0 = SC.RING[par.quadOrdPhaseAdvance].SetPointB[1]
    for j_bpm in range(bpm_ords.shape[1]):  # j_bpm: Index of BPM adjacent to magnet for BBA
        LOGGER.info(f"BPM number {j_bpm}")
        for n_dim in range(bpm_ords.shape[0]):
            LOGGER.debug(f'BBA-BPM {j_bpm}/{bpm_ords.shape[1]}, n_dim = {n_dim}')
            bpm_ind = np.where(bpm_ords[n_dim, j_bpm] == SC.ORD.BPM)[0][0]
            m_ord = mag_ords[n_dim, j_bpm]
            if par.switch_off_sextupoles:
                SC0 = _very_deep_copy(SC)
                SC = set_magnet_setpoints(SC, m_ord, setpoints=np.zeros(1), skewness=False, order=2,  method='abs')
                SC = SCfeedbackRun(SC, par.RMstruct.MinvCO, BPMords=par.RMstruct.BPMords, CMords=par.RMstruct.CMords,
                                   target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, eps=1E-6)
            if par.mode == 'ORB':
                # TODO very deep copy
                bpm_pos, tmpTra = _data_measurement_orb(SC, m_ord, bpm_ind, j_bpm, n_dim, par,
                                                   *_get_orbit_bump(SC, m_ord, bpm_ords[n_dim, j_bpm], n_dim, par))
            else:
                set_ind = np.argmax(bpm_ranges[n_dim][:, bpm_ind])
                if bpm_ranges[n_dim][set_ind, bpm_ind] < par.BBABPMtarget:
                    LOGGER.warning(f"Small range at BPM[{n_dim},{j_bpm}]")

                SC = set_magnet_setpoints(SC, par.quadOrdPhaseAdvance, quad_k[n_dim][set_ind], False, 1, method='rel', dipole_compensation=True)
                kick_vec = scalings[n_dim][set_ind] * par.maxTrajChangeAtInjection.reshape(2, 1) * np.linspace(-1, 1, par.nSteps)
                bpm_pos, tmpTra = _data_measurement_tbt(SC, m_ord, bpm_ind, j_bpm, n_dim, par, kick_vec)
                SC = set_magnet_setpoints(SC, par.quadOrdPhaseAdvance, q0, False, 1, method='abs', dipole_compensation=True)

            offset_changes[n_dim, j_bpm], error_flags[n_dim, j_bpm] = _data_evaluation(SC, bpm_pos, tmpTra, n_dim, m_ord, par)
            if par.switch_off_sextupoles:
                SC = SC0
        #     if not np.isnan(offset_change):
        #         SC.RING[bpm_ords[n_dim, j_bpm]].Offset[n_dim] += offset_change
        # if par.plotResults:
        #     plot_bba_results(SC, init_offset_errors, error_flags, j_bpm, bpm_ords, mag_ords)

    for j_bpm in range(bpm_ords.shape[1]):  # j_bpm: Index of BPM adjacent to magnet for BBA
        for n_dim in range(bpm_ords.shape[0]):
            if not np.isnan(offset_changes[n_dim, j_bpm]):
                SC.RING[bpm_ords[n_dim, j_bpm]].Offset[n_dim] += offset_changes[n_dim, j_bpm]
    if par.plotResults:
        plot_bba_results(SC, init_offset_errors, error_flags, bpm_ords, mag_ords)
    if par.fakeMeasForFailures:
        SC = _fake_measurement(SC, bpm_ords, mag_ords, error_flags)
    return SC, error_flags, offset_changes


def _very_deep_copy(SC):
    new = copy.deepcopy(SC)
    new.RING = SC.RING.deepcopy()
    new.IDEALRING = SC.IDEALRING.deepcopy()
    for ind, element in enumerate(SC.RING):
        new.RING[ind] = element.deepcopy()
        new.IDEALRING[ind] = element.deepcopy()
    return new


def _get_bpm_offset_from_mag(ring, bpm_ords, mag_ords):
    offset = np.full(bpm_ords.shape, np.nan)
    for n_dim in range(2):
        offset[n_dim, :] = (atgetfieldvalues(ring, bpm_ords[n_dim, :], 'Offset', n_dim)
                            + atgetfieldvalues(ring, bpm_ords[n_dim, :], 'SupportOffset', n_dim)
                            - atgetfieldvalues(ring, mag_ords[n_dim, :], 'MagnetOffset', n_dim)
                            - atgetfieldvalues(ring, mag_ords[n_dim, :], 'SupportOffset', n_dim))
    return offset


def _fake_measurement(SC, bpm_ords, mag_ords, error_flags):
    final_offset_errors = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
    final_offset_errors[error_flags != 0] = np.nan
    LOGGER.info(f"Final offset error is {1E6 * np.sqrt(np.nanmean(final_offset_errors ** 2, axis=1))} um (hor|ver)"
                f" with {np.sum(error_flags != 0, axis=1)} measurement failures -> being re-calculated now.\n")

    for nBPM in range(bpm_ords.shape[1]):
        for nDim in range(2):
            if error_flags[nDim, nBPM] != 0:
                fake_bpm_offset = (SC.RING[mag_ords[nDim, nBPM]].MagnetOffset[nDim]
                                   + SC.RING[mag_ords[nDim, nBPM]].SupportOffset[nDim]
                                   - SC.RING[bpm_ords[nDim, nBPM]].SupportOffset[nDim]
                                   + np.sqrt(np.nanmean(np.square(final_offset_errors[nDim, :]))) * SCrandnc(2))
                if not np.isnan(fake_bpm_offset):
                    SC.RING[bpm_ords[nDim, nBPM]].Offset[nDim] = fake_bpm_offset
                else:
                    LOGGER.info('BPM offset not reassigned, NaN.\n')
    return SC


def _data_measurement_tbt(SC, m_ord, bpm_ind, j_bpm, n_dim, par, kick_vec):
    measDim = 1 - n_dim if par.skewness else n_dim
    initialZ0 = SC.INJ.Z0.copy()
    nMsteps = kick_vec.shape[1]
    tmpTra = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm]), par.maxNumOfDownstreamBPMs), np.nan)
    BPMpos = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm])), np.nan)
    init_setpoint = getattr(SC.RING[m_ord], f"SetPoint{NUM_TO_AB[int(par.skewness)]}")[par.magnet_order]
    for nQ, setpointQ in enumerate(par.magSPvec[n_dim, j_bpm]):
        SC = set_magnet_setpoints(SC, m_ord, setpointQ, par.skewness, par.magnet_order,
                                  method=par.setpoint_method, dipole_compensation=par.dipole_compensation)
        for nKick in range(nMsteps):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initialZ0[2 * n_dim:2 * n_dim + 2] + kick_vec[:, nKick]
            B = bpm_reading(SC)
            BPMpos[nKick, nQ] = B[n_dim, bpm_ind]
            tmpTra[nKick, nQ, :] = B[measDim, bpm_ind:(bpm_ind + par.maxNumOfDownstreamBPMs)]

    SC.INJ.Z0 = initialZ0
    SC = set_magnet_setpoints(SC, m_ord, init_setpoint, par.skewness, par.magnet_order,
                              method="abs", dipole_compensation=par.dipole_compensation)
    return BPMpos, tmpTra


def _data_measurement_orb(SC, mOrd, BPMind, j_bpm, n_dim, par, CMords, cm_vec):
    meas_dim = 1 - n_dim if par.skewness else n_dim
    initial_z0 = SC.INJ.Z0.copy()
    nMsteps = cm_vec[n_dim].shape[0]
    tmpTra = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm]), len(SC.ORD.BPM)), np.nan)
    BPMpos = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm])), np.nan)
    for nQ, setpointQ in enumerate(par.magSPvec[n_dim, j_bpm]):
        SC = set_magnet_setpoints(SC, mOrd, setpointQ, par.skewness, par.magnet_order,  method=par.setpoint_method,
                                  dipole_compensation=par.dipole_compensation)
        for nKick in range(nMsteps):
            for nD in range(2):
                SC = set_cm_setpoints(SC, CMords[nD], cm_vec[nD][nKick, :], bool(nD), method='abs')
            B = bpm_reading(SC)
            BPMpos[nKick, nQ] = B[n_dim, BPMind]
            tmpTra[nKick, nQ, :] = B[meas_dim, :]

    SC.INJ.Z0 = initial_z0
    return BPMpos, tmpTra


def _data_evaluation(SC, bpm_pos, tmpTra, n_dim, m_ord, par):
    offset_change = np.nan
    Error = 5
    tmpCenter = np.full(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs), np.nan)
    tmpNorm = np.full(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs), np.nan)
    tmpRangeX = np.zeros(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs))
    tmpRangeY = np.zeros(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs))
    i = 0
    for nBPM in range(par.maxNumOfDownstreamBPMs):
        y0 = np.diff(tmpTra[:, :, nBPM], 1, axis=1)
        y0 = y0[:, np.sum(~np.isnan(y0), axis=0) > 0]
        x0 = np.tile(np.mean(bpm_pos, 1), (y0.shape[1], 1)).T
        for nKick in range(y0.shape[1]):
            i = i + 1
            y = y0[:, nKick]
            x = x0[:, nKick]
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            # if len(x) == 0 or len(y) == 0:
            #     continue
            tmpRangeX[i] = abs(np.min(x) - np.max(x))
            tmpRangeY[i] = abs(np.min(y) - np.max(y))
            sol = np.full((2,), np.nan)
            if len(x) >= par.nXPointsNeededAtMeasBPM and tmpRangeX[i] > par.minBPMrangeAtBBABBPM and tmpRangeY[i] > par.minBPMrangeOtherBPM:
                if par.fit_order == 1:
                    sol = np.linalg.lstsq(np.vstack((np.ones(x.shape), x)).T, y)[0]
                    sol = sol[[1, 0]]
                    if abs(sol[0]) < par.minSlopeForFit:
                        sol[0] = np.nan
                    tmpCenter[i] = -sol[1] / sol[0]
                    tmpNorm[i] = 1 / np.sqrt(np.sum((sol[0] * x + sol[1] - y) ** 2))
                else:
                    sol = np.polyfit(x, y, par.fit_order)
                    if par.fit_order == 2:
                        tmpCenter[i] = - (sol[1] / (2 * sol[0]))
                    else:
                        tmpCenter[i] = min(abs(np.roots(sol)))
                    tmpNorm[i] = 1 / np.linalg.norm(np.polyval(sol, x) - y)

    if np.max(tmpRangeX) < par.minBPMrangeAtBBABBPM:
        Error = 1
    elif np.max(tmpRangeY) < par.minBPMrangeOtherBPM:
        Error = 2
    elif np.std(tmpCenter, ddof=1) > par.maxStdForFittedCenters:
        Error = 3
    elif len(np.where(~np.isnan(tmpCenter))[0]) == 0:
        Error = 4
    else:
        offset_change = np.nansum(tmpCenter * tmpNorm) / np.nansum(tmpNorm)
        Error = 0
    if not par.dipole_compensation and n_dim == 1 and SC.RING[m_ord].NomPolynomB[1] != 0:
        B = getattr(SC.RING[m_ord], 'BendingAngle', 0)
        K = SC.RING[m_ord].NomPolynomB[1]
        L = SC.RING[m_ord].Length
        offset_change = offset_change + B / L / K
    if offset_change > par.outlierRejectionAt:
        offset_change = np.nan
        Error = 6
    return offset_change, Error


def single_phase_advance_injection_scan(SC, n_dim, last_bpm_ind, par):

    scaling, bpm_ranges = _scale_injection_to_reach_bpms(SC, n_dim, last_bpm_ind, par)
    if not par.quadOrdPhaseAdvance:
        return np.ones(1), np.array([scaling]), bpm_ranges[np.newaxis, :]
    q_ord = par.quadOrdPhaseAdvance
    q_vec = par.quadStrengthPhaseAdvance
    q0 = SC.RING[q_ord].SetPointB[1]
    all_bpm_ranges = np.zeros((len(q_vec) + 1, len(SC.ORD.BPM)))
    all_scalings = np.zeros(len(q_vec) + 1)
    quads_strengths = np.concatenate((np.ones(1), q_vec))
    all_bpm_ranges[0, :] = bpm_ranges
    all_scalings[0] = scaling
    for nQ in range(len(q_vec)):
        SC = set_magnet_setpoints(SC, q_ord, q_vec[nQ], False, 1, method='rel', dipole_compensation=True)
        all_scalings[nQ + 1], all_bpm_ranges[nQ + 1] = _scale_injection_to_reach_bpms(SC, n_dim, last_bpm_ind, par)
    SC = set_magnet_setpoints(SC, q_ord, q0, False, 1, method='abs', dipole_compensation=True)
    return quads_strengths, all_scalings, all_bpm_ranges


def _scale_injection_to_reach_bpms(SC, n_dim, last_bpm_ind, par):
    n_steps = min(par.nSteps, 10)
    kick_vec0 = par.maxTrajChangeAtInjection.reshape(2, 1) * np.linspace(-1, 1, n_steps)
    initial_z0 = SC.INJ.Z0.copy()
    initial_nturns = SC.INJ.nTurns
    SC.INJ.nTurns = 1
    scaling_factor = 1.0
    mask = np.ones(len(SC.ORD.BPM), dtype=bool)
    if last_bpm_ind + 1 < len(SC.ORD.BPM):
        mask[last_bpm_ind + 1:] = False
    for scaling_factor in (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1): #for _ in range(6):
        tmp_bpm_pos = np.full((par.nSteps, len(SC.ORD.BPM)), np.nan)

        for nK in range(par.nSteps):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initial_z0[2 * n_dim:2 * n_dim + 2] + scaling_factor * kick_vec0[:, nK]
            tmp_bpm_pos[nK, :] = bpm_reading(SC)[n_dim, :]
        SC.INJ.Z0 = initial_z0.copy()

        if np.sum(np.isnan(tmp_bpm_pos[:, mask])) == 0:
            bpm_ranges = np.max(tmp_bpm_pos, axis=0) - np.min(tmp_bpm_pos, axis=0)
            LOGGER.debug(f'Initial trajectory variation scaled to [{100 * scaling_factor}| '
                         f'{100 * scaling_factor}]% of its initial value, '
                         f'BBA-BPM range from {1E6 * np.min(bpm_ranges):.0f} um to {1E6 * np.max(bpm_ranges):.0f}.')
            SC.INJ.nTurns = initial_nturns
            return scaling_factor, bpm_ranges
        #scaling_factor *= 0.8
    else:
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
        SC.INJ.nTurns = initial_nturns
        return scaling_factor, np.zeros(len(SC.ORD.BPM))

def _get_orbit_bump(SC, cm_ord, bpm_ord, n_dim, par):
    tmpCMind = np.where(par.RMstruct.CMords[0] == cm_ord)[0]
    if len(tmpCMind):
        par.RMstruct.RM = np.delete(par.RMstruct.RM, tmpCMind, 1)  # TODO not nice
        par.RMstruct.CMords[0] = np.delete(par.RMstruct.CMords[0], tmpCMind)
    tmpBPMind = np.where(bpm_ord == par.RMstruct.BPMords)[0]

    R0 = bpm_reading(SC) if par.useBPMreadingsForOrbBumpRef else np.zeros((2, len(par.RMstruct.BPMords)))
    R0[n_dim, tmpBPMind] += par.BBABPMtarget
    CMords = par.RMstruct.CMords

    W0 = np.ones((2, len(par.RMstruct.BPMords)))  # TODO weight for SCFedbackRun
    W0[n_dim, max(1, tmpBPMind - par.orbBumpWindow):(tmpBPMind - 1)] = 0
    W0[n_dim, (tmpBPMind + 1):min(len(par.RMstruct.BPMords), tmpBPMind + par.orbBumpWindow)] = 0

    CUR = SCfeedbackRun(SC, par.RMstruct.MinvCO, reference=R0, CMords=CMords, BPMords=par.RMstruct.BPMords, eps=1E-6,
                        target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, )
    CMvec = []
    factor = np.linspace(-1, 1, par.nSteps)
    for n_dim in range(2):
        vec0 = get_cm_setpoints(SC, CMords[n_dim], skewness=bool(n_dim))
        vec1 = get_cm_setpoints(CUR, CMords[n_dim], skewness=bool(n_dim))
        CMvec.append(vec0 + np.outer(factor, vec0 - vec1))

    return CMords, CMvec


def _plot_bba_step(SC, ax, bpm_ind, n_dim):
    s_pos = findspos(SC.RING)
    bpm_readings, all_elements_positions = all_elements_reading(SC)
    ax.plot(s_pos[SC.ORD.BPM], 1E3 * bpm_readings[n_dim, :len(SC.ORD.BPM)], marker='o')
    ax.plot(s_pos[SC.ORD.BPM[bpm_ind]], 1E3 * bpm_readings[n_dim, bpm_ind], marker='o', markersize=10, markerfacecolor='k')
    ax.plot(s_pos, 1E3 * all_elements_positions[n_dim, 0, :, 0, 0], linestyle='-')
    return ax


def plot_bba_results(SC, init_offset_errors, error_flags, bpm_ords, mag_ords):
    plt.rcParams.update({'font.size': 10})
    fom0 = init_offset_errors
    fom = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
    #fom[:, bpm_ind + 1:] = np.nan
    n_steps = 1 if bpm_ords.shape[1] == 1 else 1.1 * np.max(np.abs(fom0)) * np.linspace(-1, 1, int(np.floor(bpm_ords.shape[1] / 3)))
    f, ax = plt.subplots(nrows=3, num=90, figsize=(8, 8), facecolor="w")
    colors = ['#1f77b4', '#ff7f0e']
    for n_dim in range(bpm_ords.shape[0]):
        a, b = np.histogram(fom[n_dim, :], n_steps)
        ax[0].plot(1E6 * b[1:], a, linewidth=2)
    a, b = np.histogram(fom0, n_steps)
    ax[0].plot(1E6 * b[1:], a, 'k-', linewidth=2)
    if bpm_ords.shape[0] > 1:
        rmss = 1E6 * np.sqrt(np.nanmean(np.square(fom), axis=1))
        init_rms = 1E6 * np.sqrt(np.nanmean(np.square(fom0)))
        legends = [f"Horizontal rms: {rmss[0]:.0f}$\\mu m$",
                   f"Vertical rms:  {rmss[1]:.0f}$\\mu m$",
                   f"Initial rms: {init_rms:.0f}$\\mu m$"]
        ax[0].legend(legends)
    ax[0].set_xlabel(r'Final BPM offset w.r.t. magnet [$\mu$m]')
    ax[0].set_ylabel('Occurrences')

    mask_errors = error_flags == 0
    plabels = ("Horizontal", "Vertical")
    for n_dim in range(bpm_ords.shape[0]):
        x = np.where(np.in1d(SC.ORD.BPM, bpm_ords[n_dim, :]))[0]
        mask = mask_errors[n_dim, :]
        if not np.all(np.isnan(fom[n_dim, mask])):
            ax[1].plot(x[mask], 1E6 * np.abs(fom[n_dim, mask]), label=plabels[n_dim], marker='o', linewidth=2, color=colors[n_dim])
        if not np.all(np.isnan(fom[n_dim, ~mask])):
            ax[1].plot(x[~mask], 1E6 * np.abs(fom[n_dim, ~mask]), label=f"{plabels[n_dim]} failed", marker='X', linewidth=2, color=colors[n_dim])
        ax[2].plot(x, 1E6 * (fom0[n_dim, :] - fom[n_dim, :]), label=plabels[n_dim], marker='d', linewidth=2)

    ax[1].set_ylabel(r'Final offset [$\mu$m]')
    ax[1].set_xlabel('Index of BPM')
    ax[1].set_xlim((1, len(SC.ORD.BPM)))
    ax[1].legend()

    ax[2].set_ylabel(r' Offset $\Delta$ [$\mu$m]')
    ax[2].set_xlabel('Index of BPM')
    ax[2].set(xlim=(1, len(SC.ORD.BPM)))
    ax[2].legend()
    f.tight_layout()
    f.show()
