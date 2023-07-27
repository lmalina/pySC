import matplotlib.pyplot as plt
import numpy as np

from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.correction.orbit_trajectory import SCfeedbackRun
from pySC.core.beam import bpm_reading, all_elements_reading
from pySC.utils.sc_tools import SCrandnc
from pySC.core.lattice_setting import set_cm_setpoints, set_magnet_setpoints, get_cm_setpoints
from pySC.utils import logging_tools
from pySC.core.classes import DotDict
from pySC.core.constants import TRACKING_MODES, TRACK_TBT

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
                       maxTrajChangeAtInjection=np.array([.9E-3, .9E-3]), quadOrdPhaseAdvance=np.array([8]),
                       quadStrengthPhaseAdvance=np.array([0.95, 1.05]), fakeMeasForFailures=False, dipole_compensation=True,
                       skewQuadrupole=False, switch_off_sextupoles=False, useBPMreadingsForOrbBumpRef=False,
                       plotLines=False, plotResults=False))
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
    kick_vec0 = par.maxTrajChangeAtInjection.reshape(2, 1) * np.linspace(-1, 1, par.nSteps)

    for j_bpm in range(bpm_ords.shape[1]):  # j_bpm: Index of BPM adjacent to magnet for BBA
        for n_dim in range(bpm_ords.shape[0]):
            LOGGER.debug(f'BBA-BPM {j_bpm}/{bpm_ords.shape[1]}, n_dim = {n_dim}')
            SC0 = SC
            bpm_ind = np.where(bpm_ords[n_dim, j_bpm] == SC.ORD.BPM)[0][0]
            m_ord = mag_ords[n_dim, j_bpm]
            if par.switch_off_sextupoles:
                SC = set_magnet_setpoints(SC, m_ord, skewness=False, order=2, setpoints=np.zeros(1), method='abs')
                SC = SCfeedbackRun(SC, par.RMstruct.MinvCO, BPMords=par.RMstruct.BPMords, CMords=par.RMstruct.CMords,
                                   target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, eps=1E-6)
            if par.mode == 'ORB':
                bpm_pos, tmpTra = _data_measurement_orb(SC, m_ord, bpm_ind, j_bpm, n_dim, par,
                                                   *_get_orbit_bump(SC, m_ord, bpm_ords[n_dim, j_bpm], n_dim, par))
            else:
                kick_vec, bpm_range = _scale_injection_to_reach_bpm(SC, bpm_ind, n_dim, kick_vec0)
                if par.quadOrdPhaseAdvance and bpm_range < par.BBABPMtarget:
                    SC, kick_vec = _scan_phase_advance(SC, bpm_ind, n_dim, kick_vec0, par)
                bpm_pos, tmpTra = _data_measurement_tbt(SC, m_ord, bpm_ind, j_bpm, n_dim, par, kick_vec)
            offset_change, error_flags[n_dim, j_bpm] = _data_evaluation(SC, bpm_ords, j_bpm, bpm_pos, tmpTra, n_dim, m_ord,
                                                                     par)
            SC = SC0
            if not np.isnan(offset_change):
                SC.RING[bpm_ords[n_dim, j_bpm]].Offset[n_dim] += offset_change
        if par.plotResults:
            plot_bba_results(SC, init_offset_errors, error_flags, j_bpm, bpm_ords, mag_ords)
    if par.fakeMeasForFailures:
        SC = _fake_measurement(SC, bpm_ords, mag_ords, error_flags)
    return SC, error_flags


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
    sPos = findspos(SC.RING)
    measDim = 1 - n_dim if par.skewQuadrupole else n_dim
    initialZ0 = SC.INJ.Z0.copy()
    nMsteps = kick_vec.shape[1]
    tmpTra = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm]), par.maxNumOfDownstreamBPMs), np.nan)

    BPMpos = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm])), np.nan)
    if par.plotLines:
        f, ax = plt.subplots(nrows=len(par.magSPvec[n_dim, j_bpm]), num=99)
    for nQ, setpointQ in enumerate(par.magSPvec[n_dim, j_bpm]):
        SC = set_magnet_setpoints(SC, np.array([m_ord]), par.skewQuadrupole, par.magnet_order, np.array([setpointQ]),
                                  method=par.setpoint_method, dipole_compensation=par.dipole_compensation)
        for nKick in range(nMsteps):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initialZ0[2 * n_dim:2 * n_dim + 2] + kick_vec[:, nKick]
            B = bpm_reading(SC)
            if par.plotLines:
                ax[nQ] = _plot_bba_step(SC, ax[nQ], bpm_ind, n_dim)
            BPMpos[nKick, nQ] = B[n_dim, bpm_ind]
            tmpTra[nKick, nQ, :] = B[measDim, bpm_ind:(bpm_ind + par.maxNumOfDownstreamBPMs)]

        if par.plotLines:
            ax[nQ].rectangle([sPos[m_ord], -1, sPos[m_ord + 1] - sPos[m_ord], 1], facecolor=[0, 0.4470, 0.7410])
            ax[nQ].set_xlim(sPos[m_ord] + np.array([-10, 10]))
            ax[nQ].set_ylim(1.3 * np.array([-1, 1]))
    plt.show()
    SC.INJ.Z0 = initialZ0
    return BPMpos, tmpTra


def _data_measurement_orb(SC, mOrd, BPMind, j_bpm, n_dim, par, CMords, cm_vec):
    s_pos = findspos(SC.RING)
    meas_dim = 1 - n_dim if par.skewQuadrupole else n_dim
    initial_z0 = SC.INJ.Z0.copy()
    nMsteps = cm_vec[n_dim].shape[0]
    tmpTra = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm]), len(SC.ORD.BPM)), np.nan)

    BPMpos = np.full((nMsteps, len(par.magSPvec[n_dim, j_bpm])), np.nan)
    if par.plotLines:
        f, ax = plt.subplots(nrows=len(par.magSPvec[n_dim, j_bpm]), num=99)
    for nQ, setpointQ in enumerate(par.magSPvec[n_dim, j_bpm]):
        SC = set_magnet_setpoints(SC, mOrd, par.skewQuadrupole, par.magnet_order, setpointQ, method=par.setpoint_method,
                                  dipole_compensation=par.dipole_compensation)
        for nKick in range(nMsteps):
            for nD in range(2):
                SC, _ = set_cm_setpoints(SC, CMords[nD], cm_vec[nD][nKick, :], bool(nD), method='abs')
            B = bpm_reading(SC)
            if par.plotLines:
                ax[nQ] = _plot_bba_step(SC, ax[nQ], BPMind, n_dim)
            BPMpos[nKick, nQ] = B[n_dim, BPMind]
            tmpTra[nKick, nQ, :] = B[meas_dim, :]

        if par.plotLines:
            ax[nQ].rectangle([s_pos[mOrd], -1, s_pos[mOrd + 1] - s_pos[mOrd], 1], facecolor=[0, 0.4470, 0.7410])
            ax[nQ].set_xlim(s_pos[mOrd] + np.array([-10, 10]))
            ax[nQ].set_ylim(1.3 * np.array([-1, 1]))
    plt.show()
    SC.INJ.Z0 = initial_z0
    return BPMpos, tmpTra


def _data_evaluation(SC, bpm_ords, j_bpm, bpm_pos, tmpTra, n_dim, m_ord, par):
    if par.plotLines:
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(num=56, facecolor="w", projection="3d")
        p1 = ax.plot(0, 1E6 * SC.RING[m_ord].T2[2 * n_dim - 1], 0, 'rD', MarkerSize=40, MarkerFaceColor='b')
    offset_change = np.nan
    Error = 5
    tmpCenter = np.full(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs), np.nan)
    tmpNorm = np.full(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs), np.nan)
    tmpRangeX = np.zeros(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs))
    tmpRangeY = np.zeros(((tmpTra.shape[0] - 1) * par.maxNumOfDownstreamBPMs))
    i = 0
    for nBPM in range(par.maxNumOfDownstreamBPMs):
        y0 = np.diff(tmpTra[:, :, nBPM], 1, axis=1)
        x0 = np.tile(np.mean(bpm_pos, 1), (y0.shape[1], 1)).T
        for nKick in range(y0.shape[1]):
            i = i + 1
            y = y0[:, nKick]
            x = x0[:, nKick]
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            if len(x) == 0 or len(y) == 0:
                continue
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
            if par.plotLines:
                p2 = ax.plot(np.tile(j_bpm + nBPM, x.shape), 1E6 * x, 1E3 * y, 'ko')
                tmp = ax.plot(np.tile(j_bpm + nBPM, x.shape), 1E6 * x, 1E3 * np.polyval(sol, x), 'k-')
                p3 = tmp[0]
                p4 = plt.plot(j_bpm + nBPM, 1E6 * tmpCenter[nBPM], 0, 'Or', MarkerSize=10)
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
        if 'BendingAngle' in SC.RING[m_ord].keys():
            B = SC.RING[m_ord].BendingAngle
        else:
            B = 0
        K = SC.RING[m_ord].NomPolynomB[1]
        L = SC.RING[m_ord].Length
        offset_change = offset_change + B / L / K
    if offset_change > par.outlierRejectionAt:
        offset_change = np.nan
        Error = 6
    if par.plotLines:
        p5 = plt.plot(0, 1E6 * offset_change, 0, 'kD', MarkerSize=30, MarkerFaceColor='r')
        p6 = plt.plot(0, 1E6 * (SC.RING[bpm_ords[n_dim, j_bpm]].Offset[n_dim] + SC.RING[bpm_ords[n_dim, j_bpm]].SupportOffset[
            n_dim] + offset_change), 0, 'kD', MarkerSize=30, MarkerFaceColor='g')
        ax.title(
            f'BBA-BPM: {j_bpm:d} \n mOrd: {m_ord:d} \n mFam: {SC.RING[m_ord].FamName} \n nDim: {n_dim:d} \n FinOffset = {1E6 * np.abs(SC.RING[bpm_ords[n_dim, j_bpm]].Offset[n_dim] + SC.RING[bpm_ords[n_dim, j_bpm]].SupportOffset[n_dim] + offset_change - SC.RING[m_ord].MagnetOffset[n_dim] - SC.RING[m_ord].SupportOffset[n_dim]):3.0f} $\\mu m$')
        ax.legend((p1, p2, p3, p4, p5, p6), (
        'Magnet center', 'Measured offset change', 'Line fit', 'Fitted BPM offset (individual)',
        'Fitted BPM offset (mean)', 'Predicted magnet center'))
        ax.set_xlabel('Index of BPM')
        ax.set_ylabel(r'BBA-BPM offset [$\mu$m]')
        ax.set_zlabel('Offset change [mm]')
        plt.show()
    return offset_change, Error


def _scale_injection_to_reach_bpm(SC, bpm_ind, n_dim, kick_vec0):
    initial_z0 = SC.INJ.Z0.copy()
    for scaling_factor in (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        tmp_bpm_pos = np.full(kick_vec0.shape[1], np.nan)
        for nK in range(kick_vec0.shape[1]):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initial_z0[2 * n_dim:2 * n_dim + 2] + scaling_factor * kick_vec0[:, nK]
            tmp_bpm_pos[nK] = bpm_reading(SC, np.array([SC.ORD.BPM[bpm_ind]]))[n_dim, 0]
        SC.INJ.Z0 = initial_z0.copy()

        if np.sum(np.isnan(tmp_bpm_pos)) == 0:
            bpm_range = np.max(tmp_bpm_pos) - np.min(tmp_bpm_pos)
            kick_vec = scaling_factor * kick_vec0
            LOGGER.debug(f'Initial trajectory variation scaled to [{100 * (kick_vec[0] / kick_vec0[0])}| '
                         f'{100 * (kick_vec[-1] / kick_vec0[-1])}]% of its initial value, '
                         f'BBA-BPM range {1E6 * bpm_range:.0f} um.')
            return kick_vec, bpm_range
    else:
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
        return kick_vec0, 0


def _scan_phase_advance(SC, bpm_ind, n_dim, kick_vec0, par):
    q_ord = par.quadOrdPhaseAdvance
    q_vec = par.quadStrengthPhaseAdvance
    q0 = SC.RING[q_ord].SetPointB[1]
    all_bpm_ranges = np.zeros(len(q_vec))
    for nQ in range(len(q_vec)):
        LOGGER.debug(f'BBA-BPM range too small, try to change phase advance with quad ord {q_ord} to {q_vec[nQ]:.2f} of nom. SP.')
        SC = set_magnet_setpoints(SC, q_ord, False, 1, q_vec[nQ], method='rel', dipole_compensation=True)
        kick_vec, bpm_range = _scale_injection_to_reach_bpm(SC, bpm_ind, n_dim, kick_vec0)
        if bpm_range >= par.BBABPMtarget:
            LOGGER.debug(f'Change phase advance with quad ord {q_ord} successful. BBA-BPM range = {1E6 * bpm_range:.0f} um.')
            return SC, kick_vec
        all_bpm_ranges[nQ] = bpm_range

    if all_bpm_ranges[-1] < np.max(all_bpm_ranges):
        LOGGER.debug(f'Changing phase advance of quad with ord {q_ord} NOT succesfull, '
                     f'returning to best value with BBA-BPM range = {1E6 * max(all_bpm_ranges):.0f}um.')
        SC = set_magnet_setpoints(SC, q_ord, False, 1, np.max(q_vec), method='rel', dipole_compensation=True)
        kick_vec, _ = _scale_injection_to_reach_bpm(SC, bpm_ind, n_dim, kick_vec0)
        return SC, kick_vec
    LOGGER.debug(f'Changing phase advance of quad with ord {q_ord} NOT succesfull, returning to initial setpoint.')
    SC = set_magnet_setpoints(SC, q_ord, False, 1, q0, method='abs', dipole_compensation=True)
    kick_vec, _ = _scale_injection_to_reach_bpm(SC, bpm_ind, n_dim, kick_vec0)
    return SC, kick_vec


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
    ax.plot(s_pos[SC.ORD.BPM], 1E3 * bpm_readings[n_dim, :], marker='o')
    ax.plot(s_pos[SC.ORD.BPM[bpm_ind]], 1E3 * bpm_readings[n_dim, bpm_ind], marker='o', markersize=10, markerfacecolor='k')
    ax.plot(s_pos, 1E3 * all_elements_positions[n_dim, 0, :, 0], linestyle='-')  # TODO 5D
    return ax


def plot_bba_results(SC, init_offset_errors, error_flags, bpm_ind, bpm_ords, mag_ords):
    plt.rcParams.update({'font.size': 18})
    fom0 = init_offset_errors
    fom = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
    fom[:, bpm_ind + 1:] = np.nan
    n_steps = 1 if bpm_ords.shape[1] == 1 else 1.1 * np.max(np.abs(fom0)) * np.linspace(-1, 1, np.floor(bpm_ords.shape[1] / 3))
    f, ax = plt.subplots(nrows=3, num=90, facecolor="w")
    colors = ['#1f77b4', '#ff7f0e']
    for n_dim in range(bpm_ords.shape[0]):
        a, b = np.histogram(fom[n_dim, :], n_steps)
        ax[0].plot(1E6 * b, a, linewidth=2)
    a, b = np.histogram(fom0, n_steps)
    ax[0].plot(1E6 * b, a, 'k-', linewidth=2)
    if bpm_ords.shape[0] > 1:
        rmss = 1E6 * np.sqrt(np.nanmean(np.square(fom), axis=1))
        init_rms = 1E6 * np.sqrt(np.nanmean(np.square(fom)))
        legends = [f"Horizontal rms: {rmss[0]:.0f}$\\mu m$",
                   f"Vertical rms:  {rmss[1]:.0f}$\\mu m$",
                   f"Initial rms: {init_rms:.0f}$\\mu m$"]
        ax[0].legend(legends)
    ax[0].set_xlabel(r'Final BPM offset w.r.t. magnet [$\mu$m]')
    ax[0].set_ylabel('Number of counts')
    ax[0].set(box="on")

    mask_errors = error_flags == 0
    plabels = ("Horizontal", "Vertical")
    for n_dim in range(bpm_ords.shape[0]):
        x = np.where(np.in1d(SC.ORD.BPM, bpm_ords[n_dim, :]))
        mask = mask_errors[n_dim, :]
        if not np.all(np.isnan(fom[n_dim, mask])):
            ax[1].plot(x[mask], 1E6 * np.abs(fom[n_dim, mask]), label=plabels[n_dim], marker='O', linewidth=2, color=colors[n_dim])
        if not np.all(np.isnan(fom[n_dim, ~mask])):
            ax[1].plot(x[~mask], 1E6 * np.abs(fom[n_dim, ~mask]), label=f"{plabels[n_dim]} failed", marker='X', linewidth=2, color=colors[n_dim])
        ax[2].plot(x, 1E6 * (fom0[n_dim, :] - fom[n_dim, :]), label=plabels[n_dim], marker='d', linewidth=2)

    ax[1].set_ylabel(r'Final offset [$\mu$m]')
    ax[1].set_xlabel('Index of BPM')
    ax[1].set(xlim=(1, len(SC.ORD.BPM)), box='on')
    ax[1].legend()

    ax[2].set_ylabel(r'Offsets change [$\mu$m]')
    ax[2].set_xlabel('Index of BPM')
    ax[2].set(xlim=(1, len(SC.ORD.BPM)), box='on')
    ax[2].legend()

    f.show()
