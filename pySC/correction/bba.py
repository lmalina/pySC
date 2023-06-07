import matplotlib.pyplot as plt
import numpy as np

from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.correction.orbit_trajectory import SCfeedbackRun
from pySC.core.beam import SCgetBPMreading
from pySC.utils.sc_tools import SCrandnc
from pySC.core.lattice_setting import SCsetCMs2SetPoints, SCsetMags2SetPoints, SCgetCMSetPoints
from pySC.utils import logging_tools
from pySC.core.classes import DotDict

LOGGER = logging_tools.get_logger(__name__)


def SCBBA(SC, bpm_ords, mag_ords, **kwargs):
    par = DotDict(dict(mode=SC.INJ.trackMode, outlierRejectionAt=np.inf, nSteps=10, fitOrder=1, magOrder=2,
                       magSPvec=[0.95, 1.05], magSPflag='rel', RMstruct=[], orbBumpWindow=5, BBABPMtarget=1E-3,
                       minBPMrangeAtBBABBPM=500E-6, minBPMrangeOtherBPM=100E-6, maxStdForFittedCenters=600E-6,
                       nXPointsNeededAtMeasBPM=3, maxNumOfDownstreamBPMs=len(SC.ORD.BPM), minSlopeForFit=0.03,
                       maxTrajChangeAtInjection=[.9E-3, .9E-3], quadOrdPhaseAdvance=[],
                       quadStrengthPhaseAdvance=[0.95, 1.05], fakeMeasForFailures=False, dipCompensation=True,
                       skewQuadrupole=False, switchOffSext=False, useBPMreadingsForOrbBumpRef=False,
                       plotLines=False, plotResults=False))
    par.update(**kwargs)
    if bpm_ords.shape != mag_ords.shape:  # both in shape 2 x N
        raise ValueError('Input arrays for BPMs and magnets must be same size.')
    if not isinstance(par.magSPvec, list):
        par.magSPvec = [par.magSPvec] * len(mag_ords)
    if par.mode not in ("TBT", "ORB"):
        raise ValueError(f"Unknown mode {par.mode}.")
    if par.mode == 'TBT' and SC.INJ.nTurns != 2:
        raise ValueError('Beam-based alignment in TBT mode works with 2 turns. Please set: SC.INJ.nTurns = 2')
    init_offset_errors = _get_bpm_offset_from_mag(SC, bpm_ords, mag_ords)
    error_flags = np.full(bpm_ords.shape, np.nan)
    kickVec0 = par.maxTrajChangeAtInjection.reshape(2, 1) * np.linspace(-1, 1, par.nSteps)

    for jBPM in range(bpm_ords.shape[1]):  # jBPM: Index of BPM adjacent to magnet for BBA
        for nDim in range(bpm_ords.shape[0]):
            LOGGER.debug(f'BBA-BPM {jBPM}/{bpm_ords.shape[1]}, nDim = {nDim}')
            SC0 = SC
            BPMind = np.where(bpm_ords[nDim, jBPM] == SC.ORD.BPM)[0][0]
            mOrd = mag_ords[nDim, jBPM]
            if par.switchOffSext:
                SC = SCsetMags2SetPoints(SC, mOrd, skewness=False, order=2, setpoints=np.zeros(1), method='abs')
                SC = SCfeedbackRun(SC, par.RMstruct.MinvCO, BPMords=par.RMstruct.BPMords, CMords=par.RMstruct.CMords,
                                   target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, eps=1E-6)
            if par.mode == 'ORB':
                BPMpos, tmpTra = _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par,
                                                   _get_orbit_bump(SC, mOrd, bpm_ords[nDim, jBPM], nDim, par))
            else:
                kickVec, BPMrange = _scale_injection_to_reach_bpm(SC, BPMind, nDim, kickVec0)
                if par.quadOrdPhaseAdvance and BPMrange < par.BBABPMtarget:
                    SC, kickVec = _scan_phase_advance(SC, BPMind, nDim, kickVec0, par)
                BPMpos, tmpTra = _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par, kickVec)
            OffsetChange, error_flags[nDim, jBPM] = _data_evaluation(SC, bpm_ords, jBPM, BPMpos, tmpTra, nDim, mOrd,
                                                                     par)
            SC = SC0
            if not np.isnan(OffsetChange):
                SC.RING[bpm_ords[nDim, jBPM]].Offset[nDim] = SC.RING[bpm_ords[nDim, jBPM]].Offset[nDim] + OffsetChange
        if par.plotResults:
            plot_bba_results(SC, init_offset_errors, error_flags, jBPM, bpm_ords, mag_ords)
    if par.fakeMeasForFailures:
        SC = _fake_measurement(SC, bpm_ords, mag_ords, error_flags)
    return SC, error_flags


def _get_bpm_offset_from_mag(SC, BPMords, magOrds):
    offset = np.full(BPMords.shape, np.nan)
    for n_dim in range(2):
        offset[n_dim, :] = (atgetfieldvalues(SC.RING, BPMords[n_dim, :], 'Offset', n_dim)
                            + atgetfieldvalues(SC.RING, BPMords[n_dim, :], 'SupportOffset', n_dim)
                            - atgetfieldvalues(SC.RING, magOrds[n_dim, :], 'MagnetOffset', n_dim)
                            - atgetfieldvalues(SC.RING, magOrds[n_dim, :], 'SupportOffset', n_dim))
    return offset


def _fake_measurement(SC, BPMords, magOrds, errorFlags):
    finOffsetErrors = _get_bpm_offset_from_mag(SC, BPMords, magOrds)
    finOffsetErrors[errorFlags != 0] = np.nan
    LOGGER.info(f"Final offset error is {1E6 * np.sqrt(np.nanmean(finOffsetErrors ** 2, axis=1))}"
                f" um (hor|ver) with {np.sum(errorFlags != 0, axis=1)} measurement failures -> being re-calculated now.\n")

    for nBPM in range(BPMords.shape[1]):
        for nDim in range(2):
            if errorFlags[nDim, nBPM] != 0:
                fakeBPMoffset = (SC.RING[magOrds[nDim, nBPM]].MagnetOffset[nDim]
                                 + SC.RING[magOrds[nDim, nBPM]].SupportOffset[nDim]
                                 - SC.RING[BPMords[nDim, nBPM]].SupportOffset[nDim]
                                 + np.sqrt(np.nanmean(np.square(finOffsetErrors[nDim, :]))) * SCrandnc(2))
                if not np.isnan(fakeBPMoffset):
                    SC.RING[BPMords[nDim, nBPM]].Offset[nDim] = fakeBPMoffset
                else:
                    LOGGER.info('BPM offset not reasigned, NaN.\n')
    return SC


def _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par, varargin):
    sPos = findspos(SC.RING)
    measDim = 1 - nDim if par.skewQuadrupole else nDim
    initialZ0 = SC.INJ.Z0.copy()
    if par.mode == 'ORB':
        CMords, CMvec = varargin
        nMsteps = CMvec[nDim].shape[0]
        tmpTra = np.nan((nMsteps, len(par.magSPvec[nDim, jBPM]), len(SC.ORD.BPM)))
    else:
        kickVec = varargin.copy()
        nMsteps = kickVec.shape[1]
        tmpTra = np.nan((nMsteps, len(par.magSPvec[nDim, jBPM]), par.maxNumOfDownstreamBPMs))

    BPMpos = np.nan((nMsteps, len(par.magSPvec[nDim, jBPM])))
    if par.plotLines:
        f, ax = plt.subplots(nrows=len(par.magSPvec[nDim, jBPM]), num=99)
    for nQ, setpointQ in enumerate(par.magSPvec[nDim, jBPM]):
        SC = SCsetMags2SetPoints(SC, mOrd, par.skewQuadrupole, par.magOrder, setpointQ, method=par.magSPflag,
                                 dipCompensation=par.dipCompensation)
        for nKick in range(nMsteps):
            if par.mode == 'ORB':
                for nD in range(2):
                    SC, _ = SCsetCMs2SetPoints(SC, CMords[nD], CMvec[nD][nKick, :], bool(nD), method='abs')
            else:
                SC.INJ.Z0[2 * nDim:2 * nDim + 2] = initialZ0[2 * nDim:2 * nDim + 2] + kickVec[:, nKick]
            B = SCgetBPMreading(SC)
            if par.plotLines:
                ax[nQ] = _plot_bba_step(SC, ax[nQ], BPMind, nDim)
            BPMpos[nKick, nQ] = B[nDim, BPMind]
            tmpTra[nKick, nQ, :] = B[measDim, :] if par.mode == 'ORB' else B[measDim, (BPMind + 1):(
                        BPMind + par.maxNumOfDownstreamBPMs)]

        if par.plotLines:
            ax[nQ].rectangle([sPos[mOrd], -1, sPos[mOrd + 1] - sPos[mOrd], 1], facecolor=[0, 0.4470, 0.7410])
            ax[nQ].set_xlim(sPos[mOrd] + np.array([-10, 10]))
            ax[nQ].set_ylim(1.3 * np.array([-1, 1]))
    plt.show()
    SC.INJ.Z0 = initialZ0
    return BPMpos, tmpTra


def _data_evaluation(SC, BPMords, jBPM, BPMpos, tmpTra, nDim, mOrd, par):
    if par.plotLines:
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(num=56, facecolor="w", projection="3d")
        p1 = ax.plot(0, 1E6 * SC.RING[mOrd].T2[2 * nDim - 1], 0, 'rD', MarkerSize=40, MarkerFaceColor='b')
    OffsetChange = np.nan
    Error = 5
    tmpCenter = np.nan((1, (tmpTra.shape[1] - 1) * par.maxNumOfDownstreamBPMs))
    tmpNorm = np.nan((1, (tmpTra.shape[1] - 1) * par.maxNumOfDownstreamBPMs))
    tmpRangeX = np.zeros((1, (tmpTra.shape[1] - 1) * par.maxNumOfDownstreamBPMs))
    tmpRangeY = np.zeros((1, (tmpTra.shape[1] - 1) * par.maxNumOfDownstreamBPMs))
    i = 0
    for nBPM in range(par.maxNumOfDownstreamBPMs):
        y0 = np.diff(tmpTra[:, :, nBPM], 1, 1)
        x0 = np.tile(np.mean(BPMpos, 1), (y0.shape[1], 1)).T
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
            sol = np.nan((1, 2))
            if len(x) >= par.nXPointsNeededAtMeasBPM and tmpRangeX[i] > par.minBPMrangeAtBBABBPM and tmpRangeY[
                i] > par.minBPMrangeOtherBPM:
                if par.fitOrder == 1:
                    sol = np.linalg.lstsq(np.vstack((np.ones(x.shape), x)).T, y)[0]
                    sol = sol[[1, 0]]
                    if abs(sol[0]) < par.minSlopeForFit:
                        sol[0] = np.nan
                    tmpCenter[i] = -sol[1] / sol[0]
                    tmpNorm[i] = 1 / np.sqrt(np.sum((sol[0] * x + sol[1] - y) ** 2))
                else:
                    sol = np.polyfit(x, y, par.fitOrder)
                    if par.fitOrder == 2:
                        tmpCenter[i] = - (sol[1] / (2 * sol[0]))
                    else:
                        tmpCenter[i] = min(abs(np.roots(sol)))
                    tmpNorm[i] = 1 / np.linalg.norm(np.polyval(sol, x) - y)
            if par.plotLines:
                p2 = ax.plot(np.tile(jBPM + nBPM, x.shape), 1E6 * x, 1E3 * y, 'ko')
                tmp = ax.plot(np.tile(jBPM + nBPM, x.shape), 1E6 * x, 1E3 * np.polyval(sol, x), 'k-')
                p3 = tmp[0]
                p4 = plt.plot(jBPM + nBPM, 1E6 * tmpCenter[nBPM], 0, 'Or', MarkerSize=10)
    if np.max(tmpRangeX) < par.minBPMrangeAtBBABBPM:
        Error = 1
    elif np.max(tmpRangeY) < par.minBPMrangeOtherBPM:
        Error = 2
    elif np.std(tmpCenter, ddof=1) > par.maxStdForFittedCenters:
        Error = 3
    elif len(np.where(~np.isnan(tmpCenter))[0]) == 0:
        Error = 4
    else:
        OffsetChange = np.sum(tmpCenter * tmpNorm) / np.sum(tmpNorm)
        Error = 0
    if not par.dipCompensation and nDim == 1 and SC.RING[mOrd].NomPolynomB[1] != 0:
        if 'BendingAngle' in SC.RING[mOrd].keys():
            B = SC.RING[mOrd].BendingAngle
        else:
            B = 0
        K = SC.RING[mOrd].NomPolynomB[1]
        L = SC.RING[mOrd].Length
        OffsetChange = OffsetChange + B / L / K
    if OffsetChange > par.outlierRejectionAt:
        OffsetChange = np.nan
        Error = 6
    if par.plotLines:
        p5 = plt.plot(0, 1E6 * OffsetChange, 0, 'kD', MarkerSize=30, MarkerFaceColor='r')
        p6 = plt.plot(0, 1E6 * (SC.RING[BPMords[nDim, jBPM]].Offset[nDim] + SC.RING[BPMords[nDim, jBPM]].SupportOffset[
            nDim] + OffsetChange), 0, 'kD', MarkerSize=30, MarkerFaceColor='g')
        ax.title(
            f'BBA-BPM: {jBPM:d} \n mOrd: {mOrd:d} \n mFam: {SC.RING[mOrd].FamName} \n nDim: {nDim:d} \n FinOffset = {1E6 * np.abs(SC.RING[BPMords[nDim, jBPM]].Offset[nDim] + SC.RING[BPMords[nDim, jBPM]].SupportOffset[nDim] + OffsetChange - SC.RING[mOrd].MagnetOffset[nDim] - SC.RING[mOrd].SupportOffset[nDim]):3.0f} $\\mu m$')
        ax.legend((p1, p2, p3, p4, p5, p6), (
        'Magnet center', 'Measured offset change', 'Line fit', 'Fitted BPM offset (individual)',
        'Fitted BPM offset (mean)', 'Predicted magnet center'))
        ax.set_xlabel('Index of BPM')
        ax.set_ylabel('BBA-BPM offset [$\mu$m]')
        ax.set_zlabel('Offset change [mm]')
        plt.show()
    return OffsetChange, Error


def _scale_injection_to_reach_bpm(SC, BPMind, nDim, kickVec0):
    initialZ0 = SC.INJ.Z0.copy()
    for scaling_factor in (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        tmp_bpm_pos = np.full(kickVec0.shape[1], np.nan)
        for nK in range(kickVec0.shape[1]):
            SC.INJ.Z0[2 * nDim:2 * nDim + 2] = initialZ0[2 * nDim:2 * nDim + 2] + scaling_factor * kickVec0[:, nK]
            tmp_bpm_pos[nK] = SCgetBPMreading(SC, SC.ORD.BPM[BPMind])[nDim, 0]
        SC.INJ.Z0 = initialZ0.copy()

        if np.sum(np.isnan(tmp_bpm_pos)) == 0:
            BPMrange = np.max(tmp_bpm_pos) - np.min(tmp_bpm_pos)
            kickVec = scaling_factor * kickVec0
            LOGGER.debug(f'Initial trajectory variation scaled to [{100 * (kickVec[0] / kickVec0[0]):.0f}| '
                         f'{100 * (kickVec[-1] / kickVec0[-1]):.0f}]% of its initial value, '
                         f'BBA-BPM range {1E6 * BPMrange:.0f} um.')
            return kickVec, BPMrange
    else:
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
        return kickVec0, 0


def _scan_phase_advance(SC, BPMind, nDim, kickVec0, par):
    mOrd = par.quadOrdPhaseAdvance
    qVec = par.quadStrengthPhaseAdvance
    q0 = SC.RING[mOrd].SetPointB[1]
    allBPMRange = np.zeros(len(qVec))
    for nQ in range(len(qVec)):
        LOGGER.debug(f'BBA-BPM range to small, try to change phase advance with quad ord {par.quadOrdPhaseAdvance} '
                     f'to {qVec[nQ]:.2f} of nom. SP.')
        SC = SCsetMags2SetPoints(SC, mOrd, False, 1, qVec[nQ], method='rel', dipCompensation=True)
        kickVec, BPMrange = _scale_injection_to_reach_bpm(SC, BPMind, nDim, kickVec0)

        if BPMrange >= par.BBABPMtarget:
            LOGGER.debug(
                f'Change phase advance with quad ord {mOrd} successful. BBA-BPM range = {1E6 * BPMrange:.0f}um.')
            return SC, kickVec
        allBPMRange[nQ] = BPMrange

    if BPMrange < np.max(allBPMRange):
        LOGGER.debug(f'Changing phase advance of quad with ord {mOrd} NOT succesfull, '
                     f'returning to best value with BBA-BPM range = {1E6 * max(allBPMRange):.0f}um.')
        return SCsetMags2SetPoints(SC, mOrd, False, 1, np.max(qVec), method='rel', dipCompensation=True), kickVec
    LOGGER.debug(f'Changing phase advance of quad with ord {mOrd} NOT succesfull, returning to initial setpoint.')
    return SCsetMags2SetPoints(SC, mOrd, False, 1, q0, method='abs', dipCompensation=True), kickVec


def _get_orbit_bump(SC, mOrd, BPMord, nDim, par):
    tmpCMind = np.where(par.RMstruct.CMords[0] == mOrd)[0]
    if len(tmpCMind):
        par.RMstruct.RM = np.delete(par.RMstruct.RM, tmpCMind, 1)  # TODO not nice
        par.RMstruct.CMords[0] = np.delete(par.RMstruct.CMords[0], tmpCMind)
    tmpBPMind = np.where(BPMord == par.RMstruct.BPMords)[0]

    R0 = SCgetBPMreading(SC) if par.useBPMreadingsForOrbBumpRef else np.zeros((2, len(par.RMstruct.BPMords)))
    R0[nDim, tmpBPMind] += par.BBABPMtarget
    CMords = par.RMstruct.CMords

    W0 = np.ones((2, len(par.RMstruct.BPMords)))  # TODO weight for SCFedbackRun
    W0[nDim, max(1, tmpBPMind - par.orbBumpWindow):(tmpBPMind - 1)] = 0
    W0[nDim, (tmpBPMind + 1):min(len(par.RMstruct.BPMords), tmpBPMind + par.orbBumpWindow)] = 0

    CUR = SCfeedbackRun(SC, par.RMstruct.MinvCO, reference=R0, CMords=CMords, BPMords=par.RMstruct.BPMords, eps=1E-6,
                        target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, )
    CMvec = []
    factor = np.linspace(-1, 1, par.nSteps)
    for nDim in range(2):
        vec0 = SCgetCMSetPoints(SC, CMords[nDim], skewness=bool(nDim))
        vec1 = SCgetCMSetPoints(CUR, CMords[nDim], skewness=bool(nDim))
        CMvec.append(vec0 + np.outer(factor, vec0 - vec1))

    return CMords, CMvec


def _plot_bba_step(SC, ax, BPMind, nDim):
    s_pos = findspos(SC.RING)
    B, T = SCgetBPMreading(SC)  # TODO handle the readout at all elements
    ax.plot(s_pos[SC.ORD.BPM], 1E3 * B[nDim, :], marker='o')
    ax.plot(s_pos[SC.ORD.BPM[BPMind]], 1E3 * B[nDim, BPMind], marker='o', markersize=10, markerfacecolor='k')
    ax.plot(s_pos, 1E3 * T[nDim, 0, :, 0], linestyle='-')
    return ax


def plot_bba_results(SC, initOffsetErrors, errorFlags, jBPM, BPMords, magOrds):
    plt.rcParams.update({'font.size': 18})
    fom0 = initOffsetErrors
    fom = _get_bpm_offset_from_mag(SC, BPMords, magOrds)
    fom[:, jBPM + 1:] = np.nan
    if BPMords.shape[1] == 1:
        nSteps = 1
    else:
        nSteps = 1.1 * np.max(np.abs(fom0)) * np.linspace(-1, 1, np.floor(BPMords.shape[1] / 3))
    f, ax = plt.subplots(nrows=3, num=90, facecolor="w")
    colors = ['#1f77b4', '#ff7f0e']
    for nDim in range(BPMords.shape[0]):
        a, b = np.histogram(fom[nDim, :], nSteps)
        ax[0].plot(1E6 * b, a, linewidth=2)
    a, b = np.histogram(fom0, nSteps)
    ax[0].plot(1E6 * b, a, 'k-', linewidth=2)
    if BPMords.shape[0] > 1:
        rmss = 1E6 * np.sqrt(np.nanmean(np.square(fom), axis=1))
        init_rms = 1E6 * np.sqrt(np.nanmean(np.square(fom)))
        legends = [f"Horizontal rms: {rmss[0]:.0f}$\\mu m$",
                   f"Vertical rms:  {rmss[1]:.0f}$\\mu m$",
                   f"Initial rms: {init_rms:.0f}$\\mu m$"]
        ax[0].legend(legends)
    ax[0].set_xlabel(r'Final BPM offset w.r.t. magnet [$\mu$m]')
    ax[0].set_ylabel('Number of counts')
    ax[0].set(box="on")

    mask_errors = errorFlags == 0
    plabels = ("Horizontal", "Vertical")
    for nDim in range(BPMords.shape[0]):
        x = np.where(np.in1d(SC.ORD.BPM, BPMords[nDim, :]))
        mask = mask_errors[nDim, :]
        if not np.all(np.isnan(fom[nDim, mask])):
            ax[1].plot(x[mask], 1E6 * np.abs(fom[nDim, mask]), label=plabels[nDim], marker='O', linewidth=2, color=colors[nDim])
        if not np.all(np.isnan(fom[nDim, ~mask])):
            ax[1].plot(x[~mask], 1E6 * np.abs(fom[nDim, ~mask]), label=f"{plabels[nDim]} failed", marker='X', linewidth=2, color=colors[nDim])
        ax[2].plot(x, 1E6 * (fom0[nDim, :] - fom[nDim, :]), label=plabels[nDim], marker='d', linewidth=2)

    ax[1].set_ylabel(r'Final offset [$\mu$m]')
    ax[1].set_xlabel('Index of BPM')
    ax[1].set(xlim=(1, len(SC.ORD.BPM)), box='on')
    ax[1].legend()

    ax[2].set_ylabel(r'Offsets change [$\mu$m]')
    ax[2].set_xlabel('Index of BPM')
    ax[2].set(xlim=(1, len(SC.ORD.BPM)), box='on')
    ax[2].legend()

    plt.show()
