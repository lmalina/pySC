import numpy as np
from matplotlib import pyplot as plt

from pySC.core.SCgenBunches import SCgenBunches
from pySC.utils.sc_tools import SCrandnc
from pySC.at_wrapper import atgetfieldvalues, atpass, findorbit6, findspos
import warnings

def SCgetBPMreading(SC, BPMords=None):
    #  lattice_pass output:            (6, N, R, T) coordinates of N particles at R reference points for T turns.
    #  findorbit second output value:  (R, 6) closed orbit vector at each specified location
    refs = np.arange(len(SC.RING)) if SC.plot else SC.ORD.BPM[:]
    n_refs = len(refs)
    if SC.plot:
        all_readings_5d = np.full((2, SC.INJ.nParticles, n_refs, SC.INJ.nTurns, SC.INJ.nShots), np.nan)
    all_bpm_orbits_4d = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns, SC.INJ.nShots), np.nan)
    for nShot in range(SC.INJ.nShots):
        if SC.INJ.trackMode == 'ORB':
            tracking_4d = np.transpose(findorbit6(SC.RING, refs, keep_lattice=False)[1])[[0, 2], :].reshape(2, 1, n_refs, 1)
        else:
            tracking_4d = atpass(SC.RING, SCgenBunches(SC), SC.INJ.nTurns, refs, keep_lattice=False)[[0, 2], :, :, :]
        all_bpm_orbits_4d[:, :, :, nShot] = _real_bpm_reading(SC, tracking_4d[:, :, SC.ORD.BPM, :] if SC.plot else tracking_4d)
        tracking_4d[:, np.isnan(tracking_4d[0, :])] = np.nan
        if SC.plot:
            all_readings_5d[:, :, :, :, nShot] = tracking_4d[:, :, :, :]

    mean_bpm_orbits_3d = _loc_nan_mean(all_bpm_orbits_4d, axis=3)  # mean_bpm_orbits_3d is 3D (dim, BPM, turn)
    if SC.plot:
        _plot_bpm_reading(SC, mean_bpm_orbits_3d, all_readings_5d)

    if SC.INJ.trackMode == 'PORB':   # ORB averaged over low amount of turns
        mean_bpm_orbits_3d = _loc_nan_mean(mean_bpm_orbits_3d, axis=2, keepdims=True)
    if BPMords is not None:
        ind = np.where(np.isin(SC.ORD.BPM, BPMords))[0]
        if len(ind) != len(BPMords):
            print('Not all specified ordinates are registered BPMs.')
        mean_bpm_orbits_3d = mean_bpm_orbits_3d[:, ind, :]
    # Organising the array the same way as in matlab version 2 x (nturns, nbpms) sorted by "arrival time"
    mean_bpm_orbits_2d = np.transpose(mean_bpm_orbits_3d, axes=(0, 2, 1)).reshape((2, np.prod(mean_bpm_orbits_3d.shape[1:])))
    # if SC.plot:
    #     return mean_bpm_orbits_2d, all_readings_5d
    return mean_bpm_orbits_2d

def _loc_nan_mean(a, axis=0, keepdims=False):
    # Workaround to avoid runtime warnings when all entries are nan, TODO: find better solution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis, keepdims=keepdims)    

def _real_bpm_reading(SC, track_mat):  # track_mat should be only x,y over all particles only at BPM positions
    nBpms, nTurns = track_mat.shape[2:]
    bpm_noise = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, ('NoiseCO' if SC.INJ.trackMode == 'ORB' else "Noise")))
    bpm_noise = bpm_noise[:, :, np.newaxis] * SCrandnc(2, (2, nBpms, nTurns))
    bpm_offset = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Offset') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportOffset'))
    bpm_cal_error = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'CalError'))
    bpm_roll = np.squeeze(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Roll') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportRoll'))
    bpm_sum_error = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SumError'))[:, np.newaxis] * SCrandnc(2, (nBpms, nTurns))
    # averaging the X and Y positions at BPMs over particles
    mean_orbit = _loc_nan_mean(track_mat, axis=1)
    beam_lost = np.nonzero(np.mean(np.isnan(track_mat[0, :, :, :]), axis=0) * (1 + bpm_sum_error) > SC.INJ.beamLostAt)
    mean_orbit[:, beam_lost[0], beam_lost[1]] = np.nan

    rolled_mean_orbit = np.einsum("ijk,jkl->ikl", _rotation_matrix(bpm_roll), mean_orbit)
    return (rolled_mean_orbit - bpm_offset[:, :, np.newaxis]) * (1 + bpm_cal_error[:, :, np.newaxis]) + bpm_noise


def _rotation_matrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _plot_bpm_reading(SC, B, T):  # T is 5D matrix
    ap_ords, apers = _get_ring_aperture(SC)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=100, facecolor="w")
    tmpCol = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ylabelStr = ['$\Delta x$ [mm]', '$\Delta y$ [mm]']
    legStr = ['Particle trajectories', 'BPM reading', 'Aperture']
    sPos = findspos(SC.RING, range(len(SC.RING)))
    sMax = sPos[-1]
    for nDim in range(2):
        x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * sMax + sPos)
        for nS in range(SC.INJ.nShots):
            y = 1E3 * T[nDim, :, :, :, nS]
            y = np.reshape(np.transpose(y, axes=(2, 1, 0)), (np.prod(y.shape[1:]), y.shape[0]))
            ax[nDim].plot(x, y, 'k')
        #legVec = legVec[0]
        x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * sMax + sPos[SC.ORD.BPM])
        y = 1E3 * np.ravel(B[nDim, :, :].T)
        ax[nDim].plot(x, y, 'ro')
        if len(ap_ords):
            apS = sPos[ap_ords]
            x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * sMax + apS)
            y = 1E3 * np.tile(apers[:, nDim, :].T, SC.INJ.nTurns)  # to mm
            ax[nDim].plot(x, y[0, :], '-', color=tmpCol[0], linewidth=4)
            ax[nDim].plot(x, y[1, :], '-', color=tmpCol[0], linewidth=4)
        x = np.arange(SC.INJ.nTurns) * sMax
        y_lims = 3*np.array([-5, 5])
        for nT in range(SC.INJ.nTurns):
            ax[nDim].plot(x[nT] * np.ones(2), y_lims, 'k:')
        ax[nDim].set_xlim([0, SC.INJ.nTurns * sPos[-1]])
        #ax[nDim].set_box('on')  # True?
        ax[nDim].set_ylim(y_lims)
        ax[nDim].set_ylabel(ylabelStr[nDim])
        #ax[nDim].legend(legVec, legStr[0:len(legVec)])
        ax[nDim].set_xlabel('$s$ [m]')
        #ax[nDim].set_position(ax[nDim].get_position() + np.array([0, .07 * (nDim - 1), 0, 0]))
        for item in ([ax[nDim].title, ax[nDim].xaxis.label, ax[nDim].yaxis.label] +
                 ax[nDim].get_xticklabels() + ax[nDim].get_yticklabels()):
            item.set_fontsize(18)
            item.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
    plt.show()
    # plt.pause(0.001)
    # plt.tight_layout()
    # plt.gcf().canvas.draw()
    # plt.gcf().canvas.flush_events()


def _get_ring_aperture(SC):
    ords, aps = [], []
    for ord in range(len(SC.RING)):
        if hasattr(SC.RING[ord], 'EApertures') or hasattr(SC.RING[ord], 'RApertures'):
            ords.append(ord)
            aps.append(np.outer(getattr(SC.RING[ord], 'EApertures'), np.array([-1, 1]))
                       if hasattr(SC.RING[ord], 'EApertures')
                       else np.reshape(getattr(SC.RING[ord], 'RApertures'), (2, 2))) #  TODO is it [-x, +x, -y, +y] if not, has to be changed # most likely swap sign
    return np.array(ords), np.array(aps)
