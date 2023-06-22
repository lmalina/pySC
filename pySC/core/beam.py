from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from pySC.core.classes import SimulatedComissioning
from pySC.utils.sc_tools import SCrandnc
from pySC.utils.at_wrapper import atgetfieldvalues, atpass, findorbit6, findspos
import warnings
from pySC.utils import logging_tools

warnings.filterwarnings("ignore", message='Mean of empty slice')
LOGGER = logging_tools.get_logger(__name__)

def SCgetBPMreading(SC, BPMords=None):
    #  lattice_pass output:            (6, N, R, T) coordinates of N particles at R reference points for T turns.
    #  findorbit second output value:  (R, 6) closed orbit vector at each specified location
    refs = np.arange(len(SC.RING) + 1) if SC.plot else SC.ORD.BPM[:]
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

    mean_bpm_orbits_3d = np.nanmean(all_bpm_orbits_4d, axis=3)  # mean_bpm_orbits_3d is 3D (dim, BPM, turn)
    if SC.plot:
        _plot_bpm_reading(SC, mean_bpm_orbits_3d, all_readings_5d)

    if SC.INJ.trackMode == 'PORB':   # ORB averaged over low amount of turns
        mean_bpm_orbits_3d = np.nanmean(mean_bpm_orbits_3d, axis=2, keepdims=True)
    if BPMords is not None:
        ind = np.where(np.isin(SC.ORD.BPM, BPMords))[0]
        if len(ind) != len(BPMords):
            LOGGER.warning('Not all specified ordinates are registered BPMs.')
        mean_bpm_orbits_3d = mean_bpm_orbits_3d[:, ind, :]
    # Organising the array the same way as in matlab version 2 x (nturns, nbpms) sorted by "arrival time"
    mean_bpm_orbits_2d = np.transpose(mean_bpm_orbits_3d, axes=(0, 2, 1)).reshape((2, np.prod(mean_bpm_orbits_3d.shape[1:])))
    return mean_bpm_orbits_2d


def SCgetBeamTransmission(SC: SimulatedComissioning, nParticles: int = None, nTurns: int = None,
                          do_plot: bool = False) -> Tuple[int, ndarray]:
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    LOGGER.debug(f'Calculating maximum beam transmission for {nParticles} particles and {nTurns} turns: ')
    T = atpass(SC.RING, SCgenBunches(SC, nParticles=nParticles), nTurns, np.array([len(SC.RING)]), keep_lattice=False)
    fraction_lost = np.mean(np.isnan(T[0, :, :, :]), axis=(0, 1))
    max_turns = np.sum(fraction_lost < SC.INJ.beamLostAt)
    if do_plot:
        fig, ax = plt.subplots()
        ax = plot_transmission(ax, fraction_lost, nTurns, SC.INJ.beamLostAt)
        fig.show()
    LOGGER.info(f'{max_turns} turns and {100 * (1 - fraction_lost[-1]):.0f}% transmission.')
    return int(max_turns), fraction_lost


def SCgenBunches(SC: SimulatedComissioning, nParticles=None) -> ndarray:
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    Z = np.tile(np.transpose(SC.INJ.randomInjectionZ * SCrandnc(2, (1, 6)) + SC.INJ.Z0), nParticles)
    if nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        Z += np.diag(np.sqrt(V)) @ L @ SCrandnc(3, (6, nParticles))
    return SC.INJ.postFun(Z)


def plot_transmission(ax, fraction_lost, n_turns, beam_lost_at):
    ax.plot(fraction_lost)
    ax.plot([0, n_turns], [beam_lost_at, beam_lost_at], 'k:')
    ax.set_xlim([0, n_turns])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of turns')
    ax.set_ylabel('CDF of lost count')
    return ax

def _real_bpm_reading(SC, track_mat):  # track_mat should be only x,y over all particles only at BPM positions
    nBpms, nTurns = track_mat.shape[2:]
    bpm_noise = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, ('NoiseCO' if SC.INJ.trackMode == 'ORB' else "Noise")))
    bpm_noise = bpm_noise[:, :, np.newaxis] * SCrandnc(2, (2, nBpms, nTurns))
    bpm_offset = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Offset') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportOffset'))
    bpm_cal_error = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'CalError'))
    bpm_roll = np.squeeze(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Roll') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportRoll'))
    bpm_sum_error = np.transpose(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SumError'))[:, np.newaxis] * SCrandnc(2, (nBpms, nTurns))
    # averaging the X and Y positions at BPMs over particles
    mean_orbit = np.nanmean(track_mat, axis=1)
    beam_lost = np.nonzero(np.mean(np.isnan(track_mat[0, :, :, :]), axis=0) * (1 + bpm_sum_error) > SC.INJ.beamLostAt)
    mean_orbit[:, beam_lost[0], beam_lost[1]] = np.nan

    rolled_mean_orbit = np.einsum("ijk,jkl->ikl", _rotation_matrix(bpm_roll), mean_orbit)
    return (rolled_mean_orbit - bpm_offset[:, :, np.newaxis]) * (1 + bpm_cal_error[:, :, np.newaxis]) + bpm_noise


def _rotation_matrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _plot_bpm_reading(SC, B, T):  # T is 5D matrix
    ap_ords, apers = _get_ring_aperture(SC)
    fig, ax = plt.subplots(num=1, nrows=2, ncols=1, figsize=(8, 6), dpi=100, facecolor="w")
    ylabelStr = [r'$\Delta x$ [mm]', r'$\Delta y$ [mm]']
    legStr = ['Particle trajectories', 'BPM reading', 'Aperture']
    sPos = findspos(SC.RING)
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
            ax[nDim].plot(x, y.T, '-', color='#1f77b4', linewidth=4)
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
    plt.pause(0.001)
    # plt.tight_layout()
    # plt.gcf().canvas.draw()
    # plt.gcf().canvas.flush_events()


def _get_ring_aperture(SC):
    ords, aps = [], []
    for ind in range(len(SC.RING)):
        if hasattr(SC.RING[ind], 'EApertures') or hasattr(SC.RING[ind], 'RApertures'):
            ords.append(ind)
            aps.append(np.outer(getattr(SC.RING[ind], 'EApertures'), np.array([-1, 1]))
                       if hasattr(SC.RING[ind], 'EApertures')
                       # from RApertures [+x, -x, +y, -y]  to [[-x, +x], [-y, +y]]
                       else np.roll(np.reshape(getattr(SC.RING[ind], 'RApertures'), (2, 2)), 1, axis=1))
    return np.array(ords), np.array(aps)
