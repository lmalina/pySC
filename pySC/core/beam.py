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


def bpm_reading(SC: SimulatedComissioning, bpm_ords: ndarray = None) -> ndarray:
    """
    Calculates BPM readings with current injection setup `SC.INJ` and included all BPM uncertainties.
    Included uncertainties are offsets, rolls, calibration errors, and position noise.
    When the beam is lost following `SC.INJ.beamLostAt` criteria with sum signal errors included,
    the readings are NaN.
    If SC.plot is True the reading is plotted.

    Args:
        SC: SimulatedComissioning instance
        bpm_ords: array of element indices of registered BPMs for which to calculate readings
            (for convenience, otherwise `SC.ORD.BPM` is used)

    Returns:
        Array of horizontal and vertical BPM readings (2, T x B) for T turns and B BPMs
    """
    all_bpm_orbits_4d = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns, SC.INJ.nShots), np.nan)
    for shot_num in range(SC.INJ.nShots):
        tracking_4d = _tracking(SC, SC.ORD.BPM)
        all_bpm_orbits_4d[:, :, :, shot_num] = _real_bpm_reading(SC, tracking_4d)

    mean_bpm_orbits_3d = np.nanmean(all_bpm_orbits_4d, axis=3)  # mean_bpm_orbits_3d is 3D (dim, BPM, turn)
    if SC.plot:
        _plot_bpm_reading(SC, mean_bpm_orbits_3d)
    if SC.INJ.trackMode == 'PORB':   # ORB averaged over low amount of turns
        mean_bpm_orbits_3d = np.nanmean(mean_bpm_orbits_3d, axis=2, keepdims=True)
    if bpm_ords is not None:
        ind = np.where(np.isin(SC.ORD.BPM, bpm_ords))[0]
        if len(ind) != len(bpm_ords):
            LOGGER.warning('Not all specified ordinates are registered BPMs.')
        mean_bpm_orbits_3d = mean_bpm_orbits_3d[:, ind, :]
    # Organising the array 2 x (nturns x nbpms) sorted by "arrival time"
    return _reshape_3d_to_matlab_like_2d(mean_bpm_orbits_3d)


def all_elements_reading(SC: SimulatedComissioning) -> Tuple[ndarray, ndarray]:
    """
    Calculates horizontal and vertical positions with current injection setup `SC.INJ`.
    Returns the measured BPM positions at all BPMs as well as true positions at all elements.
    If SC.plot is True the reading is plotted.

    Args:
        SC: SimulatedComissioning instance

    Returns:
        Array of all horizontal and vertical BPM readings (2, T x B) for T turns and B BPMs

        Array of all horizontal and vertical positions at all elements
        (2, P, B, T, S) for P particles B BPMs, T turns and S shots
    """
    n_refs = len(SC.RING) + 1
    all_readings_5d = np.full((2, SC.INJ.nParticles, n_refs, SC.INJ.nTurns, SC.INJ.nShots), np.nan)
    all_bpm_orbits_4d = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns, SC.INJ.nShots), np.nan)
    for shot_num in range(SC.INJ.nShots):
        tracking_4d = _tracking(SC, np.arange(n_refs))
        all_bpm_orbits_4d[:, :, :, shot_num] = _real_bpm_reading(SC, tracking_4d[:, :, SC.ORD.BPM, :])
        tracking_4d[:, np.isnan(tracking_4d[0, :])] = np.nan
        all_readings_5d[:, :, :, :, shot_num] = tracking_4d[:, :, :, :]

    mean_bpm_orbits_3d = np.nanmean(all_bpm_orbits_4d, axis=3)  # mean_bpm_orbits_3d is 3D (dim, BPM, turn)
    if SC.plot:
        _plot_bpm_reading(SC, mean_bpm_orbits_3d, all_readings_5d)
    return _reshape_3d_to_matlab_like_2d(mean_bpm_orbits_3d), all_readings_5d


def beam_transmission(SC: SimulatedComissioning, nParticles: int = None, nTurns: int = None,
                      do_plot: bool = False) -> Tuple[int, ndarray]:
    """
    Calculates the turn-by-turn beam transmission with current injection setup as defined in `SC.INJ`.

    Args:
        SC: SimulatedComissioning instance
        nParticles: Number of particles to track
            (for convenience, otherwise `SC.INJ.nParticles` is used)
        nTurns: Number of turns to track
            (for convenience, otherwise `SC.INJ.nTurns` is used)
        do_plot: If True, plots beam transmission

    Returns:
        Number of survived turns following `SC.INJ.beamLostAt` criteria

        Array of accumulated lost fraction of beam (turn-by-turn)

    """
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    LOGGER.debug(f'Calculating maximum beam transmission for {nParticles} particles and {nTurns} turns: ')
    T = atpass(SC.RING, generate_bunches(SC, nParticles=nParticles), nTurns, np.array([len(SC.RING)]), keep_lattice=False)
    fraction_lost = np.mean(np.isnan(T[0, :, :, :]), axis=(0, 1))
    max_turns = np.sum(fraction_lost < SC.INJ.beamLostAt)
    if do_plot:
        fig, ax = plt.subplots()
        ax = plot_transmission(ax, fraction_lost, nTurns, SC.INJ.beamLostAt)
        fig.tight_layout()
        fig.show()
    LOGGER.info(f'{max_turns} turns and {100 * (1 - fraction_lost[-1]):.0f}% transmission.')
    return int(max_turns), fraction_lost


def generate_bunches(SC: SimulatedComissioning, nParticles=None) -> ndarray:
    """
    Generates bunches according to the current injection setup as defined in `SC.INJ`.
    The random injection error is added to the mean injected beam trajectory for each bunch.
    Either a single particle (nParticles=1) is generated at the bunch centroid,
    or individual particles are randomly distributed around the bunch centroid
    using the beam sigma matrix.
    A function `SC.INJ.postFun` is applied to the generated coordinates.

    Args:
        SC: SimulatedComissioning instance
        nParticles: Number of particles to generate
            (for convenience, otherwise `SC.INJ.nParticles` is used)

    Returns:
        Array of particle coordinates (6, nParticles)
    """
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    Z = np.tile(np.transpose(SC.INJ.randomInjectionZ * SCrandnc(2, (1, 6)) + SC.INJ.Z0), nParticles)
    if nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        Z += np.diag(np.sqrt(V)) @ L @ SCrandnc(3, (6, nParticles))
    return SC.INJ.postFun(Z)


def plot_transmission(ax, fraction_lost, n_turns, beam_lost_at):
    ax.plot(fraction_lost, lw=3)
    ax.plot([0, n_turns], [beam_lost_at, beam_lost_at], 'k:', lw=2)
    ax.set_xlim([0, n_turns])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of turns')
    ax.set_ylabel('Lost fraction of beam')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
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


def _tracking(SC: SimulatedComissioning, refs: ndarray) -> ndarray:
    """Returns numpy array (2, N, R, T) of X and Y coordinates of N particles at R reference points for T turns.
       If ORB: N and T are equal to 1"""
    #  lattice_pass output:            (6, N, R, T) coordinates of N particles at R reference points for T turns.
    #  findorbit second output value:  (R, 6) closed orbit vector at each specified location
    if SC.INJ.trackMode == 'ORB':
        return np.transpose(findorbit6(SC.RING, refs, keep_lattice=False)[1])[[0, 2], :].reshape(2, 1, len(refs), 1)
    return atpass(SC.RING, generate_bunches(SC), SC.INJ.nTurns, refs, keep_lattice=False)[[0, 2], :, :, :]


def _reshape_3d_to_matlab_like_2d(mean_bpm_orbits_3d: ndarray) -> ndarray:
    """Organising the array the same way as in matlab version (2, nturns x nbpms) sorted by 'arrival time'."""
    return np.transpose(mean_bpm_orbits_3d, axes=(0, 2, 1)).reshape((2, np.prod(mean_bpm_orbits_3d.shape[1:])))


def _plot_bpm_reading(SC, bpm_orbits_3d, all_readings_5d=None):
    ap_ords, apers = _get_ring_aperture(SC)
    fig, ax = plt.subplots(num=1, nrows=2, ncols=1, sharex="all", figsize=(8, 6), dpi=100, facecolor="w")
    s_pos = findspos(SC.RING)
    circumference = s_pos[-1]

    if all_readings_5d is not None:
        x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * circumference + s_pos)
        ax = _plot_all_trajectories(ax, x, all_readings_5d)
    for n_dim in range(2):
        x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * circumference + s_pos[SC.ORD.BPM])
        y = 1E3 * np.ravel(bpm_orbits_3d[n_dim, :, :].T)
        ax[n_dim].plot(x, y, 'ro', label="BPM reading")
        if len(ap_ords):
            x = np.ravel(np.arange(SC.INJ.nTurns)[:, np.newaxis] * circumference + s_pos[ap_ords])
            y = 1E3 * np.tile(apers[:, n_dim, :].T, SC.INJ.nTurns)  # to mm
            ax[n_dim].plot(x, y.T, '-', color='#1f77b4', linewidth=4, label="Aperture")

    turn_breaks = np.arange(SC.INJ.nTurns + 1) * circumference
    ax = _plot_annotations_and_limits(ax, 3 * np.array([-5, 5]), turn_breaks)
    fig.tight_layout()
    plt.pause(0.001)
    plt.show()


def _plot_all_trajectories(ax, x, all_readings_5d):
    for n_dim in range(2):
        for shot_number in range(all_readings_5d.shape[4]):
            y = 1E3 * all_readings_5d[n_dim, :, :, :, shot_number]
            y = np.reshape(np.transpose(y, axes=(2, 1, 0)), (np.prod(y.shape[1:]), y.shape[0]))
            ax[n_dim].plot(x, y, 'k')
            if shot_number == 0:
                ax[n_dim].plot(x, y, 'k', label='Particle trajectories')
    return ax


def _plot_annotations_and_limits(ax, y_lims, turn_breaks):
    ylabels = [r'$\Delta x$ [mm]', r'$\Delta y$ [mm]']
    for n_dim in range(2):
        for x in turn_breaks[:-1]:
            ax[n_dim].plot(x * np.ones(2), y_lims, 'k:')
        ax[n_dim].set_xlim([0, turn_breaks[-1]])
        ax[n_dim].set_ylim(y_lims)
        ax[n_dim].set_ylabel(ylabels[n_dim])
        for item in ([ax[n_dim].title, ax[n_dim].xaxis.label, ax[n_dim].yaxis.label] +
                     ax[n_dim].get_xticklabels() + ax[n_dim].get_yticklabels()):
            item.set_fontsize(18)
            item.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.75))

    handles, labels = ax[1].get_legend_handles_labels()
    labels, l_inds = np.unique(labels, return_index=True)
    handles = [handles[ind] for ind in l_inds]
    ax[0].legend(handles, labels, bbox_to_anchor=(0., 1.02, 1., .15), loc='lower center',
                 ncols=3, borderaxespad=0., fontsize=(14 if len(handles) > 2 else 18))
    ax[1].set_xlabel('$s$ [m]')
    return ax


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
