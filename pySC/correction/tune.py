"""
Tune
-------------

This module contains functions to correct betatron tunes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fmin

from pySC.core.beam import beam_transmission, plot_transmission
from pySC.core.lattice_setting import set_magnet_setpoints
from pySC.utils import logging_tools
from pySC.utils.at_wrapper import atlinopt

LOGGER = logging_tools.get_logger(__name__)


def tune_scan(SC, quad_ords, rel_quad_changes, target=1, n_points=60, do_plot=False, nParticles=None, nTurns=None,
              full_scan=False):
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    nq = np.array([len(quad_ords[0]), len(quad_ords[1])], dtype=int)
    nqsp = np.array([len(rel_quad_changes[0]), len(rel_quad_changes[1])], dtype=int)
    max_turns = np.full((nqsp[0], nqsp[1]), np.nan)
    transmission = np.full((nqsp[0], nqsp[1], nTurns), np.nan)
    inds = _golden_donut_inds(np.floor_divide(nqsp, 2), n_points=n_points)
    first_quads = [SC.RING[quad_ords[0][0]].FamName, SC.RING[quad_ords[1][0]].FamName]
    ords = np.hstack(quad_ords)
    for q1, q2 in inds.T:
        q_setpoints = np.hstack((np.ones(nq[0]) * rel_quad_changes[0][q1], np.ones(nq[1]) * rel_quad_changes[1][q2]))
        SC = set_magnet_setpoints(SC, ords, False, 1, q_setpoints, method='rel')
        max_turns[q1, q2], surviving_fraction = beam_transmission(SC, nParticles=nParticles, nTurns=nTurns)
        transmission[q1, q2, :] = 1 * surviving_fraction
        SC = set_magnet_setpoints(SC, ords, False, 1, 1 / q_setpoints, method='rel')

        if do_plot:
            f, ax = plot_scan(transmission[:, :, -1], max_turns, first_quads, rel_quad_changes)
            ax[2] = plot_transmission(ax[2], surviving_fraction, nTurns, SC.INJ.beamLostAt)
            f.tight_layout()
            f.show()

        if not full_scan and transmission[q1, q2, -1] >= target:
            setpoints = [rel_quad_changes[0][q1], rel_quad_changes[1][q2]]
            LOGGER.info(f'Transmission target reached with:\n'
                        f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                        f'    {first_quads[1]} SetPoint: {setpoints[1]:.4f}')
            SC = set_magnet_setpoints(SC, ords, False, 1, q_setpoints, method='rel')
            return SC, setpoints, max_turns, transmission

    testTrans = np.zeros(n_points)
    testTurns = np.zeros(n_points)
    for i in range(n_points):
        testTrans[i] = transmission[inds[0, i], inds[1, i], -1]
        testTurns[i] = max_turns[inds[0, i], inds[1, i]]
    if np.max(testTurns) == 0:
        raise RuntimeError('Fail, no transmission at all.\n')
    max_ind = np.argmax(testTurns if np.max(testTrans) == 0 else testTrans)
    if max_ind == 0:
        LOGGER.warning('No improvement possible.\n')
    setpoints = [rel_quad_changes[0][inds[0, max_ind]], rel_quad_changes[1][inds[1, max_ind]]]
    if (max_transmission := np.max(testTrans)) == 0:
        LOGGER.warning(f'No transmission at final turn at all. Maximum of {testTurns[max_ind]} turns reached with:\n'
                       f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                       f'    {first_quads[1]} SetPoint: {setpoints[0]:.4f}')
    else:
        LOGGER.warning(f'Transmission target not reached. Best value ({max_transmission:.4f}) reached with:\n'
                       f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                       f'    {first_quads[1]} SetPoint: {setpoints[0]:.4f}')
    q_setpoints = np.hstack((setpoints[0] * np.ones(nq[0]), setpoints[1] * np.ones(nq[1])))
    SC = set_magnet_setpoints(SC, ords, False, 1, q_setpoints, method='rel')
    return SC, setpoints, max_turns, transmission


def _golden_donut_inds(r, n_points):
    ints = np.arange(n_points)
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    pos_2d = np.round(
        np.sqrt(ints / (n_points - 1)) * np.vstack((r[0] * np.cos(ints * phi), r[1] * np.sin(ints * phi))))
    return np.array(np.clip(pos_2d + r[:, np.newaxis], a_min=np.zeros((2, 1)), a_max=2 * r[:, np.newaxis]), dtype=int)


def plot_scan(fin_trans, max_turns, first_quads, rel_quad_changes):
    f = plt.figure(num=185, figsize=(12, 9))
    gs = GridSpec(2, 2, height_ratios=[2, 1])
    ax1, ax2 = f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[1, :])
    ticks = np.array([[0, rel_quad_changes[0].shape[0] // 2, rel_quad_changes[0].shape[0] - 1],
                     [0, rel_quad_changes[1].shape[0] // 2, rel_quad_changes[1].shape[0] - 1]])
    im = ax1.imshow(100 * fin_trans, vmin=0, vmax=100)
    c1 = plt.colorbar(im, ax=ax1, orientation='vertical', label='Beam transmission [%]', shrink=0.6)
    im2 = ax2.imshow(max_turns, vmin=0)
    c2 = plt.colorbar(im2, ax=ax2, orientation='vertical', label='Number of turns', shrink=0.6)
    for ax in (ax1, ax2):
        ax.set_xlabel(rf'$\Delta${first_quads[0]} [relative]')
        ax.set_ylabel(rf'$\Delta${first_quads[1]} [relative]')
        ax.set_xticks(ticks[0], np.round(rel_quad_changes[0][ticks[0, :]]-1, 5))
        ax.set_yticks(ticks[0], np.round(rel_quad_changes[1][ticks[1, :]]-1, 5))
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
    for cx in (c1, c2):
        for item in ([cx.ax.yaxis.label] + cx.ax.get_yticklabels()):
            item.set_fontsize(18)
    return f, [ax1, ax2, ax3]


def fit_tune(SC, q_ords, target_tune=None, xtol=1E-4, ftol=1E-3, fit_integer=True):
    #  TODO check if experimantally feasible
    if target_tune is None:
        target_tune = tune(SC, fit_integer, ideal=True)
    LOGGER.debug(f'Fitting tunes from [{tune(SC, fit_integer)}] to [{target_tune}].')
    SP0 = np.zeros((len(q_ords), len(q_ords[0])))  # TODO can the lengts vary
    for nFam in range(len(q_ords)):
        for n in range(len(q_ords[nFam])):
            SP0[nFam][n] = SC.RING[q_ords[nFam][n]].SetPointB[1]
    fun = lambda x: _fit_tune_fun(SC, q_ords, x, SP0, target_tune, fit_integer)
    sol = fmin(fun, xtol=xtol, ftol=ftol)
    SC = set_magnet_setpoints(SC, q_ords, False, 1, sol + SP0, method='abs', dipole_compensation=True)
    LOGGER.debug(f'       Final tune: [{tune(SC, fit_integer)}]\n  Setpoints change: [{sol}]')
    return SC


def _fit_tune_fun(SC, q_ords, setpoints, init_setpoints, target, fit_integer):
    SC = set_magnet_setpoints(SC, q_ords, False, 1, setpoints + init_setpoints, method='abs', dipole_compensation=True)
    nu = tune(SC, fit_integer)
    return np.sqrt(np.mean((nu - target) ** 2))


def tune(SC, fit_integer: bool = False, ideal: bool = False):
    ring = SC.IDEALRING if ideal else SC.RING
    if fit_integer:
        ld, _, _ = atlinopt(ring, 0, range(len(ring) + 1))
        return ld[-1].mu / 2 / np.pi
    _, nu, _ = atlinopt(ring, 0)
    return nu
