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
from pySC.utils import logging_tools
from pySC.utils.at_wrapper import atlinopt

LOGGER = logging_tools.get_logger(__name__)


def tune_scan(SC, quad_ords, rel_quad_changes, target=1, n_points=60, do_plot=False, nParticles=None, nTurns=None,
              full_scan=False):
    """
    Varies two quadrupole families to improve beam transmission, on a grid of relative setpoints specified in
    `rel_quad_changes` in a spiral-like pattern to increase the beam transmission.
    
    Args:
        SC: SimulatedCommissioning instance
        quad_ords: [1x2] array of quadrupole ordinates {`[1 x NQ1],[1 x NQ2]`}
        rel_quad_changes: [1x2] cell array of quadrupole setpoints {`[SP1_1,...,SP1_N1],[SP2_1,...,SP2_N2]`} with `N2=N1`
        target(int, optional): Transmission target at `nTurns`
        n_points(int, optional): Number of points for the scan
        nParticles(int, optional): Number of particles used for tracking (for convenience, otherwise `SC.INJ.nParticles` is used)
        nTurns(int, optional): Number of turns used for tracking (for convenience, otherwise `SC.INJ.nTurns` is used)

        full_scan( bool , optional): If false, the scan finishes as soon as the target is reached
        do_plot( bool , optional): If true, beam transmission is plotted at every step
    Returns:
        Updated SC structure with applied setpoints for maximised beam transmission
        Relative setpoints which satisfied the target condition if reached, or the values which resulted in best transmission
        Number of achieved turns
        Turn-by-turn particle loss

    see also: *bpm_reading*, *generate_bunches*
    """
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
        SC.set_magnet_setpoints(ords, q_setpoints, False, 1, method='rel')
        max_turns[q1, q2], surviving_fraction = beam_transmission(SC, nParticles=nParticles, nTurns=nTurns)
        transmission[q1, q2, :] = 1 * surviving_fraction
        SC.set_magnet_setpoints(ords, 1 / q_setpoints, False, 1, method='rel')

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
            SC.set_magnet_setpoints(ords, q_setpoints, False, 1, method='rel')
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
    SC.set_magnet_setpoints(ords, q_setpoints, False, 1, method='rel')
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


def fit_tune(SC, q_ords, target_tune=None, init_step_size=np.array([0.001, 0.001]),xtol=1E-4, ftol=1E-3, fit_integer=True):
    """
        Applies a tune correction using two quadrupole families.
        Note: this is not beam based but assumes the tunes can be measured reasonably well.

        Args:
            SC: SimulatedCommissioning instance
            q_ords: [2xN] array or list [[1 x NQF],[1 x NQD]] of quadrupole ordinates
            target_tune (optional, [1x2] array): Target tunes for correction. Default: tunes of 'SC.IDEALRING'
            init_step_size ([1x2] array, optional): Initial step size for the solver. Default: [1,1]
            xtol(float, optional): Step tolerance for solver. Default: 1e-4
            ftol(float, optional): Merit tolerance for solver. Default: 1e-4
            fit_integer(bool, optional): Flag specifying if the integer part should be fitted as well. Default: True.

        Returns:
            SC: SimulatedCommissioning instance with corrected tunes.
        Example:
            SC = fit_tune(SC, q_ords=[SCgetOrds(sc.RING, 'QF'), SCgetOrds(sc.RING, 'QD')], target_tune=numpy.array([0.16,0.21]))
        """
    if target_tune is None:
        target_tune = tune(SC, fit_integer, ideal=True)
    LOGGER.debug(f'Fitting tunes from [{SC.RING.get_tune(get_integer=fit_integer)}] to [{target_tune}].')
    SP0 = []
    for n in range(len(q_ords))
        SP0.append(np.zeros_like(q_ords[n])) #working with a list of two arrays
    for nFam in range(len(q_ords)):
        for n in range(len(q_ords[nFam])):
            SP0[nFam][n] = SC.RING[q_ords[nFam][n]].SetPointB[1]
    fun = lambda x: _fit_tune_fun(SC, q_ords, x, SP0, target_tune, fit_integer)
    sol = fmin(fun, init_step_size, xtol=xtol, ftol=ftol)
    LOGGER.debug(f'       Final tune: [{SC.RING.get_tune(get_integer=fit_integer)}]\n  Setpoints change: [{sol}]')
    return SC


def _fit_tune_fun(SC, q_ords, setpoints, init_setpoints, target, fit_integer):
    for nFam in range(len(q_ords)):
        SC.set_magnet_setpoints(q_ords[nFam], setpoints[nFam] + init_setpoints[nFam], False, 1, method='abs', dipole_compensation=True)
    nu = SC.RING.get_tune(get_integer=fit_integer)
    nu = nu[0:2]
    return np.sqrt(np.mean((nu - target) ** 2))

def tune(SC, fit_integer: bool = False, ideal: bool = False):
    ring = SC.IDEALRING if ideal else SC.RING
    if fit_integer:
        ld, _, _ = atlinopt(ring, 0, range(len(ring) + 1))
        return ld[-1].mu / 2 / np.pi
    _, nu, _ = atlinopt(ring, 0)
    return nu
