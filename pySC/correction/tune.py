import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pySC.core.beam import SCgetBeamTransmission, plot_transmission
from pySC.core.lattice_setting import SCsetMags2SetPoints
from pySC.utils import logging_tools

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
        SC = SCsetMags2SetPoints(SC, ords, False, 1, q_setpoints, method='rel')
        max_turns[q1, q2], lost_fraction = SCgetBeamTransmission(SC, nParticles=nParticles, nTurns=nTurns)
        transmission[q1, q2, :] = 1 - lost_fraction
        SC = SCsetMags2SetPoints(SC, ords, False, 1, 1 / q_setpoints, method='rel')

        if do_plot:
            f, ax = plot_scan(transmission[:, :, -1], max_turns, first_quads, rel_quad_changes)
            ax[2] = plot_transmission(ax[2], lost_fraction, nTurns, SC.INJ.beamLostAt)
            f.tight_layout()
            plt.show()

        if not full_scan and transmission[q1, q2, -1] >= target:
            setpoints = [rel_quad_changes[0][q1], rel_quad_changes[1][q2]]
            LOGGER.info(f'Transmission target reached with:\n'
                        f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                        f'    {first_quads[1]} SetPoint: {setpoints[1]:.4f}')
            SC = SCsetMags2SetPoints(SC, ords, False, 1, q_setpoints, method='rel')
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
        LOGGER.warning(f'No transmission at final turn at all. Maximum of {testTurns[max_ind]:d} turns reached with:\n'
                       f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                       f'    {first_quads[1]} SetPoint: {setpoints[0]:.4f}')
    else:
        LOGGER.warning(f'Transmission target not reached. Best value ({max_transmission:.4f}) reached with:\n'
                       f'    {first_quads[0]} SetPoint: {setpoints[0]:.4f}\n'
                       f'    {first_quads[1]} SetPoint: {setpoints[0]:.4f}')
    q_setpoints = np.hstack((setpoints[0] * np.ones(nq[0]), setpoints[1] * np.ones(nq[1])))
    SC = SCsetMags2SetPoints(SC, ords, False, 1, q_setpoints, method='rel')
    return SC, setpoints, max_turns, transmission


def _golden_donut_inds(r, n_points):
    ints = np.arange(n_points)
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    pos_2d = np.round(
        np.sqrt(ints / (n_points - 1)) * np.vstack((r[0] * np.cos(ints * phi), r[1] * np.sin(ints * phi))))
    return np.array(np.clip(pos_2d + r[:, np.newaxis], a_min=np.zeros((2, 1)), a_max=2 * r[:, np.newaxis]), dtype=int)


def plot_scan(fin_trans, max_turns, first_quads, rel_quad_changes):
    f = plt.figure(num=185)
    gs = GridSpec(2, 2, height_ratios=[2.5, 1])
    ax1, ax2 = f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[1, :])
    ticks = np.outer(np.floor_divide(np.array([len(rel_quad_changes[0]), len(rel_quad_changes[1])], dtype=int), 4),
                     np.arange(5, dtype=int))
    im = ax1.imshow(100 * fin_trans, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, orientation='vertical', label='Beam transmission [%]', shrink=0.6)
    im2 = ax2.imshow(max_turns, vmin=0)
    plt.colorbar(im2, ax=ax2, orientation='vertical', label='Number of turns', shrink=0.6)
    for ax in (ax1, ax2):
        ax.set_xlabel(f'{first_quads[0]} [relative]')
        ax.set_ylabel(f'{first_quads[1]} [relative]')
        ax.set_xticks(ticks[0], rel_quad_changes[0][ticks[0, :]])
        ax.set_yticks(ticks[0], rel_quad_changes[1][ticks[1, :]])
    return f, [ax1, ax2, ax3]
