import matplotlib.pyplot as plt
import numpy as np
import copy
from pySC.utils import at_wrapper
from pySC.core.constants import SUPPORT_TYPES
from typing import Tuple
from pySC.core.simulated_commissioning import SimulatedCommissioning


def plot_support(SC: SimulatedCommissioning, font_size: int = 8, x_lim: Tuple[float, float] = None):
    """
    Plots the offset and rolls of magnets, the support structure and BPMs.
    Specifically, this function plots the overall offsets [dx,dy,dz] and rolls [az,ax,ay] of all magnets and BPMs,
    as well as the individual contributions from different support structures (if registered).

    Args:
        `'SC'`:  :py:class: SimulatedCommissioning
            A base structure.
        `'font_size'`: int
            Figure font size.
            Axes are rearranged for grouping. Depending on screen resolution this value may be adjusted.
        `'x_lim'`: Tuple[float, float]
            Plot limits in terms of longitudinal location

    Returns:

    """
    init_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': font_size})
    # Get s - positions along the lattice
    s_pos = at_wrapper.findspos(SC.RING)
    circumference = s_pos[-1]
    if x_lim is None:
        x_lim = (0, circumference)

    s = np.linspace(x_lim[0], x_lim[1], 100 * int((x_lim[1] - x_lim[0]) / 2))  # s locations to compute
    s_mag = s_pos[SC.ORD.Magnet]
    s_bpm = s_pos[SC.ORD.BPM]

    # Loop over individual support structure types
    datadict = {'Ords': [], 'Off': {'a': [], 'b': []}, 'Roll': []}
    supdict = {'Section': copy.deepcopy(datadict), 'Plinth': copy.deepcopy(datadict), 'Girder': copy.deepcopy(datadict)}
    for sup_type in supdict.keys():
        for ord_pair in SC.ORD[sup_type].transpose():
            if (x_lim[0] <= s_pos[ord_pair[0]] <= x_lim[1]) or (x_lim[0] <= s_pos[ord_pair[1]] <= x_lim[1]):
                # Structures in range
                supdict[sup_type]['Ords'].append(ord_pair)
                # Get girder start and ending offsets
                supdict[sup_type]['Off']['a'].append(SC.RING[ord_pair[0]].__dict__[sup_type + 'Offset'])
                supdict[sup_type]['Off']['b'].append(SC.RING[ord_pair[1]].__dict__[sup_type + 'Offset'])
                # Get girder rolls
                supdict[sup_type]['Roll'].append(SC.RING[ord_pair[0]].__dict__[sup_type + 'Roll'])

    # Magnet offsets and rolls
    off_support_line, roll_support_line = SC.support_offset_and_roll(s)
    off_mag_support = at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.Magnet, "SupportOffset")
    roll_mag_support = at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.Magnet, "SupportRoll")
    off_mag_individual = at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.Magnet, "MagnetOffset")
    roll_mag_individual = at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.Magnet, "MagnetRoll")
    off_mag_total = at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.Magnet, "T2")[:, [0, 2, 5]]
    roll_mag_total = roll_mag_support + roll_mag_individual

    # BPM offsets and rolls
    # Longitudinal offsets and Pitch and Yaw angles not supported for BPMs
    pad_off, pad_roll = ((0, 0), (0, 1)), ((0, 0), (0, 2))
    off_bpm = np.pad(at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.BPM, "Offset"), pad_off)
    roll_bpm = np.pad(at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.BPM, "Roll"), pad_roll)
    off_bpm_support = np.pad(at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.BPM, "SupportOffset"), pad_off)
    roll_bpm_support = np.pad(at_wrapper.atgetfieldvalues(SC.RING, SC.ORD.BPM, "SupportRoll"), pad_roll)

    # create figure
    fig, ax = plt.subplots(nrows=9, ncols=2, num=1213, sharex="all", figsize=(10, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    blu, ora = '#1f77b4', '#ff7f0e'
    line_spec = {'Plinth': {'color': 'r', 'linewidth': 4},
                 'Section': {'color': '#e377c2', 'linewidth': 2, 'linestyle': ':'},
                 'Girder': {'color': '#9467bd', 'linewidth': 4}}

    for n_dim in range(3):
        # plot overall support offsets and rolls
        ax[3*n_dim, 0].stairs(1e6*off_support_line[n_dim, :-1], s, color=blu)
        ax[3*n_dim, 0].plot(s_mag, 1e6 * off_mag_support[:, n_dim], 'D', color=blu, label='Overall supports')
        ax[3*n_dim, 1].stairs(1e6 * roll_support_line[n_dim, :-1], s, color=blu)
        ax[3*n_dim, 1].plot(s_mag, 1e6 * roll_mag_support[:, n_dim], 'D', color=blu, label='Overall supports')
        # plot individual support and roll contributions
        for sup_type in SUPPORT_TYPES:
            for i, val in enumerate(SC.ORD[sup_type].T):  # loop supports
                ax[3*n_dim, 0] = _plot_support(ax[3 * n_dim, 0], s_pos[val],
                                               [1e6 * supdict[sup_type]['Off']['a'][i][n_dim],
                                                1e6 * supdict[sup_type]['Off']['b'][i][n_dim]],
                                               circumference,
                                               **line_spec[sup_type])
                ax[3*n_dim, 1] = _plot_support(ax[3 * n_dim, 1], s_pos[val],
                                               1e6 * supdict[sup_type]['Roll'][i][n_dim] * np.ones(2),
                                               circumference,
                                               **line_spec[sup_type])

            # plot outside to get correct legend
            ax[3*n_dim, 0].plot([-2, -1], [0, 0], **line_spec[sup_type], label=f'Individual {sup_type}')
            ax[3*n_dim, 1].plot([-2, -1], [0, 0], **line_spec[sup_type], label=f'Individual {sup_type}')

        # plot magnet offset and roll
        ax[3*n_dim+1, 0].plot(s_mag, 1e6 * off_mag_individual[:, n_dim], 'kx', ms=8, label='Individual Magnet')
        ax[3*n_dim+1, 0].plot(s_mag, 1e6 * off_mag_total[:, n_dim], 'ko-', label='Overall magnet offset')
        ax[3*n_dim+1, 1].plot(s_mag, 1e6 * roll_mag_individual[:, n_dim], 'kx', ms=8, label='Individual Magnet')
        ax[3*n_dim+1, 1].plot(s_mag, 1e6 * roll_mag_total[:, n_dim], 'ko-', label='Overall magnet roll')
        # plot BPM offset and roll
        ax[3*n_dim+2, 0].plot(s_bpm, 1e6 * off_bpm[:, n_dim], 'o', color=ora, ms=6, label='Random BPM offset')
        ax[3*n_dim+2, 0].plot(s_bpm, 1e6 * off_bpm_support[:, n_dim], '-', color=ora, label='BPM support offset')
        ax[3*n_dim+2, 1].plot(s_bpm, 1e6 * roll_bpm[:, n_dim], 'o', color=ora, ms=6, label='Random BPM roll')
        ax[3*n_dim+2, 1].plot(s_bpm, 1e6 * roll_bpm_support[:, n_dim], '-', color=ora, label='BPM support roll')

    ax = _plot_annotations_and_limits(ax, x_lim)
    plt.pause(1)  # pause needed or grey figure
    fig.show()
    plt.rcParams.update({'font.size': init_font})


def _plot_support(axes, s_locs, vals, circ=None, **plot_kwargs):
    if s_locs[1] >= s_locs[0]:
        axes.plot(s_locs, vals, **plot_kwargs)
        return axes
    val_ring_end = vals[1] - s_locs[1] * (vals[1] - vals[0]) / (s_locs[1] - s_locs[0] + circ)
    axes.plot([s_locs[0], circ], [vals[0], val_ring_end], **plot_kwargs)
    axes.plot([0, s_locs[1]], [val_ring_end, vals[1]], **plot_kwargs)
    return axes


def _plot_annotations_and_limits(ax, x_lims):
    y_labels_per_dim = [[r'$\Delta x$ [$\mu$m]', r'$a_z$ [$\mu$rad]'],
                        [r'$\Delta y$ [$\mu$m]', r'$a_x$ [$\mu$rad]'],
                        [r'$\Delta z$ [$\mu$m]', r'$a_y$ [$\mu$rad]']]
    titles_per_dim = [['Horizontal Offsets', 'Roll (roll around z-axis)'],
                      ['Vertical Offsets', 'Pitch (roll around x-axis)'],
                      ['Longitudinal Offsets', 'Yaw (roll around y-axis)']]
    for row in range(9):
        for column in range(2):
            ax[row, column].legend(loc="upper right")
            ax[row, column].set_xlim(x_lims)
            ax[row, column].set_ylabel(y_labels_per_dim[row // 3][column])
            if row % 3 == 0:
                ax[row, column].set_title(titles_per_dim[row // 3][column])
    ax[8, 0].set_xlabel('$s$ [m]')
    ax[8, 1].set_xlabel('$s$ [m]')
    return ax
