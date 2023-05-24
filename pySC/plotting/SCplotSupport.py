import matplotlib.pyplot as plt
import numpy as np
import copy
from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.core.constants import SUPPORT_TYPES
from typing import Tuple
from pySC.core.classes import SimulatedComissioning


def SCplotSupport(SC: SimulatedComissioning, fontSize: int = 8, xLim: Tuple[float, float] = None):
    """
    Plots the offset and rolls of magnets, the support structure and BPMs.
    Specifically, this function plots the overall offsets [dx,dy,dz] and rolls [az,ax,ay] of all magnets and BPMs,
    as well as the individual contributions from different support structures (if registered).

    Args:
        `'SC'`:  :py:class: SimulatedCommissioning
            A base structure.
        `'fontSize'`: int
            Figure font size.
            Axes are rearranged for grouping. Depending on screen resolution this value may be adjusted.
        `'xLim'`: Tuple[float, float]
            Plot limits in terms of longitudinal location

    Returns:

    """

    if fontSize:
        plt.rcParams.update({'font.size': fontSize})
    # Get s - positions along the lattice
    s_pos = findspos(SC.RING, np.arange(len(SC.RING) + 1, dtype=int))
    circumference = s_pos[-1]
    if xLim is None:
        xLim = (0, circumference)

    s = np.linspace(xLim[0], xLim[1], 100 * int((xLim[1]-xLim[0])/2))  # s locations to compute
    s_mag = s_pos[SC.ORD.Magnet]
    s_bpm = s_pos[SC.ORD.BPM]

    # Loop over individual support structure types
    datadict = {'Ords': [], 'Off': {'a': [], 'b': []}, 'Roll': []}
    supdict = {'Section': copy.deepcopy(datadict), 'Plinth': copy.deepcopy(datadict), 'Girder': copy.deepcopy(datadict)}
    for sup_type in supdict.keys():
        for ordPair in SC.ORD[sup_type].transpose():
            if (xLim[0] <= s_pos[ordPair[0]] <= xLim[1]) or (xLim[0] <= s_pos[ordPair[1]] <= xLim[1]):
                # Structures in range
                supdict[sup_type]['Ords'].append(ordPair)
                # Get girder start and ending offsets
                supdict[sup_type]['Off']['a'].append(SC.RING[ordPair[0]].__dict__[sup_type + 'Offset'])
                supdict[sup_type]['Off']['b'].append(SC.RING[ordPair[1]].__dict__[sup_type + 'Offset'])
                # Get girder rolls
                supdict[sup_type]['Roll'].append(SC.RING[ordPair[0]].__dict__[sup_type + 'Roll'])

    # Magnet offsets and rolls
    offSupportLine, rollSupportLine = SC.support_offset_and_roll(s)
    offMagSupport = atgetfieldvalues(SC.RING, SC.ORD.Magnet, "SupportOffset")
    rollMagSupport = atgetfieldvalues(SC.RING, SC.ORD.Magnet, "SupportRoll")
    offMagInd = atgetfieldvalues(SC.RING, SC.ORD.Magnet, "MagnetOffset")
    rollMagInd = atgetfieldvalues(SC.RING, SC.ORD.Magnet, "MagnetRoll")
    offMagTot = atgetfieldvalues(SC.RING, SC.ORD.Magnet, "T2")[:, [0, 2, 5]]
    rollMagTot = rollMagSupport + rollMagInd

    # BPM offsets and rolls
    # Longitudinal offsets and Pitch and Yaw angles not supported for BPMs
    pad_off, pad_roll = ((0, 0), (0, 1)), ((0, 0), (0, 2))
    offBPM = np.pad(atgetfieldvalues(SC.RING, SC.ORD.BPM, "Offset"), pad_off)
    rollBPM = np.pad(atgetfieldvalues(SC.RING, SC.ORD.BPM, "Roll"), pad_roll)
    offBPMSupport = np.pad(atgetfieldvalues(SC.RING, SC.ORD.BPM, "SupportOffset"), pad_off)
    rollBPMSupport = np.pad(atgetfieldvalues(SC.RING, SC.ORD.BPM, "SupportRoll"), pad_roll)

    # create figure
    fig, ax = plt.subplots(nrows=9, ncols=2, num=1213, sharex="all", figsize=(10, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    blu, ora = '#1f77b4', '#ff7f0e'
    lineSpec = {'Plinth': {'color': 'r', 'linewidth': 4},
                'Section': {'color': '#e377c2', 'linewidth': 2, 'linestyle': ':'},
                'Girder': {'color': '#9467bd', 'linewidth': 4}}

    for nDim in range(3):
        # plot overall support offsets and rolls
        ax[3*nDim, 0].stairs(1e6*offSupportLine[nDim, :-1], s, color=blu)
        ax[3*nDim, 0].plot(s_mag, 1e6 * offMagSupport[:, nDim], 'D', color=blu, label='Overall supports')
        ax[3*nDim, 1].stairs(1e6 * rollSupportLine[nDim, :-1], s, color=blu)
        ax[3*nDim, 1].plot(s_mag, 1e6 * rollMagSupport[:, nDim], 'D', color=blu, label='Overall supports')
        # plot individual support and roll contributions
        for sup_type in SUPPORT_TYPES:
            for i, val in enumerate(SC.ORD[sup_type].T):  # loop supports
                ax[3*nDim, 0] = _plot_support(ax[3 * nDim, 0], s_pos[val],
                                              [1e6 * supdict[sup_type]['Off']['a'][i][nDim],
                                               1e6 * supdict[sup_type]['Off']['b'][i][nDim]],
                                              circumference,
                                              **lineSpec[sup_type])
                ax[3*nDim, 1] = _plot_support(ax[3 * nDim, 1], s_pos[val],
                                              1e6 * supdict[sup_type]['Roll'][i][nDim] * np.ones(2),
                                              circumference,
                                              **lineSpec[sup_type])

            # plot outside to get correct legend
            ax[3*nDim, 0].plot([-2, -1], [0, 0], **lineSpec[sup_type], label=f'Individual {sup_type}')
            ax[3*nDim, 1].plot([-2, -1], [0, 0], **lineSpec[sup_type], label=f'Individual {sup_type}')

        # plot magnet offset and roll
        ax[3*nDim+1, 0].plot(s_mag, 1e6 * offMagInd[:, nDim], 'kx', ms=8, label='Individual Magnet')
        ax[3*nDim+1, 0].plot(s_mag, 1e6 * offMagTot[:, nDim], 'ko-', label='Overall magnet offset')
        ax[3*nDim+1, 1].plot(s_mag, 1e6 * rollMagInd[:, nDim], 'kx', ms=8, label='Individual Magnet')
        ax[3*nDim+1, 1].plot(s_mag, 1e6 * rollMagTot[:, nDim], 'ko-', label='Overall magnet roll')
        # plot BPM offset and roll
        ax[3*nDim+2, 0].plot(s_bpm, 1e6 * offBPM[:, nDim], 'o', color=ora, ms=6, label='Random BPM offset')
        ax[3*nDim+2, 0].plot(s_bpm, 1e6 * offBPMSupport[:, nDim], '-', color=ora, label='BPM support offset')
        ax[3*nDim+2, 1].plot(s_bpm, 1e6 * rollBPM[:, nDim], 'o', color=ora, ms=6, label='Random BPM roll')
        ax[3*nDim+2, 1].plot(s_bpm, 1e6 * rollBPMSupport[:, nDim], '-', color=ora, label='BPM support roll')

    ax = _plot_annotations_and_limits(ax, xLim)
    plt.pause(1)  # pause needed or grey figure
    plt.show(block=False)


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
