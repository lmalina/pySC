import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
import warnings
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.plotting.plot_synoptic import plot_synoptic
from pySC.plotting.plot_apertures import plot_data_apertures, plot_data_beta_disp, baseplot
from pySC.utils import at_wrapper

warnings.filterwarnings("ignore", message='Legend does not support handles for PatchCollection instances.')


def plot_lattice(SC,
                 transfer_line: bool = False,
                 s_range: ndarray = None,
                 indices: ndarray = None,
                 n_sectors: int = None,
                 ideal_ring: bool = False,
                 plot_magnet_names: bool = False,
                 font_size: int = 8):
    """
    py:func:'plot_lattice' display lattice optics apertures and correctors locations

    Parameters:
        SC: the main simulated commissioning data structure
        transfer_line: If true the function 'twissline' is used to calculate the lattice functions
        s_range: Array ['sMin','sMax'] defining the plot range [m].
        indices: If `'s_range'` is empty, `'indices'`  can be used to specify a list of ordinates
        at which the lattice should be plotted
        n_sectors: If `'indices'` is empty, `'n_sectors'` can be used to plot only the first fraction of the lattice
        ideal_ring:  Specify if 'SC.IDEALRING' should be used to plot twiss functions, otherwise 'SC.RING'.
        plot_magnet_names: Specify if magnet names should be printed next to the magnets.
        font_size: Figure font size.

    Returns:


    Examples:
    Plots the complete lattice for a ring
    ------------------------------------------------------------------
    SCplotLattice(SC);
    -----------------------------------------------------------------
    Plots the lattice from ordinate 30 to 130 for a transfer line
    ------------------------------------------------------------------
    SCplotLattice(SC,transferLine=true,oList=range(30,120));
    ------------------------------------------------------------------

    Plots the lattice of one arc for a twelve-fold symmetric ring lattice
    ------------------------------------------------------------------
    SCplotLattice(SC,nSectors=12);
    ------------------------------------------------------------------
    """
    init_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': font_size})
    ring = SC.IDEALRING if ideal_ring else SC.RING
    s_range = _get_s_range(ring, s_range, indices, n_sectors)
    # assign s_range in lattice, to be used by plot_beta and plot_synopt
    initial_s_range = ring.s_range
    ring.s_range = s_range

    # define figure
    fig, (axtop, axcent, axbottom) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    for ax in (axtop, axcent, axbottom):
        ax.set_xlim(s_range)

    # get input twiss for transferline mode
    top_kwargs = dict(axes=(axtop, axtop.twinx()), s_range=s_range)
    if transfer_line and hasattr(ring[0], 'TD'):
        top_kwargs["twiss_in"] = ring[0].TD  # input optics of TL plot

    axtop, _ = baseplot(ring, plot_data_beta_disp, **top_kwargs)
    axsyntop = plot_synoptic(SC, axes=axtop)
    handles, labels = axsyntop.get_legend_handles_labels()
    axsyntop.legend(handles, labels, bbox_to_anchor=(0., 1.02, 1., .15), loc='lower center', ncols=5, borderaxespad=0.)

    # plot apertures within +- 10 cm
    axcent, _ = baseplot(ring, plot_data_apertures, axes=(axcent, None), s_range=s_range, cut_off=0.1)
    plot_synoptic(SC, axes=axcent, famnames=plot_magnet_names)
    axcent.set_ylim(np.array(axcent.get_ylim()) * 1.3)

    axsynbottom = plot_synoptic(SC, axes=axbottom, correctors=True)
    axsynbottom.legend(loc='upper center', ncol=4)
    axsynbottom.set_ylim([-2.5, 2.5])
    plot_synoptic(SC, axes=axbottom, correctors=False)
    axbottom.set_yticks([], [])
    axbottom.set_xlabel("s [m]")

    # restore initial s_range
    ring.s_range = initial_s_range

    plt.pause(1)  # MUST BE 1 second! less does not show figure
    fig.show()
    plt.rcParams.update({'font.size': init_font})


def plot_cm_strengths(SC: SimulatedCommissioning):
    init_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 18})
    f, ax = plt.subplots(nrows=2, num=86, figsize=(9, 7.5), facecolor="w")
    s_pos = at_wrapper.findspos(SC.RING)
    for n_dim in range(2):
        setpoints = 1E6 * SC.get_cm_setpoints(SC.ORD.CM[n_dim], skewness=bool(n_dim))
        count, bins_count = np.histogram(setpoints, bins=len(setpoints))
        ax[0].bar(s_pos[SC.ORD.CM[n_dim]], setpoints)
        ax[1].plot(bins_count[1:], np.cumsum(count / np.sum(count)), lw=3)

    ax[0].set_ylabel(r'CM strength [$\mu$rad]')
    ax[0].set_xlabel('s [m]')
    ax[1].set_xlabel(r'CM strength [$\mu$rad]')
    ax[1].set_ylabel('CDF')
    ax[1].legend(['Horizontal', 'Vertical'])
    ax[1].set_ylim([0, 1])
    f.tight_layout()
    f.show()
    plt.rcParams.update({'font.size': init_font})


def _get_s_range(ring, s_range, indices, n_sectors):
    """define s_range based on input. s_range has priority, then indices, then n_sectors"""
    if s_range is not None:
        return s_range
    if indices is not None:
        s_pos = at_wrapper.findspos(ring)
        return s_pos[np.min(indices)], s_pos[np.max(indices)]
    if n_sectors is not None:
        return 0, at_wrapper.findspos(ring)[-1] / n_sectors
    return ring.s_range
