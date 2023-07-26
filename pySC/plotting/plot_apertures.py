from itertools import chain, repeat
from typing import Callable

import numpy as np
from at import Lattice, Refpts
from matplotlib import pyplot as plt

from pySC.utils.at_wrapper import findspos, atlinopt

SLICES = 400


def plot_data_apertures(ring: Lattice, refpts: Refpts, **kwargs):
    """Generates data for plotting apertures"""
    to_mm = 1e3
    if refpts[-1] == len(ring):
        refpts = refpts[:-1]

    ea_all, ra_all = _apertures(ring)
    # Extract the plot data
    s_pos = findspos(ring)[refpts]
    ea = ea_all[refpts] * to_mm
    ra = ra_all[refpts] * to_mm
    if "cut_off" in kwargs.keys():
        ea = np.where(np.abs(ea) > kwargs["cut_off"] * to_mm, np.nan, ea)
        ra = np.where(np.abs(ra) > kwargs["cut_off"] * to_mm, np.nan, ra)
    # Joining left and right / top and bottom apertures such that they are plotted in a single colour
    double_s_pos = np.concatenate((s_pos, np.array([s_pos[-1], s_pos[0]]), s_pos))
    eliptic_a = np.concatenate((ea[:, 0, 0], np.full(2, np.nan), ea[:, 0, 1]))
    eliptic_b = np.concatenate((ea[:, 1, 0], np.full(2, np.nan), ea[:, 1, 1]))
    rectangular_x = np.concatenate((ra[:, 0, 0], np.full(2, np.nan), ra[:, 0, 1]))
    rectangular_y = np.concatenate((ra[:, 1, 0], np.full(2, np.nan), ra[:, 1, 1]))
    left = (r'Aperture [mm]', double_s_pos, [eliptic_a, eliptic_b, rectangular_x, rectangular_y],
            [r'elliptic$_{a}$', r'elliptic$_{b}$', r'rectangular$_{x}$', r'rectangular$_{y}$'])
    right = ()
    return left, right


def plot_data_beta_disp(ring: Lattice, refpts, **kwargs):
    """Generates data for plotting beta functions and dispersion"""
    _, _, data = atlinopt(ring, refpts=refpts, get_chrom=True, **kwargs)
    s_pos = data['s_pos']
    betax = data['beta'][:, 0]
    betay = data['beta'][:, 1]
    dispersion = data['dispersion'][:, 0]
    return ((r'$\beta$ [m]', s_pos, [betax, betay], [r'$\beta_x$', r'$\beta_y$']),
            ('dispersion [cm]', s_pos, [100*dispersion], [r'$D_x$']))


def baseplot(ring: Lattice, plot_function: Callable, axes, **kwargs):
    def plot1(ax, yaxis_label, x, y, labels=()):
        lines = []
        for y1, prop, label in zip(y, props, chain(labels, repeat(None))):
            ll = ax.plot(x, y1, **prop)
            if label is not None:
                ll[0].set_label(label)
            lines += ll
        ax.set_ylabel(yaxis_label)
        return lines

    def labeled(line):
        return not line.properties()['label'].startswith('_')

    # extract baseplot arguments
    slices = kwargs.pop('slices', SLICES)
    legend = kwargs.pop('legend', True)
    if 's_range' in kwargs:
        ring.s_range = kwargs.pop('s_range')

    cycle_props = plt.rcParams['axes.prop_cycle']
    rg = ring.slice(slices=slices)

    # get the data for the plot
    plots = plot_function(rg, rg.i_range, **kwargs)

    axleft, axright = axes
    nplots = 1 if axright is None else len(plots)

    props = iter(cycle_props())

    # left plot
    lines1 = plot1(axleft, *plots[0])
    # right plot
    lines2 = [] if (nplots < 2) else plot1(axright, *plots[1])
    if legend:
        if nplots < 2:
            axleft.legend(handles=[li for li in lines1 if labeled(li)])
        elif axleft.get_shared_x_axes().joined(axleft, axright):
            axleft.legend(handles=[li for li in lines1 + lines2 if labeled(li)])
        else:
            axleft.legend(handles=[li for li in lines1 if labeled(li)])
            axright.legend(handles=[li for li in lines2 if labeled(li)])
    return axleft, axright


def _apertures(ring):
    two_nans, four_nans = np.full((2,), np.nan), np.full((4,), np.nan)
    e_scaling = np.array([-1, 1])
    e_apertures = [np.outer(getattr(elem, "EApertures", two_nans), e_scaling) for elem in ring]
    r_apertures = [np.reshape(getattr(elem, 'RApertures', four_nans), (2, 2)) for elem in ring]
    return np.array(e_apertures), np.array(r_apertures)
