"""Lattice synoptics"""
import matplotlib.axes
import numpy as np
from at.lattice import elements as elts
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pySC.utils.at_wrapper import findspos

MONITOR = "monitor"
NORMAL_KWARGS = ("dipole", "quadrupole", "sextupole", "multipole", MONITOR)
CORRECTOR_KWARGS = ("hcor", "vcor", "skewquadrupole", MONITOR)


def plot_synoptic(SC, axes: matplotlib.axes.Axes = None, correctors: bool = False, famnames: bool = False):
    """Plots magnet or corrector synoptic with monitors"""
    ring = SC.IDEALRING

    if axes is None:
        fig = plt.figure()
        axsyn = fig.add_subplot(111, xlim=ring.s_range)
    else:
        axsyn = axes.twinx()
    axsyn.set_axis_off()  # Set axis invisible
    axsyn.set_ylim((0.0, 10.0))  # Initial scaling of elements
    if axes is None:
        axsyn.set_zorder(-0.2)       # Put synoptic in the background

    s_pos = findspos(ring)[:-1]

    if famnames:
        #  get fam name of each main multipole
        for count, el in enumerate(ring):
            if hasattr(el, 'PolynomB'):
                axsyn.annotate("   " + el.FamName[0:10], xy=[s_pos[count] + el.Length/4, 0], xycoords='data',
                               xytext=(1.5, 1.5), textcoords='offset points', rotation=90)

    def _is_multipole(elem):
        return isinstance(elem, elts.Multipole) and not isinstance(elem, (elts.Dipole, elts.Quadrupole, elts.Sextupole))

    if correctors:
        patch_collections = dict(
            monitor=(_get_shape(s, getattr(el, "Length", 0), mtype="monitor")
                     for s, el in zip(s_pos[SC.ORD.BPM], ring[SC.ORD.BPM])),
            skewquadrupole=(_get_shape(s, el.Length, mtype=f"{'f' if el.PolynomA[1] >= 0.0 else 'd'}_skewquadrupole")
                            for s, el in zip(s_pos[SC.ORD.SkewQuad], ring[SC.ORD.SkewQuad])),
            hcor=(_get_shape(s, el.Length, mtype="hcor") for s, el in zip(s_pos[SC.ORD.HCM], ring[SC.ORD.HCM])),
            vcor=(_get_shape(s, el.Length, mtype="vcor") for s, el in zip(s_pos[SC.ORD.VCM], ring[SC.ORD.VCM]))
        )
        for key in CORRECTOR_KWARGS:
            props = _plotting_properties(key)  # TODO possibly modify the style?
            axsyn.add_collection(PatchCollection(patch_collections[key], **props))
            axsyn.bar(0, 0, facecolor=props['edgecolor'], label=props['label'])
        s_bpm = s_pos[SC.ORD.BPM]
    else:
        patch_collections = dict(
            dipole=(_get_shape(s, el.Length, mtype="dipole")
                    for s, el in zip(s_pos, ring) if isinstance(el, elts.Dipole)),
            quadrupole=(_get_shape(s, el.Length, mtype=f"{'f' if el.PolynomB[1] >= 0.0 else 'd'}_quadrupole")
                        for s, el in zip(s_pos, ring) if isinstance(el, elts.Quadrupole)),
            sextupole=(_get_shape(s, el.Length, mtype=f"{'f' if el.PolynomB[2] >= 0.0 else 'd'}_sextupole")
                       for s, el in zip(s_pos, ring) if isinstance(el, elts.Sextupole)),
            multipole=(_get_shape(s, el.Length) for s, el in zip(s_pos, ring) if _is_multipole(el)),
            monitor=(_get_shape(s, getattr(el, "Length", 0), mtype="monitor") for s, el in zip(s_pos, ring) if
                     isinstance(el, elts.Monitor)),
        )
        for key in NORMAL_KWARGS:
            props = _plotting_properties(key)  # TODO possibly modify the style?
            coll = PatchCollection(patch_collections[key], **props)
            axsyn.add_collection(coll)
            axsyn.bar(0, 0, facecolor=props['facecolor'], label=props['label'])
        s_bpm = s_pos[[isinstance(el, elts.Monitor) for el in ring]]
    axsyn.plot(s_bpm, np.zeros(s_bpm.shape), marker=10, color='k', lw=0)
    return axsyn


def _plotting_properties(element_type):
    plotting_properties = dict(
        dipole=dict(label='Dipoles', facecolor=(0.5, 0.5, 1.0)),
        quadrupole=dict(label='Quadrupoles', facecolor=(1.0, 0.5, 0.5)),
        sextupole=dict(label='Sextupoles', facecolor=(0.5, 1.0, 0.5)),
        multipole=dict(label='Multipoles', facecolor=(0.25, 0.75, 0.25)),
        monitor=dict(label='Monitors', facecolor=(0.0, 0.0, 0.0, 1.0), edgecolor=(0.0, 0.0, 0.0, 1), lw=2),
        hcor=dict(label='Hor. Cor.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(0.0, 0.0, 0.7, 1), lw=2),
        vcor=dict(label='Ver. Cor.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(0.7, 0.0, 0.0, 1), lw=2),
        skewquadrupole=dict(label='Skew Quad.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(75 / 256, 139 / 255, 39 / 255, 1), lw=2)
         )
    return plotting_properties[element_type].copy()


def _get_shape(s, length, mtype="monitor", **kwargs):
    pshape = dict(
        dipole=np.array([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=float),
        hcor=np.array([[0, 0, 0.9, 0.9], [0, 1.1, 1.1, 0]], dtype=float),
        vcor=np.array([[0, 0, 0.95, 0.95], [0, -1.05, -1.05, 0]], dtype=float),
        f_quadrupole=np.array([[0, 0, 0.5, 1, 1], [0, 1, 1.4, 1, 0]], dtype=float),
        d_quadrupole=np.array([[0, 0, 0.5, 1, 1], [0, 1, 0.6, 1, 0]], dtype=float),
        f_skewquadrupole=np.array([[0, 0, 0.5, 0.9, 0.9], [0, 1.1, 1.5, 1.1, 0]], dtype=float),
        d_skewquadrupole=np.array([[0, 0, 0.5, 0.9, 0.9], [0, 1.1, 0.7, 1.1, 0]], dtype=float),
        f_sextupole=np.array([[0, 0, 0.33, 0.66, 1, 1], [0, 0.8, 1, 1, 0.8, 0]], dtype=float),
        d_sextupole=np.array([[0, 0, 0.33, 0.66, 1, 1], [0, 0.8, 0.6, 0.6, 0.8, 0]], dtype=float),
        multipole=np.array([[0, 0, 1, 1], [0, 0.8, 0.8, 0]], dtype=float),
        monitor=np.array([[0, 0], [0, 1.2]], dtype=float),
    )
    return Polygon(pshape[mtype].T * np.array([length, 1]) + np.array([s, 0]), closed=False, **kwargs)
