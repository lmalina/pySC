"""Lattice synoptics"""
import matplotlib.axes
import numpy
# noinspection PyPackageRequirements
import matplotlib.pyplot as plt
# noinspection PyPackageRequirements
from matplotlib.patches import Polygon
# noinspection PyPackageRequirements
from matplotlib.collections import PatchCollection
from at.lattice import Lattice, elements as elts

__all__ = ['plot_synopt']

# Default properties for element representation
DIPOLE = dict(label='Dipoles', facecolor=(0.5, 0.5, 1.0))
QUADRUPOLE = dict(label='Quadrupoles', facecolor=(1.0, 0.5, 0.5))
SEXTUPOLE = dict(label='Sextupoles', facecolor=(0.5, 1.0, 0.5))
MULTIPOLE = dict(label='Multipoles', facecolor=(0.25, 0.75, 0.25))
MONITOR = dict(label='Monitors', linestyle=None, marker=10, color='k')
HCORRECTOR = dict(label='Hor. Cor.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(0.0, 0.0, 0.7, 1), lw=2, refpts=[])
VCORRECTOR = dict(label='Ver. Cor.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(0.7, 0.0, 0.0, 1), lw=2, refpts=[])
SkewQUAD = dict(label='Skew Quad.', facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(75/256, 139/255, 39/255, 1), lw=2, refpts=[])


def plot_synopt(ring: Lattice,
                famnames: bool = False,
                axes: matplotlib.axes.Axes = None,
                dipole=DIPOLE,
                quadrupole=QUADRUPOLE,
                sextupole=SEXTUPOLE,
                multipole=MULTIPOLE,
                monitor=MONITOR,
                hcorrector=HCORRECTOR,
                vcorrector=VCORRECTOR,
                skewquadrupole=SkewQUAD,
                ):
    """Plots a synoptic of a lattice

    Parameters:
        ring:           Lattice description.
        axes:           :py:class:`~matplotlib.axes.Axes` for plotting the
          synoptic. If :py:obj:`None`, a new figure will be created. Otherwise,
          a new axes object sharing the same x-axis as the given one is created.
        dipole:         Dictionary of properties overloading the default
          properties. If :py:obj:`None`, dipoles will not be shown.
        quadrupole:     Same definition as for dipole
        sextupole:      Same definition as for dipole
        multipole:      Same definition as for dipole
        monitor:        Same definition as for dipole

    Returns:
        synopt_axes (Axes): Synoptic axes
     """

    class Dipole(Polygon):
        xx = numpy.array([0, 0, 1, 1], dtype=float)
        yy = numpy.array([0, 1, 1, 0], dtype=float)

        def __init__(self, s, length, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy), axis=1)
            super(Dipole, self).__init__(xy, closed=False, **kwargs)

    class HCor(Polygon):
        xx = numpy.array([0, 0, 0.9, 0.9], dtype=float)
        yy = numpy.array([0, 1.1, 1.1, 0], dtype=float)

        def __init__(self, s, length, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy), axis=1)
            super(HCor, self).__init__(xy, closed=False, **kwargs)

    class VCor(Polygon):
        xx = numpy.array([0, 0, 0.95, 0.95], dtype=float)
        yy = numpy.array([0, -1.05, -1.05, 0], dtype=float)

        def __init__(self, s, length, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy), axis=1)
            super(VCor, self).__init__(xy, closed=False, **kwargs)

    class Quadrupole(Polygon):
        xx = numpy.array([0, 0, 0.5, 1, 1])
        yy = {True: numpy.array([0, 1, 1.4, 1, 0]),
              False: numpy.array([0, 1, 0.6, 1, 0])}

        def __init__(self, s, length, foc, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy[foc]), axis=1)
            super(Quadrupole, self).__init__(xy, closed=False, **kwargs)

    class SkewQuadrupole(Polygon):
        xx = numpy.array([0, 0, 0.5, 0.9, 0.9])
        yy = {True: numpy.array([0, 1.1, 1.5, 1.1, 0]),
              False: numpy.array([0, 1.1, 0.7, 1.1, 0])}

        def __init__(self, s, length, foc, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy[foc]), axis=1)
            super(SkewQuadrupole, self).__init__(xy, closed=False, **kwargs)

    class Sextupole(Polygon):
        xx = numpy.array([0, 0, 0.33, 0.66, 1, 1])
        yy = {True: numpy.array([0, 0.8, 1, 1, 0.8, 0]),
              False: numpy.array([0, 0.8, 0.6, 0.6, 0.8, 0])}

        def __init__(self, s, length, foc, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy[foc]), axis=1)
            super(Sextupole, self).__init__(xy, closed=False, **kwargs)

    class Multipole(Polygon):
        xx = numpy.array([0, 0, 1, 1], dtype=float)
        yy = numpy.array([0, 0.8, 0.8, 0])

        def __init__(self, s, length, **kwargs):
            xy = numpy.stack((self.xx * length + s, self.yy), axis=1)
            super(Multipole, self).__init__(xy, closed=False, **kwargs)

    class Monitor(Polygon):
        xx = numpy.array([0.0, 0.0])
        yy = numpy.array([0.0, 1.2])

        def __init__(self, s, **kwargs):
            xy = numpy.stack((self.xx + s, self.yy), axis=1)
            super(Monitor, self).__init__(xy, closed=False, **kwargs)

    def ismultipole(elem):
        return isinstance(elem, elts.Multipole) and not isinstance(elem, (
            elts.Dipole, elts.Quadrupole, elts.Sextupole))

    if axes is None:
        fig = plt.figure()
        axsyn = fig.add_subplot(111, xlim=ring.s_range)

    else:
        axsyn = axes.twinx()

    axsyn.set_axis_off()  # Set axis invisible
    axsyn.set_ylim((0.0, 20.0))  # Initial scaling of elements

    if axes is None:
        axsyn.set_zorder(-0.2)       # Put synoptic in the background

    s_pos = ring.get_s_pos(range(len(ring)))

    if famnames:
        #  get fam name of each main multipole
        for count, el in enumerate(ring):
            s = s_pos[count]
            if hasattr(el, 'PolynomB'):
                axsyn.annotate(" " + el.FamName[0:10], xy=[s+el.Length/4, 0],
                               xycoords='data',
                               xytext=(1.5, 1.5),
                               textcoords='offset points',
                               rotation=90)


    if dipole is not None:
        # print('plot dipoles')

        props = DIPOLE.copy()
        props.update(dipole)
        dipoles = PatchCollection(
            (Dipole(s, el.Length) for s, el in zip(s_pos, ring)
             if isinstance(el, elts.Dipole)), **props)
        axsyn.add_collection(dipoles)
        axsyn.bar(0, 0, facecolor=props['facecolor'], label=props['label'])


    if quadrupole is not None:
        # print('plot quads')
        props = QUADRUPOLE.copy()
        props.update(quadrupole)
        quadrupoles = PatchCollection(
            (Quadrupole(s, el.Length, el.PolynomB[1] >= 0.0)
             for s, el in zip(s_pos, ring)
             if isinstance(el, elts.Quadrupole)), **props)
        axsyn.add_collection(quadrupoles)
        axsyn.bar(0, 0, facecolor=props['facecolor'], label=props['label'])

    if sextupole is not None:
        # print('plot sextupoles')

        props = SEXTUPOLE.copy()
        props.update(sextupole)
        sextupoles = PatchCollection(
            (Sextupole(s, el.Length, el.PolynomB[2] >= 0.0)
             for s, el in zip(s_pos, ring)
             if isinstance(el, elts.Sextupole)), **props)
        axsyn.add_collection(sextupoles)
        axsyn.bar(0, 0, facecolor=props['facecolor'], label=props['label'])

    if multipole is not None:
        # print('plot multipoles')

        props = MULTIPOLE.copy()
        props.update(multipole)
        multipoles = PatchCollection(
            (Multipole(s, el.Length) for s, el in zip(s_pos, ring)
             if ismultipole(el)), **props)
        axsyn.add_collection(multipoles)
        axsyn.bar(0, 0, facecolor=props['facecolor'], label=props['label'])

    if monitor is not None:
        # print('plot monitors')

        props = MONITOR.copy()
        props.update(monitor)
        s = s_pos[[isinstance(el, elts.Monitor) for el in ring]]
        y = numpy.zeros(s.shape)
        # noinspection PyUnusedLocal
        monitors = axsyn.plot(s, y, **props)
        
    if hcorrector is None:
        props = HCORRECTOR.copy()
    else:
        props = hcorrector.copy()

    # print('plot hor correctors')

    refpts = props.pop('refpts')
    #     props.update(hcorrector)
    hcor = PatchCollection(
        (HCor(s, el.Length) for s, el in zip(s_pos[refpts], ring[refpts])
         ), **props)
    axsyn.add_collection(hcor)
    if len(refpts)>0:
        axsyn.bar(0, 0, facecolor=props['edgecolor'], label=props['label'])

    if vcorrector is None:
        props = VCORRECTOR.copy()
    else:
        props = vcorrector.copy()
    # print('plot Ver correctors')

    refpts = props.pop('refpts')
    # props.update(vcorrector)
    vcor = PatchCollection(
        (VCor(s, el.Length) for s, el in zip(s_pos[refpts], ring[refpts])
         ), **props)
    axsyn.add_collection(vcor)
    if len(refpts) > 0:
        axsyn.bar(0, 0, facecolor=props['edgecolor'], label=props['label'])

    if skewquadrupole is None:
        props = SkewQUADRUPOLE.copy()
    else:
        props = skewquadrupole.copy()
    # print('plot skew quad correctors')

    refpts = props.pop('refpts')
    # props.update(skewquadrupole)
    squadrupoles = PatchCollection(
        (SkewQuadrupole(s, el.Length, el.PolynomB[1] >= 0.0)
         for s, el in zip(s_pos[refpts], ring[refpts])
         ), **props)
    axsyn.add_collection(squadrupoles)
    if len(refpts) > 0:
        axsyn.bar(0, 0, facecolor=props['edgecolor'], label=props['label'])

    return axsyn