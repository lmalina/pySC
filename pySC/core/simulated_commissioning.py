"""
Simulated Commissioning
------------------------

This module contains the main data structure of ``pySC`` package
built up around the ``at.Lattice`` under study.
"""
import copy
import re
from typing import Tuple, Union, List

import numpy as np
from at import Lattice
from numpy import ndarray

from pySC.core.classes import Injection, Sigmas, Indices, DotDict
from pySC.core.constants import (BPM_ERROR_FIELDS, RF_ERROR_FIELDS, RF_PROPERTIES, MAGNET_TYPE_FIELDS,
                                 MAGNET_ERROR_FIELDS, AB, SUPPORT_TYPES, SUPPORT_ERROR_FIELDS, SETTING_ABS, SETTING_REL,
                                 SETTING_ADD, NUM_TO_AB, SETPOINT)
from pySC.utils import logging_tools
from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.utils.classdef_tools import update_double_ordinates, add_padded, intersect, randn_cutoff, s_interpolation, \
    check_input_and_setpoints
from pySC.utils.sc_tools import SCrandnc, SCscaleCircumference, update_transformation

LOGGER = logging_tools.get_logger(__name__)


class SimulatedCommissioning:
    """
    The main structure of ``pySC``, which holds all the information about
    lattice error sources and errors, injection settings and its errors.
    The class is initialized from ``at.Lattice``.

    SC.register* functions assign uncertainties
    SC.apply_errors functions apply those sigmas to specific elements in the lattice
    SC.update* functions transfer the information to the AT elements fields

    Examples:

        >>> import at
        >>> SC = SimulatedCommissioning(at.Lattice(at.Monitor('BPM'))
        >>> SC.register_bpms(np.array(0), Offset=2e-6*np.ones(2))
        >>> print(SC.ORD)
        0
        >>> print(SC.SIG.BPM[0].Offset)
        >>> SC.apply_errors()

    The properties of the class are:
        RING:
            the AT ring lattice to which the errors will be added
        IDEALRING:
            the ideal AT ring lattice
        INJ:
            Class for definition of injection properties.
            See Also *pySC.core.classes.Injection*
        SIG:
            Class to store the error sigma's.
            This parameter is set via *register_magnets*, *register_bpms*, *register_cavities*, *register_supports*
            See Also *pySC.core.classes.Sigmas*
        ORD:
            Class to store the indices of registered elements in the lattice
            This parameter is set via *register_magnets*, *register_bpms*, *register_cavities*, *register_supports*
            See Also *pySC.core.classes.Indices*
        plot:
            (default=False) a boolean flag to trigger plots
    """
    def __init__(self, ring: Lattice):
        self.RING: Lattice = ring.deepcopy()
        self.IDEALRING: Lattice = ring.deepcopy()
        for ind, element in enumerate(ring):
            self.RING[ind] = element.deepcopy()
            self.IDEALRING[ind] = element.deepcopy()

        self.INJ: Injection = Injection()
        self.SIG: Sigmas = Sigmas()
        self.ORD: Indices = Indices()
        self.plot: bool = False

    def register_bpms(self, ords: ndarray, **kwargs):
        """registers BPMs specified by the `ords` (element indices in the lattice) in the `SC` class instance
        and initializes all required fields in the lattice elements. The ordinates of all registered BPMs are stored in
        `SC.ORD.BPM`.

        Args:
            ords:  BPM ordinates in the lattice structure.

        The BPM fields in the lattice elements are:
            Noise:
                2 elements array of hor./ver. turn-by-turn BPM noise uncertainties (sigmas)
            NoiseCO:
                2 elements array of hor./ver. orbit BPM noise uncertainties (sigmas)
            CalError:
                2 elements array of hor./ver. BPM calibration errors uncertainties (sigmas)
            Offset:
                2 elements array of individual hor./ver. BPM offsets uncertainties (sigmas)
            Roll:
                BPM roll around z-axis w.r.t. the support structure
            SumError:
                Calibration error of the sum signal. The sum signal is used to determine
                the beam loss location with a cutoff as defined `SC.INJ.beamLostAt`.

        Examples:
            Identify the ordinates of all elements named `BPM` and registers them as BPMs in `SC`::

                ords = SCgetOrds(SC.RING,'BPM')
                SC.register_bpms(ords)

            Register the BPMs specified in `ords` in `SC` and set the uncertainty of the offset to `500um` in
            both planes. A subsequent call of *SC.apply_errors* would generate a random BPM offset errors with
            `sigma=500um`::

                SC.register_bpms(ords, Offset=500E-6*np.ones(2))

            Register the BPMs specified in `ords` in `SC` and set the uncertainty of the offset to `500um` in
            both planes. A subsequent call of *SC.apply_errors* would generate a random BPM offset errors with
            `sigma=500um` and a cutoff at 3 sigmas for this error::

                SC.register_bpms(ords, Offset=[500E-6*np.ones(2), 3])

            Register the BPMs specified in `ords` in `SC` and set the uncertainty of the offset to `500um` in
            both planes and a calibration error of the sum signal of 20%::

                SC.register_bpms(ords, Offset=500E-6*np.ones(2), SumError=0.2)

        See also:
            *bpm_reading*, *SCgetOrds*, *SC.verify_structure*, *SC.apply_errors*, *SC.register_support*, *SC.update_support*
        """
        self._check_kwargs(kwargs, BPM_ERROR_FIELDS)
        self.ORD.BPM = np.unique(np.concatenate((self.ORD.BPM, ords)))
        for ind in np.unique(ords):
            if ind not in self.SIG.BPM.keys():
                self.SIG.BPM[ind] = DotDict()
            self.SIG.BPM[ind].update(kwargs)

            self.RING[ind].Noise = np.zeros(2)
            self.RING[ind].NoiseCO = np.zeros(2)
            self.RING[ind].Offset = np.zeros(2)
            self.RING[ind].SupportOffset = np.zeros(2)
            self.RING[ind].Roll = 0
            self.RING[ind].SupportRoll = 0
            self.RING[ind].CalError = np.zeros(2)
            self.RING[ind].SumError = 0

    def register_cavities(self, ords: ndarray, **kwargs):
        """Register cavities specified in `ords` in `SC` by initializing all required fields in the
        corresponding cavity lattice elements and storing the ordinates in `SC.ORD.RF`.

        Args:
            ords: Cavity ordinates in the lattice structure.

        Keyword Args:
            VoltageOffset:
                Offset of cavity voltage wrt. to the setpoint
            VoltageCalError:
                Calibration error of cavity voltage wrt. to the setpoint
            FrequencyOffset:
                Offset of cavity frequency wrt. to the setpoint
            FrequencyCalError:
                Calibration error of cavity frequency wrt. to the setpoint
            TimeLagOffset:
                Offset of cavity phase wrt. to the setpoint
            TimeLagCalError:
                Calibration error of cavity phase wrt. to the setpoint

        Examples:
            Identify the ordinates of all elements named `'CAV'` and register them as cavities in `SC`::

                ords = SCgetOrds(SC.RING, 'CAV')
                SC.register_cavities(ords)

            Register the cavities specified in `ords` in `SC` and sets the uncertainty of the frequency offset
            to 1kHz. A subsequent call of *SC.apply_errors* would generate a random frequency offset error with `sigma=1kHz`::

                SC.register_cavities(ords, FrequencyOffset=1E3)

            Register the cavities specified in `ords` in `SC` and sets the uncertainty of the frequency offset
            to 1kHz. A subsequent call of *SC.apply_errors* would generate a random frequency offset error with
            `sigma=1kHz` and a random timelag offset error ('phase error') with `sigma=0.3m`::

                SC.register_cavities(ords, FrequencyOffset=1E3, TimeLagOffset=0.3)

        See also:
            *SCgetOrds*, *SC.verify_structure*, *SC.apply_errors*

        """
        self._check_kwargs(kwargs, RF_ERROR_FIELDS)
        self.ORD.RF = np.unique(np.concatenate((self.ORD.RF, ords)))
        for ind in np.unique(ords):
            if ind not in self.SIG.RF.keys():
                self.SIG.RF[ind] = DotDict()
            self.SIG.RF[ind].update(kwargs)
            for field in RF_PROPERTIES:
                setattr(self.RING[ind], f"{field}SetPoint", getattr(self.RING[ind], field))
                setattr(self.RING[ind], f"{field}Offset", 0)
                setattr(self.RING[ind], f"{field}CalError", 0)

    def register_magnets(self, ords: ndarray, **kwargs):  # TODO docstring is too long also verify all relates to Python
        """
        Registers magnets specified by `ords` in the `SC` structure and initializes all required fields
        in the lattice elements. The ordinates of all registered magnets are stored in `SC.ORD.Magnet`.

        Args:
            ords: Magnet ordinates in the lattice structure.

        Keyword Args:
            CalErrorB:
                Calibration error of the `PolynomB` fields wrt. the corresponding setpoints.
                Each element of the numpy.array is the error for the corresponding element in 'PolynomB'
            CalErrorA:
                Calibration error of the `PolynomA` fields wrt. the corresponding setpoints.
                Each element of the numpy.array is the error for the corresponding element in 'PolynomA'
            MagnetOffset:
                3 element array of horizontal, vertical and longitudinal magnet offsets (wrt. the support structure).
            MagnetRoll:
                3 element array [az,ax,ay] defineing magnet roll (around z-axis), pitch (roll around x-axis) and yaw (roll around
                y-axis); all wrt. the support structure.
            BendingAngleError :
                Error of the main bending field (corresponding uncertainty defined with `BendingAngle`).
            CF :
                Flag identifying the corresponding magnet as a combined function dipole/quadrupole. That implies that the
                bending angle depends on the quadrupole setpoint. A variation from the design value will therefore result in a
                bending angle error which is added to the `PolynomB[0]` field.
            HCM :
                Flag identifying the corresponding magnet as a horizontal corrector magnet. The corresponding value is the
                horizontal CM limit and stored in `SC.RING[ords].CMlimit[0]`. E.g. set limit to `Inf`.
            VCM :
                Flag identifying the corresponding magnet as a vertical corrector magnet. The corresponding value is the
                vertical CM limit and stored in `SC.RING[ords].CMlimit[1]`. E.g. set limit to `Inf`.
            SkewQuad :
                Flag identifying the corresponding magnet as a skew quadrupole corrector magnet. The corresponding value
                is the skew quadrupole limit and stored in `SC.RING[ords].SkewLimit`. E.g. set limit to `Inf`.
            MasterOf :
                Array of ordinates to which the corresponding magnet acts as master (split magnets).
                The magnets at ordinates `ords` are identified as a split magnets each with `N` childs as specified in
                the corresponding value which must be a [`N` x `length(ords)`] array.
                The field calculation in *SC.update_magnets* uses the setpoints and errors of the master magnet to
                calculate the child fields.
                The relative bending angle error of the master magnet e.g. is applied
                on the corresponding child bending angle appropriately.
                Split quadrupole magnets with different design gradients, however, can
                currently not be updated correctly.

        If CMs or skew quadrupole correctors are specified, the ordinates are also
        stored in the corresponding fields `SC.ORD.CM` and `SC.ORD.SkewQuad`, respectively.

        Examples:
            Identify the ordinates of all elements named `QF` and register them in `SC`::

                ords = SCgetOrds(SC.RING, 'QF')
                SC.register_magnets(ords)

            Register the magnets specified in `ords` in `SC` and set the uncertainty of
            the quadrupole component to 1E-3 and 30um horizontal and vertical offset::

                SC.register_magnets(ords,
                                    CalErrorB=np.array([0, 1E-3]),
                                    MagnetOffset=np.array([30E-6, 30E-6 0]))

            Register the magnets specified in `ords` in `SC` and set the uncertainty of
            the quadrupole component to 1E-3, 30um horizontal and vertical offset and
            100um longitudinal offset::

                SC.register_magnets(ords,
                                    CalErrorB=np.array([0, 1E-3]),
                                    MagnetOffset=np.array([30E-6, 30E-6, 100E-6]))

            Register the magnets specified in `ords` in `SC` and set the uncertainty of
            the roll, pitch and yaw angle to 100urad::

                SC.register_magnets(ords, Roll=np.array([100E-6, 100E-6, 100E-6]))

            Register the magnets specified in `ords` in `SC` and set the uncertainty of
            the roll, pitch and yaw angle to 100urad and 3.4 sigmas cutoff::

                SC.register_magnets(ords, Roll=[100E-6*np.ones(3), 3.4])

            Register split magnets.
            Identify the magnets named `BENDa` ([`1xN`] array `masterOrds`) and the
            magnets named `BENDb` and `BENDc` ([`2xN`] array `childOrds`) and register
            the `masterOrds` as the master magnets of the children in the corresponding
            columns of `childOrds`.
            The uncertainty of the bending angle is set to 1E-4::

                masterOrds = SCgetOrds(SC.RING,'BENDa')
                childOrds  = numpy.vstack((SCgetOrds(SC.RING,'BENDb'), SCgetOrds(SC.RING,'BENDc')))
                SC.register_magnets(masterOrds, BendingAngle=1E-4, MasterOf=childOrds)

            Register the magnets specified in `ords` in `SC` as combined function magnets
            and sets the uncertainty of the quadrupole component to 1E-3::

                SC.register_magnets(ords, CF=1, CalErrorB=np.array([0, 1E-3]))

            Register the magnets specified in `ords` in `SC` and set the uncertainty of
            the skew quadrupole component to 2E-3 and the uncertainty of the sextupole
            component to 1E-3::

                SC.register_magnets(ords, CalErrorA=np.array([0, 2E-3, 0]), CalErrorB=np.array([0, 0, 1E-3]))

            Register the magnets specified in `ords` in `SC` as horizontal and vertical
            CMs, set their dipole uncertainties to 5% and 1%, respectively and define no CM limits::

                SC.register_magnets(ords, HCM=np.inf, VCM=np.inf, CalErrorB=5E-2, CalErrorA=1E-2)

            Register the magnets specified in `ords` in `SC` as horizontal and vertical
            CMs, set their uncertainties to 5% and 1%, respectively and their limits to 1
            mrad. Furthermore, set the uncertainty of the skew quadrupole component to
            2E-3 and the uncertainty of the sextupole component to 1E-3::

                SC.register_magnets(ords,
                                    HCM=1E-3,
                                    VCM=1E-3,
                                    CalErrorB=np.array([5E-2, 0, 1E-3]),
                                    CalErrorA=np.array([1E-2, 2E-3, 0]))

        See Also:
            *SCgetOrds*, *SC.update_magnets*, *SC.verify_structure*, *SC.apply_errors*, *SC.register_support*

        """
        self._check_kwargs(kwargs, MAGNET_TYPE_FIELDS + MAGNET_ERROR_FIELDS)
        nvpairs = {key: value for key, value in kwargs.items() if key not in MAGNET_TYPE_FIELDS}
        self.ORD.Magnet = np.unique(np.concatenate((self.ORD.Magnet, ords)))
        if 'SkewQuad' in kwargs.keys():
            self.ORD.SkewQuad = np.unique(np.concatenate((self.ORD.SkewQuad, ords)))
        if 'HCM' in kwargs.keys():
            self.ORD.HCM = np.unique(np.concatenate((self.ORD.HCM, ords)))
        if 'VCM' in kwargs.keys():
            self.ORD.VCM = np.unique(np.concatenate((self.ORD.VCM, ords)))
        for ind in ords:
            if ind not in self.SIG.Magnet.keys():
                self.SIG.Magnet[ind] = DotDict()
            self.SIG.Magnet[ind].update(nvpairs)
        # TODO Correctors
            for ab in AB:
                order = len(getattr(self.RING[ind], f"Polynom{ab}"))
                for field in ("NomPolynom", "SetPoint", "CalError"):
                    setattr(self.RING[ind], f"{field}{ab}", np.zeros(order))
            self.RING[ind].NomPolynomB += self.RING[ind].PolynomB
            self.RING[ind].NomPolynomA += self.RING[ind].PolynomA
            self.RING[ind].SetPointB += self.RING[ind].PolynomB
            self.RING[ind].SetPointA += self.RING[ind].PolynomA
            self.RING[ind].MagnetOffset = np.zeros(3)
            self.RING[ind].SupportOffset = np.zeros(3)
            self.RING[ind].MagnetRoll = np.zeros(3)
            self.RING[ind].SupportRoll = np.zeros(3)
            self.RING[ind].T1 = np.zeros(6)
            self.RING[ind].T2 = np.zeros(6)
            self._optional_magnet_fields(ind, ords, **kwargs)

    def register_supports(self, support_ords: ndarray, support_type: str, **kwargs):
        """Initializes magnet support structures such as sections, plinths and girders in SC. The function
        input be given as name-value pairs, starting with the structure type and structure ordinates
        defining start-end endpoints. Optional arguments are set as the uncertainties of e.g. girder
        offsets in the sigma structure `SC.SIG.Support`.

        Args:
            support_ords: [2xN] array of ordinates defining start and end locations of `N` registered support structures
            support_type: String specifying the support structure type. Valid are 'Plinth', 'Girder' or 'Section'
            **kwargs: any of those listed below

        Keyword Args:
            Offset:
                A [1x3] array defining horizontal, vertical and longitudinal offset uncertainties for the start
                points or [2x3] array defining horizontal, vertical and longitudinal offset uncertainties for
                the start end endpoints. If end points have dedicated uncertainties, *SC.apply_errors* applies
                random offset errors of both start end endpoints of the corresponding support structure,
                effectively tilting the support structure.
                If only start points have asigned uncertainties, *SC.apply_errors* applies to the support
                structure endpoints the same offset error as to the start points, resulting in a paraxial
                translation of the element. Only in this case dedicated `'Roll'` uncertainties may be given which
                then tilt the structure around it's center.
                The actual magnet or BPM offsets resulting from the support structure offsets is calculated in
                *SC.update_support* by interpolating on a straight line between girder start- and endpoints. Note
                that the coordinate system change due to bending magnets are ignored in this calculation. Thus,
                the accuracy of the result is limited if dipole magnets are involved. This may be particularly
                true in case of large sections and/or longitudinal offsets.
            Roll:
                [1x3] array [az,ax,ay] defining roll (around z-axis), pitch (roll around x-axis) and yaw (roll
                around y-axis) angle uncertainties.

        Examples:
            Registers the girder start end endpoints defined in `ords` and assigns the horizontal,
            vertical and longitudinal girder offset uncertainties `dX`, `dY` and `dZ`, respectively, to the
            girder start points. When the support errors are applied the girder endpoints will get the same
            offset error as the start points, resulting in a paraxial translation of the girder::

                SC.register_support(ords, "Girder", Offset=[dX, dY, dZ])

            Registers the section start- end endpoints defined in `ords` and assigns the horizontal and
            vertical section offset uncertainties `dX` and `dY`, respectively, to the start points. When
            the support errors are applied the section endpoints will get the same offset as the start points::

                SC.register_support(ords, "Section", Offset=np.array([dX, dY, 0]))

            Registers the section start- end endpoints defined in `ords` and assigns the horizontal and
            vertical section offset uncertainties `dX` and `dY`, respectively, to the start points. When
            the support errors are applied the section endpoints will get the same offset as the start points.
            Also set a 4.2 sigmas cutoff::

                SC.register_support(ords, "Section", Offset=[np.array([dX, dY, 0]), 4.2])

            Registers the girder start end endpoints defined in `ords`, assigns the roll uncertainty `dPhi`
            and the horizontal and vertical girder offset uncertainties `dX1` and `dY1`, respectively to the
            start points and `dX2` and `dY2` to the endpoints. When the support errors are applied, all
            girder start- and endpoints will get random offset errors and the resulting yaw and pitch angles
            are calculated accordingly::

                SC.register_support(ords, "Girder",
                                    Offset=np.array([dX1, dY1, 0; dX2, dY2, 0]),
                                    Roll=np.array([dPhi, 0, 0]))

            Registers the girder start end endpoints defined in `ords` and assigns the horizontal,
            vertical and longitudinal girder offset uncertainties `dX`, `dY` and `dZ`, respectively, and the
            roll, pitch and yaw angle uncertainties `az`, `ax` and `ay`. When the support errors are applied
            the girders will experience a paraxial translation according to the offsets plus the proper
            rotations around the three x-, y- and z-axes::

                SC.register_support(ords, "Girder", Offset=np.array([dX dY dZ]), Roll=np.array([az ax ay]));

        See Also:
            *SCgetOrds*, *SC.update_support*, *SC.support_offset_and_roll*, *plot_support*, *SC.apply_errors*,
            *SC.register_magnets*, *update_transformation*

        """
        if support_type not in SUPPORT_TYPES:
            raise ValueError(f'Unknown support type ``{support_type}`` found. Allowed are {SUPPORT_TYPES}.')
        self._check_kwargs(kwargs, SUPPORT_ERROR_FIELDS)
        if not len(support_ords) or support_ords.shape[0] != 2:
            raise ValueError('Ordinates must be a 2xn array of ordinates.')
        if upstream := np.sum(np.diff(support_ords, axis=0) < 0):
            LOGGER.warning(f"{upstream} {support_type} endpoints(s) may be upstream of startpoint(s).")
        # TODO check the dimensions of Roll and Offset values
        self.ORD[support_type] = update_double_ordinates(self.ORD[support_type], support_ords)
        for ind in np.ravel(support_ords):
            setattr(self.RING[ind], f"{support_type}Offset", np.zeros(3))  # [x,y,z]
            setattr(self.RING[ind], f"{support_type}Roll", np.zeros(3))  # [az,ax,ay]
            self.SIG.Support[ind] = DotDict()
        for ord_pair in support_ords.T:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    if value[0].ndim == 1:
                        self.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value
                    else:
                        self.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = [value[0][0, :], value[1]]
                        self.SIG.Support[ord_pair[1]][f"{support_type}{key}"] = [value[0][1, :], value[1]]

                else:
                    if value.ndim == 1:
                        self.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value
                    else:
                        self.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value[0, :]
                        self.SIG.Support[ord_pair[1]][f"{support_type}{key}"] = value[1, :]

    def set_systematic_multipole_errors(self, ords: ndarray, BA, order: int, skewness: bool):
        """
        Applies multipole errors specified in `AB` in the lattice elements `ords` of `RING` depending on
        the specified options.
        It sets the systematic multipoles of the field component defined by option `'order'` and `'type'`.
        It is required that the `BA` entries are normalized by that component, e.g. `BA[1, 0]=1` for skew-quadrupole
        systematic multipoles.
        The systematic multipoles are from now on scaled with the current magnet excitation and added to the
        PolynomA/B fields.

        Args:
            ords: Ordinates of the considered magnets.
            BA: [N x 2] array of PolynomA/B multipole errors.
            order: Numeric value defining the order of the considered magnet: [0,1,2,...] => [dip,quad,sext,...]
            skewness: if True apply errors to skew fields (PolynomA) if False to normal fields (PolynomB)

        Examples:
            Defines systematic multipole components for the 'QF' magnet and
            adds it to the field offsets of all magnets named 'QF'::

                ords = SCgetOrds(SC.RING,'QF')
                BA = np.array([[1E-5, 0], [1E-4, 0], [0, 0], [1E-2, 0]])
                RING = SC.set_systematic_multipole_errors(RING, ords, BA, 1, False)

        See Also:
            *SC.update_magnets*, *SC.set_random_multipole_errors*
        """
        if BA.ndim != 2 or BA.shape[1] != 2:
            raise ValueError("BA has to  be numpy.array of shape N x 2.")
        ind, source = (1, "A") if skewness else (0, "B")
        newBA = BA[:, :]
        newBA[order, ind] = 0
        for ord in ords:
            for target in ("A", "B"):
                attr_name = f'SysPol{target}From{source}'
                syspol = getattr(self.RING[ord], attr_name) if hasattr(self.RING[ord], attr_name) else DotDict()
                syspol[order] = newBA[:, ind]
                setattr(self.RING[ord], attr_name, syspol)

    def set_random_multipole_errors(self, ords: ndarray, BA):
        """
        Applies random multipole errors specified in `BA` in the lattice elements `ords` of `RING`.
        It generates random multipole components with a 2-sigma truncated Gaussian distribution
        from each of the `BA` entries.
        The final multipole errors are stored in the PolynomA/BOffset of the lattice elements.

        Args:
            ords: Ordinates of the considered magnets.
            BA: [N x 2] array of PolynomB/A multipole errors. [normal, skew] components.

        Examples:
            Defines random multipole components for the 'QF' magnet
            and adds it to the field offsets of all magnets named 'QF'::

                ords = SCgetOrds(SC.RING,'QF')
                BA = np.array([[1E-5, 0], [1E-4, 0], [0, 0], [1E-2, 0]])
                SC.set_random_multipole_errors(ords, BA)

        See Also:
            *SC.update_magnets*, *SC.set_systematic_multipole_errors*
        """
        if BA.ndim != 2 or BA.shape[1] != 2:
            raise ValueError("BA has to  be numpy.array of shape N x 2.")
        for ord in ords:
            randBA = SCrandnc(2, BA.shape) * BA  # TODO this should be registered in SC.SIG
            for ind, target in enumerate(("B", "A")):
                attr_name = f"Polynom{target}Offset"
                setattr(self.RING[ord], attr_name,
                        add_padded(getattr(self.RING[ord], attr_name), randBA[:, ind])
                        if hasattr(self.RING[ord], attr_name) else randBA[:, ind])

    def apply_errors(self, nsigmas: float = 2):
        """
        Applies errors to cavities, injection trajectory, BPMs, circumference,
        support structures and magnets if the corresponding uncertainties defined in
        `SC.SIG` are set. For example, for a magnet with ordinate `ord` every field
        defined in `SC.SIG.Mag{ord}` will be used to generate a random number using a
        Gaussian distribution with a cutoff (see option below) and `sigma` being the
        value of the uncertainty field. The number will be stored in the
        corresponding field of the lattice structure, thus `SC.RING{ord}`. An
        exeption are bending angle errors which are stored in the `BendingAngleError`
        field. See examples in the SC.register* functions for more details.

        *SC.apply_errors* uses the uncertainties defined in `SC.SIG` to
        generate random errors and applies them to the corresponding attributes of elements in `SC.RING`.

        Args:
            nsigmas: Number of sigmas at which the Gaussian distribution of errors is truncated

        See Also:
            *SC.register_magnets*, *SC.register_support*, *SC.register_bpms*, *SC.register_cavities*
        """
        # RF
        for ind in intersect(self.ORD.RF, self.SIG.RF.keys()):
            for field in self.SIG.RF[ind]:
                setattr(self.RING[ind], field, randn_cutoff(self.SIG.RF[ind][field], nsigmas))
        # BPM
        for ind in intersect(self.ORD.BPM, self.SIG.BPM.keys()):
            for field in self.SIG.BPM[ind]:
                if re.search('Noise', field):
                    setattr(self.RING[ind], field, self.SIG.BPM[ind][field])
                else:
                    setattr(self.RING[ind], field, randn_cutoff(self.SIG.BPM[ind][field], nsigmas))
        # Magnet
        for ind in intersect(self.ORD.Magnet, self.SIG.Magnet.keys()):
            for field in self.SIG.Magnet[ind]:
                setattr(self.RING[ind], 'BendingAngleError' if field == 'BendingAngle' else field,
                        randn_cutoff(self.SIG.Magnet[ind][field], nsigmas))
        # Injection
        self.INJ.Z0 = self.INJ.Z0ideal + self.SIG.staticInjectionZ * SCrandnc(nsigmas, (6,))
        self.INJ.randomInjectionZ = 1 * self.SIG.randomInjectionZ
        # Circumference
        if 'Circumference' in self.SIG.keys():
            circScaling = 1 + self.SIG.Circumference * SCrandnc(nsigmas, (1, 1))
            self.RING = SCscaleCircumference(self.RING, circScaling, 'rel')
            LOGGER.info('Circumference error applied.')
        # Misalignments
        self._apply_support_alignment_error(nsigmas)

        self.update_supports()
        if len(self.ORD.Magnet):
            self.update_magnets()
        if len(self.ORD.RF) and len(self.SIG.RF):
            self.update_cavities()

    def _apply_support_alignment_error(self, nsigmas):
        s_pos = findspos(self.RING)
        for support_type in SUPPORT_TYPES:
            for ordPair in self.ORD[support_type].T:
                if ordPair[0] not in self.SIG.Support.keys():
                    continue
                for field, value in self.SIG.Support[ordPair[0]].items():
                    if support_type not in field:
                        continue
                    setattr(self.RING[ordPair[0]], field, randn_cutoff(value, nsigmas))
                    setattr(self.RING[ordPair[1]], field,
                            randn_cutoff(value, nsigmas) if field in self.SIG.Support[ordPair[1]].keys()
                            else getattr(self.RING[ordPair[0]], field))

                struct_length = np.remainder(np.diff(s_pos[ordPair]), s_pos[-1])
                rolls0 = copy.deepcopy(getattr(self.RING[ordPair[0]], f"{support_type}Roll"))  # Twisted supports are not considered
                offsets0 = copy.deepcopy(getattr(self.RING[ordPair[0]], f"{support_type}Offset"))
                offsets1 = copy.deepcopy(getattr(self.RING[ordPair[1]], f"{support_type}Offset"))

                if rolls0[1] != 0:
                    if f"{support_type}Offset" in self.SIG.Support[ordPair[1]].keys():
                        raise Exception(f'Pitch angle errors can not be given explicitly if {support_type} '
                                        f'start and endpoints each have offset uncertainties.')
                    offsets0[1] -= rolls0[1] * struct_length / 2
                    offsets1[1] += rolls0[1] * struct_length / 2

                else:
                    rolls0[1] = (offsets1[1] - offsets0[1]) / struct_length
                if rolls0[2] != 0:
                    if f"{support_type}Offset" in self.SIG.Support[ordPair[1]].keys():
                        raise Exception(f'Yaw angle errors can not be given explicitly if {support_type} '
                                        f'start and endpoints each have offset uncertainties.')
                    offsets0[0] -= rolls0[2] * struct_length / 2
                    offsets1[0] += rolls0[2] * struct_length / 2
                else:
                    rolls0[2] = (offsets1[0] - offsets0[0]) / struct_length
                setattr(self.RING[ordPair[0]], f"{support_type}Roll", rolls0)
                setattr(self.RING[ordPair[0]], f"{support_type}Offset", offsets0)
                setattr(self.RING[ordPair[1]], f"{support_type}Offset", offsets1)

    def update_cavities(self, ords: ndarray = None):
        """
        Updates the cavity fields `Voltage`, `Frequency` and `TimeLag` in `SC.RING` as specified in `ords`.
        If no ordinates are given explicitly, all registered cavities defined in `SC.ORD.RF` are
        updated. For each cavity and each field, the setpoints, calibration errors and offsets are considered.

        Args:
            ords: Cavity ordinates to be updated.

        See Also:
            *SC.register_cavities*, *SC.apply_errors*
        """
        for ind in (self.ORD.RF if ords is None else ords):
            for field in RF_PROPERTIES:
                setattr(self.RING[ind], field,
                        getattr(self.RING[ind], f"{field}SetPoint")
                        * (1 + getattr(self.RING[ind], f"{field}CalError"))
                        + getattr(self.RING[ind], f"{field}Offset"))

    def update_magnets(self, ords: ndarray = None):
        """
        Updates the magnets specified in `RING` as specified in `ords`. If no ordinates are given
        explicitly, all registered magnets defined in `SC.ORD.Magnet` are updated. For each magnet the
        setpoints (`SetPointA/B`) and calibration errors (`CalErrorA/B`) are evaluated.
        If systematic multipole components are specified, e.g. in `SysPolBFromB` for systematic
        PolynomB-multipoles induced by PolynomB entries, the corresponding multipole components are scaled
        by the current magnet excitation and added, as well as static field offsets (if specified in
        `PolynomA/BOffset`).
        If the considered magnet has a bending angle error (from pure bending angle error or due to a
        combined function magnet), the corresponding horizontal dipole magnetic field is calculated and
        added to the PolynomB(1) term. It is thereby assured that a dipole error doesn't alter the
        coordinate system.
        If the considered magnet is registered as a slpit magnet (`'MasterOf'`), the errors and setpoints
        of the master magnet are applied to the fields of the child magnets. Note that split quadrupole
        magnets with different gradients, however, or split CMs can currently not be updated correctly.

        Args:
            ords: Magnets ordinates to be updated.

        See Also:
            *SC.register_magnets*, *SC.apply_errors*, *SC.set_systematic_multipole_errors*,
            *SC.set_random_multipole_errors*, *set_magnet_setpoints*, *set_cm_setpoints*

        """
        for ind in (self.ORD.Magnet if ords is None else ords):
            self._update_magnets(ind, ind)
            if hasattr(self.RING[ind], 'MasterOf'):
                for child_ind in self.RING[ind].MasterOf:
                    self._update_magnets(ind, child_ind)

    def update_supports(self, offset_bpms: bool = True, offset_magnets: bool = True):
        """
        This function updates the offsets and rolls of the elements in `SC.RING`
        based on the current support errors, by setting the lattice fields `T1`, `T2`, and
        `R1`, `R2` for magnets and the fields `SupportOffset` and `SupportRoll` for BPMs.

        Args:
            offset_bpms: If true, BPM offsets are updated.
            offset_magnets: If true, magnet offsets are updated.

        See Also:
            *SC.register_support*, *SC.support_offset_and_roll*, *plot_support*

        """
        s_pos = findspos(self.RING)
        if offset_magnets:
            if len(self.ORD.Magnet):
                offsets, rolls = self.support_offset_and_roll(s_pos[self.ORD.Magnet])
                for i, ind in enumerate(self.ORD.Magnet):
                    setattr(self.RING[ind], "SupportOffset", offsets[:, i])
                    setattr(self.RING[ind], "SupportRoll", rolls[:, i])
                    self.RING[ind] = update_transformation(self.RING[ind])
                    if hasattr(self.RING[ind], 'MasterOf'):
                        for child_ind in self.RING[ind].MasterOf:
                            for field in ("T1", "T2", "R1", "R2"):
                                setattr(self.RING[child_ind], field, getattr(self.RING[ind], field))
            else:
                LOGGER.warning('SC: No magnets have been registered!')
        if offset_bpms:
            if len(self.ORD.BPM):
                offsets, rolls = self.support_offset_and_roll(s_pos[self.ORD.BPM])
                for i, ind in enumerate(self.ORD.BPM):
                    setattr(self.RING[ind], "SupportOffset", offsets[0:2, i])  # No longitudinal BPM offsets implemented
                    setattr(self.RING[ind], "SupportRoll",
                            np.array([rolls[0, i]]))  # BPM pitch and yaw angles not  implemented
            else:
                LOGGER.warning('SC: No BPMs have been registered!')

    def support_offset_and_roll(self, s_locations: ndarray) -> Tuple[ndarray, ndarray]:
        """
        This function evaluates the total offsets, roll, pitch and yaw angles of the support structures that have
        been defined via *SC.register_support* at the longitudinal positions `s` by linearly interpolating
        between support structure start- and endpoints (girder + sections + plinths, if registered).
        Note that this calculation may not provide the proper values if magnets with non-zero bending
        angle are within the support structure because it does not account for the rotation of the
        local coordinate system along the beam trajectory.

        Args:
            s_locations: Array of s-positions at which the offset is evaluated.

        Return type:

            [3,length(s)]-array containing the [dx/dy/dz] total support structure offsets at `s`.

            [3,length(s)]-array containing the [az/ax/ay] total support structure rolls at `s`.

        See Also:
            *SC.register_support*, *SC.update_support*, *plot_support*
        """
        s_pos = findspos(self.RING)
        ring_length = s_pos[-1]
        off0 = np.zeros((3, len(s_pos)))
        roll0 = np.zeros((3, len(s_pos)))

        for suport_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
            if suport_type in self.ORD:
                ord1 = self.ORD[suport_type][0, :]  # Beginning ordinates
                ord2 = self.ORD[suport_type][1, :]  # End ordinates
                tmpoff1 = np.zeros((3, len(ord1)))
                tmpoff2 = np.zeros((3, len(ord2)))
                for i in range(len(ord1)):
                    tmpoff1[:, i] = off0[:, ord1[i]] + getattr(self.RING[ord1[i]], f"{suport_type}Offset")
                    tmpoff2[:, i] = off0[:, ord2[i]] + getattr(self.RING[ord2[i]], f"{suport_type}Offset")
                for i in range(3):
                    off0[i, :] = s_interpolation(off0[i, :], s_pos, ord1, tmpoff1[i, :], ord2, tmpoff2[i, :])

        for support_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
            for ords in self.ORD[support_type].T:
                roll_start0 = getattr(self.RING[ords[0]], f"{support_type}Roll")[0]
                struct_length = s_pos[ords[1]] - s_pos[ords[0]]
                mask = np.zeros(len(s_pos), dtype=bool)
                mask[ords[0]:ords[1]] = True
                offset1 = off0[1, ords[1]] - off0[1, ords[0]]
                offset2 = off0[0, ords[1]] - off0[0, ords[0]]
                if ords[0] > ords[1]:
                    struct_length = ring_length + struct_length
                    mask[ords[0]] = False
                    mask = ~mask
                else:
                    mask[ords[1]] = True
                roll0[0, mask] += roll_start0
                roll0[1, mask] = offset1 / struct_length
                roll0[2, mask] = offset2 / struct_length

        if not np.array_equal(s_locations, s_pos):
            b = np.unique(s_pos, return_index=True)[1]
            off, roll = np.empty((3, len(s_locations))), np.empty((3, len(s_locations)))
            for i in range(3):
                off[i, :] = np.interp(s_locations, s_pos[b], off0[i, b])
                roll[i, :] = np.interp(s_locations, s_pos[b], roll0[i, b])
            return off, roll
        return off0, roll0

    def set_cavity_setpoints(self, ords: Union[int, List[int], ndarray],
                             setpoints: Union[float, List[float], ndarray],
                             param: str, method: str = SETTING_ABS):
        """
        Set RF properties to setpoints

        Set the setpoints of `Voltage`, `Frequency` or `TimeLag` as specified in "param" of the rf
        cavities specified in `ords`. If only a single setpoint is given for multiple cavities,
        the setpoint is applied to all cavities.

        Args:
            ords: Array of cavity ordinates in the lattice structure (SC.ORD.RF)
            setpoints: Setpoints (array or single value for all cavities)
            param: String ('Voltage', 'Frequency' or 'TimeLag') specifying which cavity field should be set.
            method: 'abs' (default), Use absolute setpoint
                    'rel', Use relative setpoint to nominal value
                    'add', Add setpoints to current value

        Examples:
            Sets the time lag of all cavities registered in SC to zero::

                SC.set_cavity_setpoints(ords=SC.ORD.RF, setpoints=0.0, param='TimeLag')

            Adds 1kHz to the frequency of the first cavity::

                SC.set_cavity_setpoints(ords=SC.ORD.RF[0], setpoints=1E3, param='Frequency', method='add')

        """
        ords_1d, setpoints_1d = check_input_and_setpoints(method, ords, setpoints)
        setpoint_str = f"{param}{SETPOINT}"
        if method == SETTING_REL:
            setpoints_1d *= atgetfieldvalues(self.RING, ords_1d, setpoint_str)
        if method == SETTING_ADD:
            setpoints_1d += atgetfieldvalues(self.RING, ords_1d, setpoint_str)
        for i, ord in enumerate(ords_1d):
            setattr(self.RING[ord], setpoint_str, setpoints_1d[i])
        self.update_cavities(ords_1d)

    def get_cm_setpoints(self, ords: Union[int, List[int], ndarray], skewness: bool) -> ndarray:
        """

        Return current dipole Corrector Magnets (CM) setpoints

        Reads the setpoints of the CMs specified in `ords` in the dimension `skewness`.

        Args:
            ords: Array of CM ordinates in the lattice structure (ex: SC.ORD.CM[0])
            skewness: boolean specifying CM dimension ([False|True] -> [hor|ver])

        Returns:
            CM setpoints [rad]
        """
        ords_1d = np.ravel(np.array([ords], dtype=int))
        order = 0
        ndim = int(skewness)
        letter = NUM_TO_AB[ndim]
        setpoints = atgetfieldvalues(self.RING, ords_1d, f"{SETPOINT}{letter}", order)
        for i, ord1d in enumerate(ords_1d):
            if self.RING[ord1d].PassMethod != 'CorrectorPass':
                # positive setpoint -> positive kick -> negative horizontal field
                setpoints[i] *= (-1) ** (ndim + 1) * self.RING[ord1d].Length
        return setpoints

    def set_cm_setpoints(self, ords: Union[int, List[int], ndarray],
                         setpoints: Union[float, List[float], ndarray],
                         skewness: bool, method: str = SETTING_ABS):
        """
        Sets dipole corrector magnets to different setpoints

        Sets horizontal or vertical CMs as specified in `ords` and `skewness`, respectively, to `setpoints`
        [rad] and updates the magnetic fields. If the corresponding setpoint exceeds the CM limit
        specified in the corresponding lattice field `CMlimit`, the CM is clipped to that value
        and a warning is being printed (to switch off, use `warning('off','SC:CM1'))`. Positive setpoints
        will result in kicks in the positive horizontal or vertical direction.

        Args:
            ords:  Array of CM ordinates in the lattice structure (ex: SC.ORD.CM[0])
            setpoints:  CM setpoints (array or single value for all CMs) [rad]
            skewness: boolean specifying CM dimension ([False|True] -> [hor|ver])
            method: 'abs' (default), Use absolute setpoint
                    'rel', Use relative setpoint to current value
                    'add', Add setpoints to current value

        Examples:
            Set all registered horizontal CMs to zero::

                SC.set_cm_setpoints(ords=SC.ORD.HCM, skewness=False, setpoints=0.0)

            Add 10urad to the fourth registered vertical CM::

                SC.set_cm_setpoints(ords=SC.ORD.VCM[4], setpoints=1E-5, skewness=True, method='add')

        """
        # TODO corrector does not have PolynomA/B in at?
        ords_1d, setpoints_1d = check_input_and_setpoints(method, ords, setpoints)
        order = 0
        ndim = int(skewness)
        letter = NUM_TO_AB[ndim]
        for i, ord in enumerate(ords_1d):
            # positive setpoint -> positive kick -> negative horizontal field
            norm_by = (-1) ** (ndim + 1) * self.RING[ord].Length if self.RING[ord].PassMethod != 'CorrectorPass' else 1
            if method == SETTING_REL:
                setpoints_1d[i] *= getattr(self.RING[ord], f"{SETPOINT}{letter}")[order] * norm_by
            if method == SETTING_ADD:
                setpoints_1d[i] += getattr(self.RING[ord], f"{SETPOINT}{letter}")[order] * norm_by
            if hasattr(self.RING[ord], 'CMlimit') and abs(setpoints_1d[i]) > abs(self.RING[ord].CMlimit[ndim]):
                LOGGER.info(f'CM (ord: {ord} / dim: {ndim}) is clipping')
                setpoints_1d[i] = np.sign(setpoints_1d[i]) * self.RING[ord].CMlimit[ndim]
            getattr(self.RING[ord], f"{SETPOINT}{letter}")[order] = setpoints_1d[i] / norm_by

        self.update_magnets(ords_1d)

    def set_magnet_setpoints(self, ords: Union[int, List[int], ndarray],
                             setpoints: Union[float, List[float], ndarray],
                             skewness: bool, order: int, method: str = SETTING_ABS,
                             dipole_compensation: bool = False):
        """
        Sets magnets to setpoints

        Sets magnets (except CMs) as specified in `ords` to `setpoints` while `order` and `skewness` defines
        which field entry should be used (see below). The setpoints may be given relative to their nominal
        value or in absolute terms. If the considered quadrupole is a combined function magnet with
        non-zero bending angle and the kick compensation flag 'dipole_compensation'=True, the appropriate bending
        angle difference is calculated and the horizontal CM setpoint is changed accordingly to compensate
        for that dipole kick difference.
        If the setpoint of a skew quadrupole exceeds the limit specified in the corresponding lattice
        field `SkewQuadLimit`, the setpoint is clipped to that value and a warning is logged.

        Args:
            ords:  Array of magnets ordinates in the lattice structure (ex: SC.ORD.HCM) (numpy.array() or list of int [int,int,..])
            setpoints:  magnets setpoints (array or single value for all magnets).
                        setpoints are assigned to the given order and skewness, i.e. once updated through
                        SimulatedCommissioning.apply_errors, they correspond to a single element of PolynomA or PolynomB
            skewness: boolean specifying magnet plane ([False|True] -> [PolynomB|PolynomA])
            method: 'abs' (default), Use absolute setpoint
                    'rel', Use relative setpoint to nominal value
                    'add', Add setpoints to current value
            order: Numeric value defining the order of the considered magnet: [0,1,2,...] => [dip,quad,sext,...]
            dipole_compensation: (default = False) Used for combined function magnets. If this flag is set and if there is a horizontal CM
                                registered in the considered magnet, the CM is used to compensate the bending angle difference
                                if the applied quadrupole setpoints differs from the design value.

        Examples:
            Identify the ordinates of all elements named `'SF'` and switch their sextupole component off::

                ords = SCgetOrds(SC.RING,'SF')
                SC.register_magnets(ords)
                SC.set_magnet_setpoints(ords=ords, skewness=False, order=2, setpoints=0.0, method='abs')

            Identify the ordinates of all elements named `QF` and `QD` and set their quadrupole component to 99% of their design value::

                ords = SCgetOrds(SC.RING,'QF|QD')
                SC.register_magnets(ords)
                SC.set_magnet_setpoints(ords=ords, skewness=False, order=1, setpoints=0.99, method='rel')

        """
        ords_1d, setpoints_1d = check_input_and_setpoints(method, ords, setpoints)
        letter = NUM_TO_AB[int(skewness)]
        if method == SETTING_REL:
            setpoints_1d *= atgetfieldvalues(self.RING, ords_1d, f"NomPolynom{letter}", order)
        if method == SETTING_ADD:
            setpoints_1d += atgetfieldvalues(self.RING, ords_1d, f"{SETPOINT}{letter}", order)
        for i, ord in enumerate(ords_1d):
            if skewness and order == 1 and getattr(self.RING[ord], 'SkewQuadLimit', np.inf) < np.abs(setpoints_1d[i]):
                LOGGER.info(f'SkewLim \n Skew quadrupole (ord: {ord}) is clipping')
                setpoints_1d[i] = np.sign(setpoints_1d[i]) * self.RING[ord].SkewQuadLimit
            # TODO should check CF magnets
            if dipole_compensation and order == 1:  # quad  # TODO check also skewness?
                self._dipole_compensation(ord, setpoints_1d[i])
            getattr(self.RING[ord], f"{SETPOINT}{letter}")[order] = setpoints_1d[i]

        self.update_magnets(ords_1d)

    def _dipole_compensation(self, ord, setpoint):
        if getattr(self.RING[ord], 'BendingAngle', 0) != 0 and ord in self.ORD.HCM:
            self.set_cm_setpoints(
                ord,
                (setpoint - self.RING[ord].SetPointB[1]) / self.RING[ord].NomPolynomB[1] * self.RING[ord].BendingAngle,
                skewness=False, method=SETTING_ADD)

    def very_deep_copy(self):
        copied_structure = copy.deepcopy(self)
        copied_structure.RING = self.RING.deepcopy()
        copied_structure.IDEALRING = self.IDEALRING.deepcopy()
        for ind, element in enumerate(self.RING):
            copied_structure.RING[ind] = element.deepcopy()
            copied_structure.IDEALRING[ind] = element.deepcopy()
        return copied_structure

    def verify_structure(self):
        """
        Verifies the integrity of SimilatedCommissionining class instance and warns if things look fishy.
        If you find something that is missing please contact us.

        See Also:
            *SC.register_magnets*, *SC.register_bpms*, *SC.register_cavities*
        """
        # BPMs
        if (n_bpms := len(self.ORD.BPM)) == 0:
            LOGGER.warning('No BPMs registered. Use "register_bpms".')
        else:
            LOGGER.info(f'{n_bpms:d} BPMs registered.')
            if len(np.unique(self.ORD.BPM)) != n_bpms:
                LOGGER.warning('BPMs not uniquely defined.')
        # Supports
        if len(self.ORD.Girder[0]) == 0 and (len(self.ORD.Plinth[0]) or len(self.ORD.Section[0])):
            raise ValueError('Girders must be registered for other support structure misalingments to work.')
        # Corrector magnets
        if (n_hcms := len(self.ORD.HCM)) == 0:
            LOGGER.warning('No horizontal CMs registered. Use "register_magnets".')
        else:
            LOGGER.info(f'{n_hcms:d} HCMs registered.')
            if len(np.unique(self.ORD.HCM)) != n_hcms:
                LOGGER.warning('Horizontal CMs not uniquely defined.')
        if (n_vcms := len(self.ORD.VCM)) == 0:
            LOGGER.warning('No vertical CMs registered. Use "register_magnets".')
        else:
            LOGGER.info(f'{n_vcms:d} VCMs registered.')
            if len(np.unique(self.ORD.VCM)) != n_vcms:
                LOGGER.warning('Vertical CMs not uniquely defined.')
        for ord in self.ORD.HCM:
            if self.RING[ord].CMlimit[0] == 0:
                LOGGER.warning(f'HCM limit is zero (Magnet ord: {ord:d}). Sure about that?')
        for ord in self.ORD.VCM:
            if self.RING[ord].CMlimit[1] == 0:
                LOGGER.warning(f'VCM limit is zero (Magnet ord: {ord:d}). Sure about that?')

        # magnets
        for ord in self.ORD.Magnet:
            if len(self.RING[ord].PolynomB) != len(self.RING[ord].PolynomA):
                raise ValueError(f'Length of PolynomB and PolynomA are not equal (Magnet ord: {ord:d})')
            elif len(self.RING[ord].SetPointB) != len(self.RING[ord].CalErrorB): # TODO: make consistent with _updateMagnets (calError adn Setpoints could get mult_padded)
                raise ValueError(f'Length of SetPointB and CalErrorB are not equal (Magnet ord: {ord:d})')
            elif len(self.RING[ord].SetPointA) != len(self.RING[ord].CalErrorA):
                raise ValueError(f'Length of SetPointA and CalErrorA are not equal (Magnet ord: {ord:d})')
            if hasattr(self.RING[ord],'PolynomBOffset') and len(self.RING[ord].PolynomBOffset) != len(self.RING[ord].PolynomAOffset):
                    raise ValueError(f'Length of PolynomBOffset and PolynomAOffset are not equal (Magnet ord: {ord:d})')
            if hasattr(self.RING[ord],'CombinedFunction') and self.RING[ord].CombinedFunction:
                if not hasattr(self.RING[ord],'BendingAngle'):
                    raise ValueError(f'Combined function magnet (ord: {ord:d}) requires field "BendingAngle".')
                if self.RING[ord].NomPolynomB[1] == 0:
                    LOGGER.warning(f'Combined function magnet (ord: {ord:d}) has zero design quadrupole component.')
                if self.RING[ord].BendingAngle == 0:
                    LOGGER.warning(f'Combined function magnet (ord: {ord:d}) has zero bending angle.')
            if len(self.SIG.Magnet[ord]) == 0:
                LOGGER.info(f'Magnet (ord: {ord:d}) is registered without uncertainties.')
            else:
                for field in self.SIG.Magnet[ord]:
                    if not hasattr(self.RING[ord],field):
                        LOGGER.warning(f'Field "{field:s}" in SC.SIG.Magnet doesnt match lattice element (Magnet ord: {ord:d})')
            if hasattr(self.RING[ord],'MasterOf'):
                for cOrd in self.RING[ord].MasterOf:
                    for field in self.RING[cOrd]:
                        if not hasattr(self.RING[ord],field): 
                            LOGGER.warning(f'Child magnet (ord: {ord:d}) has different field "{field:s}" than master magnet (ord: %{cOrd:d}).')
       
        if len(self.ORD.RF) == 0:
            LOGGER.warning('No cavity registered. Use "SCregisterCavity".')
        else:
            LOGGER.info(f'{len(self.ORD.RF):d} cavity/cavities registered.')
        if len(np.unique(self.ORD.RF)) != len(self.ORD.RF):
            LOGGER.warning('Cavities not uniquely defined.')
        if 'RF' in self.SIG:
            for ord in self.ORD.RF:
                for field in self.SIG.RF[ord]:
                    if not hasattr(self.RING[ord],field):
                        LOGGER.warning(f'Field "{field:s}" in SC.SIG.RF doesnt match lattice element (Cavity ord: {ord:d})')
        if self.INJ.beamSize.shape != (6, 6):
            raise ValueError('"SC.INJ.beamSize" must be a 6x6 array.')
        if self.SIG.randomInjectionZ.shape != (6,):
            raise ValueError('"SC.SIG.randomInjectionZ" must be a len(6) array.')
        if self.SIG.staticInjectionZ.shape != (6,):
            raise ValueError('"SC.SIG.staticInjectionZ" must be a len(6) array.')
        apEl = []
        for ord in range(len(self.RING)):
            if hasattr(self.RING[ord],'EApertures') and hasattr(self.RING[ord],'RApertures'):
                LOGGER.warning(f'Lattice element #{ord:d} has both EAperture and RAperture')
            if hasattr(self.RING[ord],'EApertures') or hasattr(self.RING[ord],'RApertures'):
                apEl.append(ord)
        if len(apEl) == 0:
            LOGGER.warning('No apertures found.')
        else:
            LOGGER.info(f'Apertures defined in {len(apEl):d} out of {len(self.RING):d} elements.')

    @staticmethod
    def _check_kwargs(kwargs, allowed_options):
        if len(unknown_keys := [key for key in kwargs.keys() if key not in allowed_options]):
            raise ValueError(f"Unknown keywords {unknown_keys}. Allowed keywords are {allowed_options}")

    def _optional_magnet_fields(self, ind, MAGords, **kwargs):
        if 'CF' in kwargs.keys():
            self.RING[ind].CombinedFunction = True
        if intersect(("HCM", "VCM"), kwargs.keys()) and not hasattr(self.RING[ind], 'CMlimit'):
            self.RING[ind].CMlimit = np.zeros(2)
        if 'HCM' in kwargs.keys():
            self.RING[ind].CMlimit[0] = kwargs["HCM"]
        if 'VCM' in kwargs.keys():
            self.RING[ind].CMlimit[1] = kwargs['VCM']
        if 'SkewQuad' in kwargs.keys():
            self.RING[ind].SkewQuadLimit = kwargs['SkewQuad']
        if 'MasterOf' in kwargs.keys():
            if np.count_nonzero(MAGords == ind) > 1:
                raise ValueError(f"Non-unique element index {ind} found together with ``MasterOf``")
            self.RING[ind].MasterOf = kwargs['MasterOf'][:, np.nonzero(MAGords == ind)].ravel()

    def _update_magnets(self, source_ord, target_ord):
        setpoints_a, setpoints_b = self.RING[source_ord].SetPointA, self.RING[source_ord].SetPointB
        polynoms = dict(A=setpoints_a * add_padded(np.ones(len(setpoints_a)), self.RING[source_ord].CalErrorA),
                        B=setpoints_b * add_padded(np.ones(len(setpoints_b)), self.RING[source_ord].CalErrorB))
        for target in AB:
            new_polynom = polynoms[target][:]
            if hasattr(self.RING[target_ord], f'Polynom{target}Offset'):
                new_polynom = add_padded(new_polynom, getattr(self.RING[target_ord], f'Polynom{target}Offset'))
            for source in AB:
                if hasattr(self.RING[target_ord], f'SysPol{target}From{source}'):
                    polynom_errors = getattr(self.RING[target_ord], f'SysPol{target}From{source}')
                    for n in polynom_errors.keys():
                        new_polynom = add_padded(new_polynom, polynoms[source][n] * polynom_errors[n])
            setattr(self.RING[target_ord], f"Polynom{target}", new_polynom)

        if hasattr(self.RING[source_ord], 'BendingAngleError'):
            self.RING[target_ord].PolynomB[0] += (self.RING[source_ord].BendingAngleError
                                                  * self.RING[target_ord].BendingAngle / self.RING[target_ord].Length)
        if hasattr(self.RING[source_ord], 'BendingAngle'):
            if hasattr(self.RING[source_ord], 'CombinedFunction') and self.RING[source_ord].CombinedFunction:
                alpha_act = (self.RING[source_ord].SetPointB[1] * (1 + self.RING[source_ord].CalErrorB[1])
                             / self.RING[source_ord].NomPolynomB[1])
                effBendingAngle = alpha_act * self.RING[target_ord].BendingAngle
                self.RING[target_ord].PolynomB[0] += ((effBendingAngle - self.RING[target_ord].BendingAngle)
                                                      / self.RING[target_ord].Length)
        if self.RING[source_ord].PassMethod == 'CorrectorPass':
            self.RING[target_ord].KickAngle = np.array([self.RING[target_ord].PolynomB[0],
                                                        self.RING[target_ord].PolynomA[0]])
        self.RING[target_ord].MaxOrder = len(self.RING[target_ord].PolynomB) - 1
