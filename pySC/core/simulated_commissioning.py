"""
Simulated Commissioning
-------------

This module contains the main data structure of ``pySC`` package
built up around the ``at.Lattice`` under study.
"""
import copy
import re
from typing import Tuple

import numpy as np
from at import Lattice
from numpy import ndarray

from pySC.core.classes import Injection, Sigmas, Indices, DotDict
from pySC.core.constants import (BPM_ERROR_FIELDS, RF_ERROR_FIELDS, RF_PROPERTIES, MAGNET_TYPE_FIELDS,
                                 MAGNET_ERROR_FIELDS, AB, SUPPORT_TYPES, SUPPORT_ERROR_FIELDS)
from pySC.utils import logging_tools
from pySC.utils.at_wrapper import findspos
from pySC.utils.classdef_tools import update_double_ordinates, add_padded, intersect, randn_cutoff, s_interpolation
from pySC.utils.sc_tools import SCrandnc, SCscaleCircumference, update_transformation

LOGGER = logging_tools.get_logger(__name__)


class SimulatedCommissioning:
    """
    The main structure of ``pySC``, which holds all the information about
    lattice error sources and errors, injection settings and its errors.
    The class is initialized from ``at.Lattice``.
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

    def register_magnets(self, ords: ndarray, **kwargs):
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
        for ind in (self.ORD.RF if ords is None else ords):
            for field in RF_PROPERTIES:
                setattr(self.RING[ind], field,
                        getattr(self.RING[ind], f"{field}SetPoint")
                        * (1 + getattr(self.RING[ind], f"{field}CalError"))
                        + getattr(self.RING[ind], f"{field}Offset"))

    def update_magnets(self, ords: ndarray = None):
        for ind in (self.ORD.Magnet if ords is None else ords):
            self._update_magnets(ind, ind)
            if hasattr(self.RING[ind], 'MasterOf'):
                for child_ind in self.RING[ind].MasterOf:
                    self._update_magnets(ind, child_ind)

    def update_supports(self, offset_bpms: bool = True, offset_magnets: bool = True):
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

    def verify_structure(self):
        # BPMs
        n_bpms = len(self.ORD.BPM)
        if n_bpms == 0:
            LOGGER.warning('No BPMs registered. Use "register_bpms".')
        else:
            LOGGER.info(f'{n_bpms:d} BPMs registered.')
            if len(np.unique(self.ORD.BPM)) != n_bpms:
                LOGGER.warning('BPMs not uniquely defined.')
        # Supports
        if len(self.ORD.Girder[0]) == 0 and (len(self.ORD.Plinth[0]) or len(self.ORD.Section[0])):
            raise ValueError('Girders must be registered for other support structure misalingments to work.')
        # Corrector magnets
        n_hcms = len(self.ORD.HCM)
        n_vcms = len(self.ORD.VCM)
        if n_hcms == 0:
            LOGGER.warning('No horizontal CMs registered. Use "register_magnets".')
        else:
            LOGGER.info(f'{n_hcms:d} HCMs registered.')
            if len(np.unique(self.ORD.HCM)) != n_hcms:
                LOGGER.warning('Horizontal CMs not uniquely defined.')
        if n_vcms == 0:
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
            elif len(self.RING[ord].SetPointB) != len(self.RING[ord].CalErrorB):
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
            if len(self.SIG.Magnet[ord]) != 0:
                for field in self.SIG.Magnet[ord]:
                    if field not in dir(self.RING[ord]):
                        LOGGER.warning(f'Field "{field:s}" in SC.SIG.Mag doesnt match lattice element (Magnet ord: {ord:d})')
                    if field == 'MagnetOffset':
                        if isinstance(self.SIG.Magnet[ord][field], list):
                            off = self.SIG.Magnet[ord][field][0]
                        else:
                            off = self.SIG.Magnet[ord][field]
                        if len(off) != 3:
                            raise ValueError(f'SC.SIG.Magnet[{ord:d}].MagnetOffset must be a [1x3] (dx,dy,dz) array.')
            if hasattr(self.RING[ord],'MasterOf'):
                masterFields = dir(self.RING[ord])
                for cOrd in self.RING[ord].MasterOf:
                    for field in dir(self.RING[cOrd]):
                        if field not in masterFields:
                            LOGGER.warning(f'Child magnet (ord: {ord:d}) has different field "{field:s}" than master magnet (ord: %{cOrd:d}).')
       
        if len(self.ORD.RF) == 0:
            LOGGER.warning('No cavity registered. Use "SCregisterBPMs".')
        else:
            LOGGER.info(f'{len(self.ORD.RF):d} cavity/cavities registered.')
        if len(np.unique(self.ORD.RF)) != len(self.ORD.RF):
            LOGGER.warning('Cavities not uniquely defined.')
        if 'RF' in self.SIG:
            for ord in self.ORD.RF:
                for field in self.SIG.RF[ord]:
                    if field not in dir(self.RING[ord]):
                        LOGGER.warning(f'Field "{field:s}" in SC.SIG.RF doesnt match lattice element (Cavity ord: {ord:d})')
        if self.INJ.beamSize.shape != (6, 6):
            raise ValueError('"SC.INJ.beamSize" must be a 6x6 array.')
        if self.SIG.randomInjectionZ.shape != (6,):
            raise ValueError('"SC.SIG.randomInjectionZ" must be a 6x1 array.')
        if self.SIG.staticInjectionZ.shape != (6,):
            raise ValueError('"SC.SIG.staticInjectionZ" must be a 6x1 array.')
        apEl = []
        for ord in range(len(self.RING)):
            if 'EApertures' in dir(self.RING[ord]) and 'RApertures' in dir(self.RING[ord]):
                LOGGER.warning(f'Lattice element #{ord:d} has both EAperture and RAperture')
            if 'EApertures' in dir(self.RING[ord]) or 'RApertures' in dir(self.RING[ord]):
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
