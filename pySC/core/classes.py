import copy
import re
from typing import Tuple

import numpy as np
from at import Lattice
from numpy import ndarray
from pySC.core.constants import (RF_PROPERTIES, SUPPORT_TYPES, AB, BPM_ERROR_FIELDS, MAGNET_ERROR_FIELDS, MAGNET_TYPE_FIELDS,
                            RF_ERROR_FIELDS, SUPPORT_ERROR_FIELDS, TRACKING_MODES)
from pySC.utils.classdef_tools import add_padded, randn_cutoff, update_double_ordinates, intersect, s_interpolation
from pySC.utils.sc_tools import SCrandnc, SCscaleCircumference, SCgetTransformation
from pySC.utils.at_wrapper import findspos


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key in self:
            if isinstance(self[key], dict):
                self[key] = DotDict(self[key])

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return super(DotDict, self).__getitem__(key)
        except KeyError as e:
            raise AttributeError(e).with_traceback(e.__traceback__) from e

    def deepcopy(self) -> "DotDict":
        """Returns a deep copy"""
        return copy.deepcopy(self)


class Injection:
    def __init__(self):
        self.beamLostAt: float = 1.0
        self.Z0ideal: ndarray = np.zeros(6)
        self.Z0: ndarray = np.zeros(6)
        self.beamSize: ndarray = np.zeros((6, 6))
        self.randomInjectionZ: ndarray = np.zeros(6)
        self.staticInjectionZ: ndarray = np.zeros(6)
        self.nParticles: int = 1
        self.nTurns: int = 1
        self.nShots: int = 1
        self._trackMode: str = 'TBT'
        self.postFun = self._dummy_func

    @staticmethod
    def _dummy_func(matrix: ndarray) -> ndarray:
        return matrix

    @property
    def trackMode(self):
        return self._trackMode

    @trackMode.setter
    def trackMode(self, mode):
        if mode not in TRACKING_MODES:
            raise AttributeError(f"trackMode property has to be one of {TRACKING_MODES}")
        self._trackMode = mode
        if mode == 'ORB':
            self.nTurns = 1
            self.nParticles = 1


class Indices(DotDict):

    def __init__(self):
        super(Indices, self).__init__()
        self.BPM: ndarray = np.array([], dtype=int)
        self.RF: ndarray = np.array([], dtype=int)
        self.Magnet: ndarray = np.array([], dtype=int)
        self.SkewQuad: ndarray = np.array([], dtype=int)
        self.HCM: ndarray = np.array([], dtype=int)
        self.VCM: ndarray = np.array([], dtype=int)
        self.Girder: ndarray = np.zeros((2, 0), dtype=int)
        self.Plinth: ndarray = np.zeros((2, 0), dtype=int)
        self.Section: ndarray = np.zeros((2, 0), dtype=int)

    @property
    def CM(self):
        return [self.HCM, self.VCM]


class Sigmas(DotDict):

    def __init__(self):
        super(Sigmas, self).__init__()
        self.BPM: DotDict = DotDict()
        self.Magnet: DotDict = DotDict()
        self.RF: DotDict = DotDict()
        self.Support: DotDict = DotDict()
        self.randomInjectionZ: ndarray = np.zeros(6)
        self.staticInjectionZ: ndarray = np.zeros(6)
        self.Circumference: float = 0.0  # Circumference error reletive / or absolute


class SimulatedComissioning(DotDict):
    def __init__(self, ring: Lattice):
        super(SimulatedComissioning, self).__init__()
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
        if len(unknown_keys := [key for key in kwargs.keys() if key not in BPM_ERROR_FIELDS]):
            raise ValueError(f"Unknown keywords {unknown_keys}. Allowed keywords are {BPM_ERROR_FIELDS}")
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
        if len(unknown_keys := [key for key in kwargs.keys() if key not in RF_ERROR_FIELDS]):
            raise ValueError(f"Unknown keywords {unknown_keys}. Allowed keywords are {RF_ERROR_FIELDS}")
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
        if len(unknown_keys := [key for key in kwargs.keys() if key not in MAGNET_TYPE_FIELDS + MAGNET_ERROR_FIELDS]):
            raise ValueError(f"Unknown keywords {unknown_keys}. Allowed keywords are {MAGNET_TYPE_FIELDS + MAGNET_ERROR_FIELDS}")
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
        if len(unknown_keys := [key for key in kwargs.keys() if key not in SUPPORT_ERROR_FIELDS]):
            raise ValueError(f"Unknown keywords {unknown_keys}. Allowed keywords are {SUPPORT_ERROR_FIELDS}")
        if not len(support_ords) or support_ords.shape[0] != 2:
            raise ValueError('Ordinates must be a 2xn array of ordinates.')
        if upstream := np.sum(np.diff(support_ords, axis=0) < 0):
            print(f"{upstream} {support_type} endpoints(s) may be upstream of startpoint(s).")
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
            print('Circumference error applied.')
        # Misalignments
        self._apply_support_alignment_error(nsigmas)

        self.update_supports()
        if len(self.ORD.Magnet):
            self.update_magnets()
        if len(self.ORD.RF) and len(self.SIG.RF):
            self.update_cavities()

    def _apply_support_alignment_error(self, nsigmas):
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

                struct_length = np.diff(findspos(self.RING, ordPair)) if np.diff(ordPair) > 0 else -np.diff(
                    findspos(self.RING, ordPair[::-1]))
                if ordPair[0] > ordPair[1]:
                    struct_length = findspos(self.RING, len(self.RING))[0] - struct_length

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
        if offset_magnets:
            if len(self.ORD.Magnet):
                s = findspos(self.RING, self.ORD.Magnet)
                offsets, rolls = self.support_offset_and_roll(s)
                for i, ind in enumerate(self.ORD.Magnet):
                    setattr(self.RING[ind], "SupportOffset", offsets[:, i])
                    setattr(self.RING[ind], "SupportRoll", rolls[:, i])
                    magLength = self.RING[ind].Length
                    magTheta = self.RING[ind].BendingAngle if hasattr(self.RING[ind], 'BendingAngle') else 0
                    magnet_offsets = self.RING[ind].SupportOffset + self.RING[ind].MagnetOffset
                    magnet_rolls = np.roll(self.RING[ind].MagnetRoll + self.RING[ind].SupportRoll, -1)  # z,x,y -> x,y,z
                    self.RING[ind].T1, self.RING[ind].T2, self.RING[ind].R1, self.RING[ind].R2 = SCgetTransformation(
                        magnet_offsets, magnet_rolls, magTheta, magLength)
                    if hasattr(self.RING[ind], 'MasterOf'):
                        for child_ind in self.RING[ind].MasterOf:
                            for field in ("T1", "T2", "R1", "R2"):
                                setattr(self.RING[child_ind], field, getattr(self.RING[ind], field))
            else:
                print('SC: No magnets have been registered!')
        if offset_bpms:
            if len(self.ORD.BPM):
                s = findspos(self.RING, self.ORD.BPM)
                offsets, rolls = self.support_offset_and_roll(s)
                for i, ind in enumerate(self.ORD.BPM):
                    setattr(self.RING[ind], "SupportOffset", offsets[0:2, i])  # No longitudinal BPM offsets implemented
                    setattr(self.RING[ind], "SupportRoll",
                            np.array([rolls[0, i]]))  # BPM pitch and yaw angles not  implemented
            else:
                print('SC: No BPMs have been registered!')

    def support_offset_and_roll(self, s_locations: ndarray) -> Tuple[ndarray, ndarray]:
        lengths = np.array([self.RING[i].Length for i in range(len(self.RING))])
        ring_length = np.sum(lengths)
        s0 = np.cumsum(lengths)
        sposMID = s0 - lengths / 2
        off0 = np.zeros((3, len(s0)))
        roll0 = np.zeros((3, len(s0)))

        for suport_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
            if suport_type in self.ORD:
                ord1 = self.ORD[suport_type][0, :]  # Beginning ordinates
                ord2 = self.ORD[suport_type][1, :]  # End ordinates
                s1 = sposMID[ord1]
                s2 = sposMID[ord2]
                tmpoff1 = np.zeros((3, len(ord1)))
                tmpoff2 = np.zeros((3, len(ord2)))
                for i in range(len(ord1)):
                    tmpoff1[:, i] = off0[:, ord1[i]] + getattr(self.RING[ord1[i]], f"{suport_type}Offset")
                    tmpoff2[:, i] = off0[:, ord2[i]] + getattr(self.RING[ord2[i]], f"{suport_type}Offset")
                for i in range(3):
                    off0[i, :] = s_interpolation(off0[i, :], s0, ring_length, s1, ord1, tmpoff1[i, :], s2, ord2,
                                                tmpoff2[i, :])

        for support_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
            for ords in self.ORD[support_type].T:
                roll_start0 = getattr(self.RING[ords[0]], f"{support_type}Roll")[0]
                struct_length = s0[ords[1]] - s0[ords[0]]
                mask = np.zeros(len(s0), dtype=bool)
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

        if not np.array_equal(s_locations, s0):
            b = np.unique(s0, return_index=True)[1]
            off, roll = np.empty((3, len(s_locations))), np.empty((3, len(s_locations)))
            for i in range(3):
                off[i, :] = np.interp(s_locations, s0[b], off0[i, b])
                roll[i, :] = np.interp(s_locations, s0[b], roll0[i, b])
            return off, roll
        return off0, roll0

    def verify_structure(self):
        raise NotImplementedError
    # if 'ORD' not in SC:
    #     raise ValueError('Nothing is registered.')
    # else:
    #     if 'BPM' not in SC['ORD']:
    #         print('No BPMs registered. Use ''SCregisterBPMs''.')
    #     else:
    #         if len(SC['ORD']['BPM']) == 0:
    #             print('No BPMs registered. Use ''SCregisterBPMs''.')
    #         else:
    #             print('%d BPMs registered.' % len(SC['ORD']['BPM']))
    #         if len(np.unique(SC['ORD']['BPM'])) != len(SC['ORD']['BPM']):
    #             print('BPMs not uniquely defined.')
    #     if 'Girder' not in SC['ORD'] and ('Plinth' in SC['ORD'] or 'Section' in SC['ORD']):
    #         print('Girders must be registered for other support structure misalingments to work.')
    #     if 'CM' not in SC['ORD']:
    #         print('No CMs registered. Use ''SCregisterCMs''.')
    #     else:
    #         if len(SC['ORD']['CM'][0]) == 0:
    #             print('No horizontal CMs registered. Use ''SCregisterCMs''.')
    #         else:
    #             print('%d HCMs registered.' % len(SC['ORD']['CM'][0]))
    #         if len(SC['ORD']['CM']) != 2 or len(SC['ORD']['CM'][1]) == 0:
    #             print('No vertical CMs registered. Use ''SCregisterCMs''.')
    #         else:
    #             print('%d VCMs registered.' % len(SC['ORD']['CM'][1]))
    #         if len(np.unique(SC['ORD']['CM'][0])) != len(SC['ORD']['CM'][0]):
    #             print('Horizontal CMs not uniquely defined.')
    #         if len(np.unique(SC['ORD']['CM'][1])) != len(SC['ORD']['CM'][1]):
    #             print('Vertical CMs not uniquely defined.')
    #         for ord in SC['ORD']['CM'][0]:
    #             if SC['RING'][ord]['CMlimit'][0] == 0:
    #                 print('HCM limit is zero (Magnet ord: %d). Sure about that?' % ord)
    #         for ord in SC['ORD']['CM'][1]:
    #             if SC['RING'][ord]['CMlimit'][1] == 0:
    #                 print('VCM limit is zero (Magnet ord: %d). Sure about that?' % ord)
    #     if 'Magnet' not in SC['ORD']:
    #         print('No magnets are registered. Use ''SCregisterMagnets''.')
    #     else:
    #         for ord in SC['ORD']['Magnet']:
    #             if len(SC['RING'][ord]['PolynomB']) != len(SC['RING'][ord]['PolynomA']):
    #                 raise ValueError('Length of PolynomB and PolynomA are not equal (Magnet ord: %d)' % ord)
    #             elif len(SC['RING'][ord]['SetPointB']) != len(SC['RING'][ord]['CalErrorB']):
    #                 print('Length of SetPointB and CalErrorB are not equal (Magnet ord: %d)' % ord)
    #             elif len(SC['RING'][ord]['SetPointA']) != len(SC['RING'][ord]['CalErrorA']):
    #                 print('Length of SetPointA and CalErrorA are not equal (Magnet ord: %d)' % ord)
    #             if 'PolynomBOffset' in SC['RING'][ord]:
    #                 if len(SC['RING'][ord]['PolynomBOffset']) != len(SC['RING'][ord]['PolynomAOffset']):
    #                     raise ValueError(
    #                         'Length of PolynomBOffset and PolynomAOffset are not equal (Magnet ord: %d)' % ord)
    #             if 'CombinedFunction' in SC['RING'][ord] and SC['RING'][ord]['CombinedFunction'] == 1:
    #                 if 'BendingAngle' not in SC['RING'][ord]:
    #                     raise ValueError('Combined function magnet (ord: %d) requires field ''BendingAngle''.' % ord)
    #                 if SC['RING'][ord]['NomPolynomB'][1] == 0 or SC['RING'][ord]['BendingAngle'] == 0:
    #                     print(
    #                         'Combined function magnet (ord: %d) has zero bending angle or design quadrupole component.' % ord)
    #             if 'Mag' in SC['SIG'] and len(SC['SIG']['Mag'][ord]) != 0:
    #                 for field in SC['SIG']['Mag'][ord]:
    #                     if field not in SC['RING'][ord]:
    #                         print('Field ''%s'' in SC.SIG.Mag doesn''t match lattice element (Magnet ord: %d)' % (
    #                         field, ord))
    #                     if field == 'MagnetOffset':
    #                         if isinstance(SC['SIG']['Mag'][ord][field], list):
    #                             off = SC['SIG']['Mag'][ord][field][0]
    #                         else:
    #                             off = SC['SIG']['Mag'][ord][field]
    #                         if len(off) != 3:
    #                             print('SC.SIG.Mag{%d}.MagnetOffset should be a [1x3] (dx,dy,dz) array.' % ord)
    #             if 'MasterOf' in SC['RING'][ord]:
    #                 masterFields = SC['RING'][ord].keys()
    #                 for cOrd in SC['RING'][ord]['MasterOf']:
    #                     for field in SC['RING'][cOrd]:
    #                         if field not in masterFields:
    #                             raise ValueError(
    #                                 'Child magnet (ord: %d) has different field ''%s'' than master magnet (ord: %d).' % (
    #                                 cOrd, field, ord))
    #     if 'Cavity' not in SC['ORD']:
    #         print('No cavity registered. Use ''SCregisterCAVs''.')
    #     else:
    #         if len(SC['ORD']['Cavity']) == 0:
    #             print('No cavity registered. Use ''SCregisterBPMs''.')
    #         else:
    #             print('%d cavity/cavities registered.' % len(SC['ORD']['Cavity']))
    #         if len(np.unique(SC['ORD']['Cavity'])) != len(SC['ORD']['Cavity']):
    #             print('Cavities not uniquely defined.')
    #         if 'RF' in SC['SIG']:
    #             for ord in SC['ORD']['Cavity']:
    #                 for field in SC['SIG']['RF'][ord]:
    #                     if field not in SC['RING'][ord]:
    #                         print('Field in SC.SIG.RF doesn''t match lattice element (Cavity ord: %d)' % ord)
    #     if SC['INJ']['beamSize'].shape != (6, 6):
    #         raise ValueError('6x6 sigma matrix has to be used!')
    #     apEl = []
    #     for ord in range(len(SC['RING'])):
    #         if 'EApertures' in SC['RING'][ord] and 'RApertures' in SC['RING'][ord]:
    #             print('Lattice element #%d has both EAperture and RAperture' % ord)
    #         if 'EApertures' in SC['RING'][ord] or 'RApertures' in SC['RING'][ord]:
    #             apEl.append(ord)
    #     if len(apEl) == 0:
    #         print('No apertures found.')
    #     else:
    #         print('Apertures defined in %d out of %d elements.' % (len(apEl), len(SC['RING'])))

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
