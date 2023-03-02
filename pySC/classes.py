import copy
import re
import numpy as np
from at import Lattice
from numpy import ndarray
from pySC.constants import RF_PROPERTIES, SUPPORT_TYPES
from pySC.utils.classdef_tools import add_padded, randn_cutoff, update_double_ordinates, intersect
from pySC.utils.sc_tools import SCrandnc, SCscaleCircumference, SCgetTransformation
from pySC.at_wrapper import findspos
from pySC.core.SCgetSupportOffsetRoll import support_offset_and_roll  # TODO maybe move


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


class Injection(DotDict):
    def __init__(self):
        super(Injection, self).__init__()
        self.beamLostAt: int = 1
        self.Z0ideal: ndarray = np.zeros(6)
        self.Z0: ndarray = np.zeros(6)
        self.beamSize: ndarray = np.zeros((6, 6))
        self.randomInjectionZ: ndarray = np.zeros(6)
        self.staticInjectionZ: ndarray = np.zeros(6)
        self.nParticles: int = 1
        self.nTurns: int = 1
        self.nShots: int = 1
        self.trackMode: str = 'TBT'
        self.postFun = self._dummy_func

    @staticmethod
    def _dummy_func(matrix: ndarray) -> ndarray:
        return matrix


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
        self.INJ: Injection = Injection()
        self.SIG: Sigmas = Sigmas()
        self.ORD: Indices = Indices()

    def register_bpms(self, ords: ndarray, **kwargs):
        self.ORD.BPM = np.unique(np.concatenate((self.ORD.BPM, ords)))
        for ord in np.unique(ords):
            if ord not in self.SIG.BPM.keys():
                self.SIG.BPM[ord] = DotDict()
            self.SIG.BPM[ord].update(kwargs)

            self.RING[ord].Noise = np.zeros(2)
            self.RING[ord].NoiseCO = np.zeros(2)
            self.RING[ord].Offset = np.zeros(2)
            self.RING[ord].SupportOffset = np.zeros(2)
            self.RING[ord].Roll = 0
            self.RING[ord].SupportRoll = 0
            self.RING[ord].CalError = np.zeros(2)
            self.RING[ord].SumError = 0

    def register_cavities(self, ords: ndarray, **kwargs):
        self.ORD.RF = np.unique(np.concatenate((self.ORD.RF, ords)))
        for ord in np.unique(ords):
            if ord not in self.SIG.RF.keys():
                self.SIG.RF[ord] = DotDict()
            self.SIG.RF[ord].update(kwargs)
            for field in RF_PROPERTIES:
                setattr(self.RING[ord], f"{field}SetPoint", getattr(self.RING[ord], field))
                setattr(self.RING[ord], f"{field}Offset", 0)
                setattr(self.RING[ord], f"{field}CalError", 0)

    def register_magnets(self, ords: ndarray, **kwargs):
        keywords = ['HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf']
        nvpairs = {key: value for key, value in kwargs.items() if key not in keywords}
        self.ORD.Magnet = np.unique(np.concatenate((self.ORD.Magnet, ords)))
        if 'SkewQuad' in kwargs.keys():
            self.ORD.SkewQuad = np.unique(np.concatenate((self.ORD.SkewQuad, ords)))
        if 'HCM' in kwargs.keys():
            self.ORD.HCM = np.unique(np.concatenate((self.ORD.HCM, ords)))
        if 'VCM' in kwargs.keys():
            self.ORD.VCM = np.unique(np.concatenate((self.ORD.VCM, ords)))
        for ord in ords:
            if ord not in self.SIG.Magnet.keys():
                self.SIG.Magnet[ord] = DotDict()
            self.SIG.Magnet[ord].update(nvpairs)

            self.RING[ord].NomPolynomB = self.RING[ord].PolynomB[:]
            self.RING[ord].NomPolynomA = self.RING[ord].PolynomA[:]
            self.RING[ord].SetPointB = self.RING[ord].PolynomB[:]
            self.RING[ord].SetPointA = self.RING[ord].PolynomA[:]
            self.RING[ord].CalErrorB = np.zeros(len(self.RING[ord].PolynomB))
            self.RING[ord].CalErrorA = np.zeros(len(self.RING[ord].PolynomA))
            self.RING[ord].MagnetOffset = np.zeros(3)
            self.RING[ord].SupportOffset = np.zeros(3)
            self.RING[ord].MagnetRoll = np.zeros(3)
            self.RING[ord].SupportRoll = np.zeros(3)
            self.RING[ord].T1 = np.zeros(6)
            self.RING[ord].T2 = np.zeros(6)
            self._optional_magnet_fields(ord, ords, **kwargs)

    def register_supports(self, support_ords: ndarray, support_type: str, **kwargs):
        if support_type not in SUPPORT_TYPES:
            raise ValueError(f'Unknown support type ``{support_type}`` found. Allowed are {SUPPORT_TYPES}.')
        if not len(support_ords) or support_ords.shape[0] != 2:
            raise ValueError('Ordinates must be a 2xn array of ordinates.')
        # _checkInput(args)
        self.ORD[support_type] = update_double_ordinates(self.ORD[support_type], support_ords)
        for ord in np.ravel(support_ords):
            setattr(self.RING[ord], f"{support_type}Offset", np.zeros(3))  # [x,y,z]
            setattr(self.RING[ord], f"{support_type}Roll", np.zeros(3))  # [az,ax,ay]
            self.SIG.Support[ord] = DotDict()
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
        for ord in intersect(self.ORD.RF, self.SIG.RF.keys()):
            for field in self.SIG.RF[ord]:
                setattr(self.RING[ord], field, randn_cutoff(self.SIG.RF[ord][field], nsigmas))
        # BPM
        for ord in intersect(self.ORD.BPM, self.SIG.BPM.keys()):
            for field in self.SIG.BPM[ord]:
                if re.search('Noise', field):
                    setattr(self.RING[ord], field, self.SIG.BPM[ord][field])
                else:
                    setattr(self.RING[ord], field, randn_cutoff(self.SIG.BPM[ord][field], nsigmas))
        # Magnet
        for ord in intersect(self.ORD.Magnet, self.SIG.Magnet.keys()):
            for field in self.SIG.Magnet[ord]:
                setattr(self.RING[ord], 'BendingAngleError' if field == 'BendingAngle' else field,
                        randn_cutoff(self.SIG.Magnet[ord][field], nsigmas))
        # Injection
        self.INJ.Z0 = self.INJ.Z0ideal + self.SIG.staticInjectionZ * SCrandnc(nsigmas, (6,))
        self.INJ.randomInjectionZ = self.SIG.randomInjectionZ[:]
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

                rolls0 = getattr(self.RING[ordPair[0]], f"{support_type}Roll")  # Twisted supports are not considered
                offsets0 = getattr(self.RING[ordPair[0]], f"{support_type}Offset")
                offsets1 = getattr(self.RING[ordPair[1]], f"{support_type}Offset")

                if rolls0[1] != 0:
                    if f"{support_type}Offset" in self.SIG.Support[ordPair[1]].keys():
                        raise Exception(
                            f'Pitch angle errors can not be given explicitly if {support_type} start and endpoints '
                            f'each have offset uncertainties.')
                    offsets0[1] -= rolls0[1] * struct_length / 2
                    offsets1[1] += rolls0[1] * struct_length / 2

                else:
                    rolls0[1] = (offsets1[1] - offsets0[1]) / struct_length
                if rolls0[2] != 0:
                    if f"{support_type}Offset" in self.SIG.Support[ordPair[1]].keys():
                        raise Exception(
                            f'Yaw angle errors can not be given explicitly if {support_type} start and endpoints '
                            f'each have offset uncertainties.')
                    offsets0[0] -= rolls0[2] * struct_length / 2
                    offsets1[0] += rolls0[2] * struct_length / 2
                else:
                    rolls0[2] = (offsets1[0] - offsets0[0]) / struct_length
                setattr(self.RING[ordPair[0]], f"{support_type}Roll", rolls0)
                setattr(self.RING[ordPair[0]], f"{support_type}Offset", offsets0)
                setattr(self.RING[ordPair[1]], f"{support_type}Offset", offsets1)

    def update_cavities(self, ords: ndarray = None):
        for ord in (self.ORD.RF if ords is None else ords):
            for field in RF_PROPERTIES:
                setattr(self.RING[ord], field,
                        getattr(self.RING[ord], f"{field}SetPoint")
                        * (1 + getattr(self.RING[ord], f"{field}CalError"))
                        + getattr(self.RING[ord], f"{field}Offset"))

    def update_magnets(self, ords: ndarray = None):
        for ord in (self.ORD.Magnet if ords is None else ords):
            self._updateMagnets(ord, ord)
            if hasattr(self.RING[ord], 'MasterOf'):
                for childOrd in self.RING[ord].MasterOf:
                    self._updateMagnets(ord, childOrd)

    def update_supports(self, offset_bpms: bool = True, offset_magnets: bool = True):
        if offset_magnets:
            if len(self.ORD.Magnet):
                s = findspos(self.RING, self.ORD.Magnet)
                offsets, rolls = support_offset_and_roll(self, s)
                for i, ord in enumerate(self.ORD.Magnet):
                    setattr(self.RING[ord], "SupportOffset", offsets[:, i])
                    setattr(self.RING[ord], "SupportRoll", rolls[:, i])
                    magLength = self.RING[ord].Length
                    magTheta = self.RING[ord].BendingAngle if hasattr(self.RING[ord], 'BendingAngle') else 0
                    magnet_offsets = self.RING[ord].SupportOffset + self.RING[ord].MagnetOffset
                    magnet_rolls = np.roll(self.RING[ord].MagnetRoll + self.RING[ord].SupportRoll, -1)  # z,x,y -> x,y,z
                    self.RING[ord].T1, self.RING[ord].T2, self.RING[ord].R1, self.RING[ord].R2 = SCgetTransformation(
                        magnet_offsets, magnet_rolls, magTheta, magLength)
                    if hasattr(self.RING[ord], 'MasterOf'):
                        for childOrd in self.RING[ord].MasterOf:
                            for field in ("T1", "T2", "R1", "R2"):
                                setattr(self.RING[childOrd], field, getattr(self.RING[ord], field))
            else:
                print('SC: No magnets have been registered!')
        if offset_bpms:
            if len(self.ORD.BPM):
                s = findspos(self.RING, self.ORD.BPM)
                offsets, rolls = support_offset_and_roll(self, s)
                for i, ord in enumerate(self.ORD.BPM):
                    setattr(self.RING[ord], "SupportOffset", offsets[0:2, i])  # Longitudinal BPM offsets not implemented
                    setattr(self.RING[ord], "SupportRoll",
                            np.array([rolls[0, i]]))  # BPM pitch and yaw angles not  implemented
            else:
                print('SC: No BPMs have been registered!')

    def verify_structure(self):
        raise NotImplementedError

    def _optional_magnet_fields(self, ord, MAGords, **kwargs):
        if 'CF' in kwargs.keys():
            self.RING[ord].CombinedFunction = 1
        if "HCM" in kwargs.keys() or "VCM" in kwargs.keys():
            self.RING[ord].CMlimit = np.zeros(2)
        if 'HCM' in kwargs.keys():
            self.RING[ord].CMlimit[0] = kwargs["HCM"]
        if 'VCM' in kwargs.keys():
            self.RING[ord].CMlimit[1] = kwargs['VCM']
        if 'SkewQuad' in kwargs.keys():
            self.RING[ord].SkewQuadLimit = kwargs['SkewQuad']
        if 'MasterOf' in kwargs.keys():
            if np.count_nonzero(MAGords == ord) > 1:
                raise ValueError(f"Non-unique element index {ord} found together with ``MasterOf``")
            self.RING[ord].MasterOf = kwargs['MasterOf'][:, np.nonzero(MAGords == ord)].ravel()


    def _updateMagnets(self, source, target):  # TODO simplify AB calculated in place
        self.RING[target].PolynomB = self.RING[source].SetPointB * add_padded(np.ones(len(self.RING[source].SetPointB)),
                                                                              self.RING[source].CalErrorB)
        self.RING[target].PolynomA = self.RING[source].SetPointA * add_padded(np.ones(len(self.RING[source].SetPointA)),
                                                                              self.RING[source].CalErrorA)
        sysPolynomB = []
        sysPolynomA = []
        if hasattr(self.RING[target], 'SysPolBFromB'):
            for n in range(len(self.RING[target].SysPolBFromB)):
                if self.RING[target].SysPolBFromB[n] is not None:
                    sysPolynomB.append(self.RING[target].PolynomB[n] * self.RING[target].SysPolBFromB[n])
        if hasattr(self.RING[target], 'SysPolBFromA'):
            for n in range(len(self.RING[target].SysPolBFromA)):
                if self.RING[target].SysPolBFromA[n] is not None:
                    sysPolynomB.append(self.RING[target].PolynomA[n] * self.RING[target].SysPolBFromA[n])
        if hasattr(self.RING[target], 'SysPolAFromB'):
            for n in range(len(self.RING[target].SysPolAFromB)):
                if self.RING[target].SysPolAFromB[n] is not None:
                    sysPolynomA.append(self.RING[target].PolynomB[n] * self.RING[target].SysPolAFromB[n])
        if hasattr(self.RING[target], 'SysPolAFromA'):
            for n in range(len(self.RING[target].SysPolAFromA)):
                if self.RING[target].SysPolAFromA[n] is not None:
                    sysPolynomA.append(self.RING[target].PolynomA[n] * self.RING[target].SysPolAFromA[n])
        if len(sysPolynomA) > 0:
            for n in range(len(sysPolynomA) - 1):
                sysPolynomA[n + 1] = add_padded(sysPolynomA[n + 1], sysPolynomA[n])
            self.RING[target].PolynomA = add_padded(self.RING[target].PolynomA, sysPolynomA[-1])
        if len(sysPolynomB) > 0:
            for n in range(len(sysPolynomB) - 1):
                sysPolynomB[n + 1] = add_padded(sysPolynomB[n + 1], sysPolynomB[n])
            self.RING[target].PolynomB = add_padded(self.RING[target].PolynomB, sysPolynomB[-1])
        if hasattr(self.RING[target], 'PolynomBOffset'):
            self.RING[target].PolynomB = add_padded(self.RING[target].PolynomB, self.RING[target].PolynomBOffset)
            self.RING[target].PolynomA = add_padded(self.RING[target].PolynomA, self.RING[target].PolynomAOffset)
        if hasattr(self.RING[source], 'BendingAngleError'):
            self.RING[target].PolynomB[0] = self.RING[target].PolynomB[0] + self.RING[source].BendingAngleError * self.RING[
                target].BendingAngle / self.RING[target].Length
        if hasattr(self.RING[source], 'BendingAngle'):
            if hasattr(self.RING[source], 'CombinedFunction') and self.RING[source].CombinedFunction == 1:
                alpha_act = self.RING[source].SetPointB[1] * (1 + self.RING[source].CalErrorB[1]) / self.RING[source].NomPolynomB[
                    1]
                effBendingAngle = alpha_act * self.RING[target].BendingAngle
                self.RING[target].PolynomB[0] = self.RING[target].PolynomB[0] + (
                        effBendingAngle - self.RING[target].BendingAngle) / self.RING[target].Length
        if self.RING[source].PassMethod == 'CorrectorPass':
            self.RING[target].KickAngle[0] = self.RING[target].PolynomB[0]
            self.RING[target].KickAngle[1] = self.RING[target].PolynomA[0]
        self.RING[target].MaxOrder = len(self.RING[target].PolynomB) - 1
