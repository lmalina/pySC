import copy
import numpy as np
from at import Lattice
from numpy import ndarray
from pySC.constants import RF_PROPERTIES, SUPPORT_TYPES


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

    def copy(self) -> "DotDict":
        """Returns a shallow copy"""
        return copy.copy(self)

    def deepcopy(self) -> "DotDict":
        """Returns a deep copy"""
        return copy.deepcopy(self)


class SimulatedComissioning(DotDict):
    def __init__(self, ring: Lattice):
        super(SimulatedComissioning, self).__init__()
        self.RING: Lattice = ring.deepcopy()
        self.IDEALRING: Lattice = ring.deepcopy()
        self.INJ: Injection = Injection()
        self.SIG: Sigmas = Sigmas()
        self.ORD: Indices = Indices()

    def register_bpms(self, BPMords: ndarray, **kwargs):
        self.ORD.BPM = np.unique(np.concatenate((self.ORD.BPM, BPMords)))
        for ord in np.unique(BPMords):
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

    def register_cavities(self, CAVords: ndarray, **kwargs):
        self.ORD.Cavity = np.unique(np.concatenate((self.ORD.Cavity, CAVords)))
        for ord in np.unique(CAVords):
            if ord not in self.SIG.RF.keys():
                self.SIG.RF[ord] = DotDict()
            self.SIG.RF[ord].update(kwargs)  # TODO unify self.SIG and self.ORD (Cavity vs RF)
            for field in RF_PROPERTIES:
                setattr(self.RING[ord], f"{field}SetPoint", getattr(self.RING[ord], field))
                setattr(self.RING[ord], f"{field}Offset", 0)
                setattr(self.RING[ord], f"{field}CalError", 0)

    def register_magnets(self, MAGords: ndarray, **kwargs):
        keywords = ['HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf']
        nvpairs = {key: value for key, value in kwargs.items() if key not in keywords}
        self.ORD.Magnet = np.unique(np.concatenate((self.ORD.Magnet, MAGords)))  # TODO unify with Mag in self.SIG
        if 'SkewQuad' in kwargs.keys():
            self.ORD.SkewQuad = np.unique(np.concatenate((self.ORD.SkewQuad, MAGords)))
        if 'HCM' in kwargs.keys():
            self.ORD.HCM = np.unique(np.concatenate((self.ORD.HCM, MAGords)))
        if 'VCM' in kwargs.keys():
            self.ORD.VCM = np.unique(np.concatenate((self.ORD.VCM, MAGords)))
        for ord in MAGords:
            if ord not in self.SIG.Mag.keys():
                self.SIG.Mag[ord] = DotDict()
            self.SIG.Mag[ord].update(nvpairs)

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
            self._optional_magnet_fields(ord, MAGords, **kwargs)

    def register_supports(self, support_ords: ndarray, support_type: str, **kwargs):
        if support_type not in SUPPORT_TYPES:
            raise ValueError(f'Unknown support type ``{support_type}`` found. Allowed are {SUPPORT_TYPES}.')
        if not len(support_ords) or support_ords.shape[0] != 2:
            raise ValueError('Ordinates must be a 2xn array of ordinates.')
        # _checkInput(args)
        self.ORD[support_type] = self._update_double_ordinates(self.ORD[support_type], support_ords)
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
    @staticmethod
    def _update_double_ordinates(ords1, ords2):
        con = np.concatenate((ords1, ords2), axis=1)
        con = con[:, np.lexsort((con[0, :], con[1, :]))]
        return con[:, np.where(np.sum(np.abs(np.diff(con, axis=1)), axis=0))[0]]


class Injection(DotDict):
    def __init__(self):
        super(Injection, self).__init__()
        self.beamLostAt: int = 1
        self.Z0ideal: ndarray = np.zeros(6)
        self.Z0: ndarray = np.zeros(6)
        self.beamSize: ndarray = np.zeros((6, 6))
        self.randomInjectionZ: ndarray = np.zeros(6)
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
        self.Cavity: ndarray = np.array([], dtype=int)
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
        self.Mag: DotDict = DotDict()
        self.RF: DotDict = DotDict()
        self.Support: DotDict = DotDict()
