import copy
import numpy as np
from at import Lattice
from numpy import ndarray


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
        global plotFunctionFlag
        global SCinjections
        self.RING: Lattice = ring.deepcopy()
        self.IDEALRING: Lattice = ring.deepcopy()
        self.INJ: Injection = Injection()
        self.SIG: Sigmas = Sigmas()
        self.ORD: Indices = Indices()
        SCinjections = 0
        plotFunctionFlag = []


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


class Indices(DotDict):

    def __init__(self):
        super(Indices, self).__init__()
        self.BPM: ndarray = np.array([], dtype=int)
        self.Cavity: ndarray = np.array([], dtype=int)
        self.Magnet: ndarray = np.array([], dtype=int)
        self.SkewQuad: ndarray = np.array([], dtype=int)


class Sigmas(DotDict):

    def __init__(self):
        super(Sigmas, self).__init__()
        self.BPM: DotDict = DotDict()
        self.Mag: DotDict = DotDict()
        self.RF: DotDict = DotDict()
        self.Support: DotDict = DotDict()
