"""
Sub-classes
-------------

This module contains the classes needed in the main data structure (``SimulatedCommissioning``).
"""
import copy

import numpy as np
from numpy import ndarray
from pySC.core.constants import TRACKING_MODES, TRACK_TBT, TRACK_ORB


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
    """
    Define injection parameters for "pySC"

    properties of this class are:
        beamLostAt:
            (default=1) Relative amount of partcles which may be lost before BPM reading is NaN
        Z0ideal:
            (default=numpy.zeros(6)) Design injected trajectory
        Z0:
            (default=numpy.zeros(6)) Injetced trajectory
        beamSize:
            (default=numpy.zeros((6,6))) Injected bunch beam size
        randomInjectionZ:
             (default=numpy.zeros((6,6)))  Injected beam random trajectory jitter
        nParticles:
            (default=1) Number of particles per bunch
        nTurns:
            (default=1) Number of turns for tracking
        nShots:
            (default=1) Number of injections for averaging BPM reading
        trackMode:
            (default='TBT') Tracking mode can be one of:
                TBT (Turn By Turn),
                ORB (Closed Orbit),
                PORB (pseudo Closed Orbit) turn-by-turn tracking with trajectories averaged over the turns

    """
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
        self._trackMode: str = TRACK_TBT
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
        if mode == TRACK_ORB:
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
