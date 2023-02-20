import numpy as np
from pySC.classes import DotDict, SimulatedComissioning
from numpy import ndarray


def SCregisterBPMs(SC: SimulatedComissioning, BPMords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.ORD.BPM = np.unique(np.concatenate((SC.ORD.BPM, BPMords)))
    for ord in BPMords:
        if ord not in SC.SIG.BPM.keys():
            SC.SIG.BPM[ord] = DotDict()
        SC.SIG.BPM[ord].update(kwargs)

        SC.RING[ord].Noise = np.zeros(2)
        SC.RING[ord].NoiseCO = np.zeros(2)
        SC.RING[ord].Offset = np.zeros(2)
        SC.RING[ord].SupportOffset = np.zeros(2)
        SC.RING[ord].Roll = 0
        SC.RING[ord].SupportRoll = 0
        SC.RING[ord].CalError = np.zeros(2)
        SC.RING[ord].SumError = 0
    return SC
