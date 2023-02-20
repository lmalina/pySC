import numpy as np
from pySC.constants import RF_PROPERTIES
from pySC.classes import DotDict, SimulatedComissioning
from numpy import ndarray

def SCregisterCAVs(SC: SimulatedComissioning, CAVords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.ORD.Cavity = np.unique(np.concatenate((SC.ORD.Cavity, CAVords)))
    for ord in CAVords:
        if ord not in SC.SIG.RF.keys():
            SC.SIG.RF[ord] = DotDict()
        SC.SIG.RF[ord].update(kwargs)  # TODO unify SC.SIG and SC.ORD (Cavity vs RF)
        for field in RF_PROPERTIES:
            setattr(SC.RING[ord], f"{field}SetPoint", getattr(SC.RING[ord], field))
            setattr(SC.RING[ord], f"{field}Offset", 0)
            setattr(SC.RING[ord], f"{field}CalError", 0)
    return SC
