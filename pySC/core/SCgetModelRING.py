from pySC.constants import RF_PROPERTIES
from pySC.classes import SimulatedComissioning
from at import Lattice

def SCgetModelRING(SC: SimulatedComissioning, includeAperture: bool =False) -> Lattice:
    ring = SC.IDEALRING.deepcopy()
    for ord in range(len(SC.RING)):
        if hasattr(SC.RING[ord], 'SetPointA') and hasattr(SC.RING[ord], 'SetPointB'):
            ring[ord].PolynomA = SC.RING[ord].SetPointA
            ring[ord].PolynomB = SC.RING[ord].SetPointB
            ring[ord].PolynomA[0] = 0.0
            ring[ord].PolynomB[0] = 0.0
        if includeAperture:
            if 'EApertures' in SC.RING[ord]:
                ring[ord].EApertures = SC.RING[ord].EApertures
            if 'RApertures' in SC.RING[ord]:
                ring[ord].RApertures = SC.RING[ord].RApertures
        if len(SC.ORD.RF) and hasattr(SC.RING[ord], 'Frequency'):
            for field in RF_PROPERTIES:
                setattr(ring[ord], field, getattr(SC.RING[ord], f"{field}SetPoint"))
    return ring
# End
