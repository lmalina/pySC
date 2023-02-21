from pySC.constants import RF_PROPERTIES


def SCgetModelRING(SC,includeAperture=0):
    RING = SC.IDEALRING
    for ord in range(len(SC.RING)):
        if hasattr(SC.RING[ord], 'SetPointA') and hasattr(SC.RING[ord], 'SetPointB'):
            RING[ord].PolynomA = SC.RING[ord].SetPointA
            RING[ord].PolynomB = SC.RING[ord].SetPointB
            RING[ord].PolynomA[0] = 0.0
            RING[ord].PolynomB[0] = 0.0
        if includeAperture:
            if 'EApertures' in SC.RING[ord]:
                RING[ord].EApertures = SC.RING[ord].EApertures
            if 'RApertures' in SC.RING[ord]:
                RING[ord].RApertures = SC.RING[ord].RApertures
        if len(SC.ORD.Cavity) and hasattr(SC.RING[ord], 'Frequency'):
            for field in RF_PROPERTIES:
                setattr(RING[ord], field, getattr(SC.RING[ord], f"{field}SetPoint"))
    return RING
# End
