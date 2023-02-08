def SCgetModelRING(SC,includeAperture=0):
    RING = SC.IDEALRING
    for ord in range(len(SC.RING)):
        if 'SetPointA' in SC.RING[ord] and 'SetPointB' in SC.RING[ord]:
            RING[ord].PolynomA = SC.RING[ord].SetPointA
            RING[ord].PolynomB = SC.RING[ord].SetPointB
            RING[ord].PolynomA[0] = 0.0
            RING[ord].PolynomB[0] = 0.0
        if includeAperture:
            if 'EApertures' in SC.RING[ord]:
                RING[ord].EApertures = SC.RING[ord].EApertures
            if 'RApertures' in SC.RING[ord]:
                RING[ord].RApertures = SC.RING[ord].RApertures
        if 'Cavity' in SC.ORD and 'Frequency' in RING[ord]:
            for field in ['Frequency','Voltage','TimeLag']:
                RING[ord][field] = SC.RING[ord][field+'SetPoint']
    return RING
# End
