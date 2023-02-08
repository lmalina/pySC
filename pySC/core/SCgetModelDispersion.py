import numpy as np


def SCgetModelDispersion(SC,BPMords,CAVords,rfStep=1E3,useIdealRing=0):
    RING = SC.IDEALRING
    if not useIdealRing:
        for ord in range(len(RING)):
            if 'SetPointA' in RING[ord] and 'SetPointB' in RING[ord]:
                RING[ord]['PolynomA'] = SC.RING[ord]['SetPointA']
                RING[ord]['PolynomB'] = SC.RING[ord]['SetPointB']
                RING[ord]['PolynomA'][0] = 0.0
                RING[ord]['PolynomB'][0] = 0.0
            rmfs = list(set(RING[ord].keys()).intersection({'T1','T2','R1','R2','EApertures','RApertures'}))
            for rmf in rmfs:
                del RING[ord][rmf]
    nBPM = len(BPMords)
    eta = np.nan * np.ones((2 * nBPM , 1))
    Bref = findorbit6(RING,BPMords)
    if np.any(np.isnan(Bref)):
        print('Initial orbit is NaN. Aborting. \n')
        return
    for ord in CAVords:
        RING[ord]['Frequency'] = RING[ord]['Frequency'] + rfStep
    B = findorbit6(RING,BPMords)
    eta = np.reshape((B[0:2,:] - Bref[0:2,:])/rfStep,[],1)
    return eta
# End
# Test

# eta = SCgetModelDispersion(SC,BPMords,CAVords)
# print(eta)
# End
