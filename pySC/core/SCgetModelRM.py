import numpy as np

from pySC.core.SCgetModelRING import SCgetModelRING


def SCgetModelRM(SC,BPMords,CMords,trackMode='TBT',Z0=np.zeros(6),nTurns=1,dkick=1e-5,useIdealRing=0):
    print('Calculating model response matrix')
    if useIdealRing:
        RING = SC.IDEALRING
    else:
        RING = SCgetModelRING(SC)
    if trackMode == 'TBT':
        trackmethod = atpass
    elif trackMode == 'ORB':
        trackmethod = orbpass
        nTurns = 1
    else:
        raise ValueError('trackMode "%s" unknown. Valid values are "TBT" and "ORB".' % trackMode)
    nBPM = len(BPMords)
    nCM = len([CMords[0]])
    RM = np.nan * np.ones((2 * nBPM * nTurns, nCM))
    Ta = trackmethod(RING, Z0, 1, nTurns, BPMords)
    if np.any(np.isnan(Ta)):
        print('Initial trajectory/orbit is NaN. Aborting. ')
        return
    PolynomDim=['PolynomB','PolynomA']
    cnt=0
    for nDim in range(2):
        for CMord in CMords[nDim]:
            if RING[CMord].PassMethod == 'CorrectorPass':
                KickNominal = RING[CMord].KickAngle[nDim]
                RING[CMord].KickAngle[nDim] = KickNominal + dkick
                TdB = trackmethod(RING, Z0, 1, nTurns, BPMords)
                RING[CMord].KickAngle[nDim] =  KickNominal
            else:
                PolynomNominal = RING[CMord][PolynomDim[nDim]]
                delta = dkick / RING[CMord].Length
                RING[CMord][PolynomDim[nDim]][0] = PolynomNominal[0]  + (-1)**(nDim) * delta
                TdB = trackmethod(RING, Z0, 1, nTurns, BPMords)
                RING[CMord][PolynomDim[nDim]] =  PolynomNominal
            dTdB = ( TdB - Ta ) / dkick
            RM[:,cnt] = np.concatenate((dTdB[0,:], dTdB[2,:]))
            cnt=cnt+1
    return RM

def orbpass(RING, Z0, newlat, nTurns, REFPTS):
    OUT = findorbit6(RING,REFPTS,Z0)
    return OUT

# End
