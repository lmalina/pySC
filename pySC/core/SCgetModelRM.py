import numpy as np
from pySC.core.SCgetModelRING import SCgetModelRING
from pySC.at_wrapper import atpass, findorbit6
from pySC.constants import NUM_TO_AB
import copy


def SCgetModelRM(SC, BPMords, CMords, trackMode='TBT', Z0=np.zeros(6), nTurns=1, dkick=1e-5, useIdealRing=True):
    print('Calculating model response matrix')
    track_methods = dict(TBT=atpass, ORB=orbpass)
    if trackMode not in track_methods.keys():
        ValueError(f'Unknown track mode {trackMode}. Valid values are {track_methods.keys()}')
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)
    trackmethod = track_methods[trackMode]
    if trackMode == 'ORB':
        nTurns = 1
    nBPM = len(BPMords)
    nCM = len(CMords[0]) + len(CMords[1])
    RM = np.full((2 * nBPM * nTurns, nCM), np.nan)
    Ta = trackmethod(ring, Z0,  nTurns, BPMords, keep_lattice=False)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')

    cnt = 0
    for nDim in range(2):
        for CMord in CMords[nDim]:
            if ring[CMord].PassMethod == 'CorrectorPass':
                KickNominal = ring[CMord].KickAngle[nDim]
                ring[CMord].KickAngle[nDim] = KickNominal + dkick
                TdB = trackmethod(ring, Z0, nTurns, BPMords, keep_lattice=False)
                ring[CMord].KickAngle[nDim] = KickNominal
            else:
                PolynomNominal = getattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}")
                delta = dkick / ring[CMord].Length
                changed_polynom = copy.deepcopy(PolynomNominal[:])
                changed_polynom[0] += (-1) ** (nDim+1) * delta
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", changed_polynom[:])
                TdB = trackmethod(ring, Z0, nTurns, BPMords, keep_lattice=False)
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", PolynomNominal[:])
            dTdB = (TdB - Ta) / dkick
            RM[:, cnt] = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))), np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
            cnt = cnt + 1
    return RM


def orbpass(RING, Z0,  nTurns, REFPTS, keep_lattice):
    return np.transpose(findorbit6(RING, REFPTS, keep_lattice=keep_lattice)[1])[[0,1,2,3], :].reshape(4, 1, len(REFPTS), 1)
