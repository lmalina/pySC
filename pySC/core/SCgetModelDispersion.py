import numpy as np
from pySC.at_wrapper import atpass, findorbit6
from pySC.core.SCgetModelRING import SCgetModelRING
import copy


def SCgetModelDispersion(SC, BPMords, CAVords, rfStep=1E3, Z0=np.zeros(6), trackMode='ORB', useIdealRing=True):
    print('Calculating model dispersion')
    track_methods = dict(TBT=atpass, ORB=orbpass)
    if trackMode not in track_methods.keys():
        ValueError(f'Unknown track mode {trackMode}. Valid values are {track_methods.keys()}')

    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)

    trackmethod = track_methods[trackMode]
    if trackMode == 'ORB':
        nTurns = 1
    nBPM = len(BPMords)
    eta = np.full((2 * nBPM * nTurns, 1), np.nan)
    Ta = trackmethod(ring, Z0,  nTurns, BPMords, keep_lattice=False)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')


    for ord in CAVords:
        ring[ord].Frequency += rfStep
    TdB = trackmethod(ring, Z0,  nTurns, BPMords, keep_lattice=False)
   

    dTdB = (TdB - Ta) / rfStep
    eta = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))), np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
    return eta

def orbpass(RING, Z0,  nTurns, REFPTS, keep_lattice):
    return np.transpose(findorbit6(RING, REFPTS, keep_lattice=keep_lattice)[1])[[0,1,2,3], :].reshape(4, 1, len(REFPTS), 1)
