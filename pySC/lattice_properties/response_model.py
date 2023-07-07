import numpy as np
from at import Lattice

from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.at_wrapper import atpass, findorbit6
from pySC.core.constants import NUM_TO_AB, RF_PROPERTIES
import copy
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

def SCgetModelRM(SC, BPMords, CMords, trackMode='TBT', Z0=np.zeros(6), nTurns=1, dkick=1e-5, useIdealRing=True):
    LOGGER.info('Calculating model response matrix')
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
    Ta = trackmethod(ring, Z0,  nTurns, BPMords)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')

    cnt = 0
    for nDim in range(2):
        for CMord in CMords[nDim]:
            if ring[CMord].PassMethod == 'CorrectorPass':
                KickNominal = ring[CMord].KickAngle[nDim]
                ring[CMord].KickAngle[nDim] = KickNominal + dkick
                TdB = trackmethod(ring, Z0, nTurns, BPMords)
                ring[CMord].KickAngle[nDim] = KickNominal
            else:
                PolynomNominal = getattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}")
                delta = dkick / ring[CMord].Length
                changed_polynom = copy.deepcopy(PolynomNominal[:])
                changed_polynom[0] += (-1) ** (nDim+1) * delta
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", changed_polynom[:])
                TdB = trackmethod(ring, Z0, nTurns, BPMords)
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", PolynomNominal[:])
            dTdB = (TdB - Ta) / dkick
            RM[:, cnt] = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))),
                                         np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
            cnt = cnt + 1
    return RM


def orbpass(RING, Z0,  nTurns, REFPTS):
    return np.transpose(findorbit6(RING, REFPTS)[1])[[0,1,2,3], :].reshape(4, 1, len(REFPTS), 1)


def SCgetModelDispersion(SC, BPMords, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1, rfStep=1E3, useIdealRing=True):
    LOGGER.info('Calculating model dispersion')
    track_methods = dict(TBT=atpass, ORB=orbpass)
    if trackMode not in track_methods.keys():
        ValueError(f'Unknown track mode {trackMode}. Valid values are {track_methods.keys()}')
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)
    trackmethod = track_methods[trackMode]
    if trackMode == 'ORB':
        nTurns = 1

    Ta = trackmethod(ring, Z0,  nTurns, BPMords)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')

    for ord in CAVords:  # Single point with all cavities with the same frequency shift
        ring[ord].Frequency += rfStep
    TdB = trackmethod(ring, Z0,  nTurns, BPMords)
    dTdB = (TdB - Ta) / rfStep
    eta = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))),
                          np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
    return eta


def SCgetModelRING(SC: SimulatedCommissioning, includeAperture: bool =False) -> Lattice:
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
