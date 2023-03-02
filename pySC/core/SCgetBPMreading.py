import numpy as np

from pySC.core.SCgenBunches import SCgenBunches
from pySC.core.SCparticlesIn3D import SCparticlesIn3D
#from pySC.core.SCplotBPMreading import SCplotBPMreading
from pySC.utils.sc_tools import SCrandnc
from pySC.at_wrapper import atgetfieldvalues, atpass, findorbit6


def SCgetBPMreading(SC, BPMords=[], plotFunctionFlag=False):
    if SC.INJ.trackMode == 'ORB':
        nTurns = 1
        nParticles = 1
    else:
        nTurns = SC.INJ.nTurns
        nParticles = SC.INJ.nParticles
    B1 = np.full((2, nTurns * len(SC.ORD.BPM), SC.INJ.nShots), np.nan)
    if plotFunctionFlag:
        T1 = np.full((6, nTurns * nParticles * len(SC.RING), SC.INJ.nShots), np.nan)
        refOrds = np.arange(len(SC.RING))
    else:
        refOrds = SC.ORD.BPM
    for nShot in range(SC.INJ.nShots):
        if SC.INJ.trackMode == 'ORB':
            T = findorbit6(SC.RING, refOrds)  # ,SC.INJ.Z0)
        else:
            Zin = SCgenBunches(SC)
            T = atpass(SC.RING, Zin, nTurns, refOrds, keep_lattice=False)
        T[:, np.isnan(T[0, :])] = np.nan
        if plotFunctionFlag:
            T1[:, :, nShot] = T
        B1[:, :, nShot] = calcBPMreading(SC, T, atAllElements=plotFunctionFlag)
    B = np.nanmean(B1, 2)
    if plotFunctionFlag:
        pass  # TODO SCplotBPMreading(SC, B, T1)

    if SC.INJ.trackMode == 'pORB':   #  TODO what is pORB?
        Bpseudo = np.full((2, len(SC.ORD.BPM)), np.nan)
        for nBPM in range(len(SC.ORD.BPM)):
            Bpseudo[:, nBPM] = np.nanmean(B[:, nBPM::len(SC.ORD.BPM)], 1)
        B = Bpseudo
    if len(BPMords) > 0:
        ind = np.where(np.isin(SC.ORD.BPM, BPMords))[0]
        if len(ind) != len(BPMords):
            print('Not all specified ordinates are registered BPMs.')
        if SC.INJ.trackMode == 'TBT':
            ind = np.arange(nTurns) * len(SC.ORD.BPM) + ind
        B = B[:, ind]
    return B


def calcBPMreading(SC, T, atAllElements=0):
    if SC.INJ.trackMode == 'ORB':
        nTurns = 1
        BPMnoise = np.array(atgetfieldvalues(SC.RING,SC.ORD.BPM, 'NoiseCO'))
        nParticles = 1
    else:
        nTurns = SC.INJ.nTurns
        BPMnoise = np.array(atgetfieldvalues(SC.RING,SC.ORD.BPM, 'Noise'))
        nParticles = SC.INJ.nParticles
    # TODO for later $D matrices and here no repetition
    BPMoffset = np.array(atgetfieldvalues(SC.RING,SC.ORD.BPM, 'Offset')) + np.array(atgetfieldvalues(SC.RING,SC.ORD.BPM, 'SupportOffset'))
    BPMcalError = np.array(atgetfieldvalues(SC.RING,SC.ORD.BPM, 'CalError'))
    BPMroll = atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Roll') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportRoll')
    BPMnoise = np.repeat(BPMnoise, nTurns, axis=0) * SCrandnc(2, (nTurns * len(SC.ORD.BPM), 2))
    BPMsumError = np.repeat(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SumError'), nTurns)
    if atAllElements:
        nE = np.reshape((np.arange(nTurns) * len(SC.RING) + SC.ORD.BPM), (1, []))
    else:
        nE = np.arange(len(SC.ORD.BPM) * nTurns)
    if nParticles > 1:
        M = SCparticlesIn3D(T, nParticles)
        Tx = np.squeeze(M[0, nE, :])
        Ty = np.squeeze(M[2, nE, :])
        Bx1 = np.nanmean(Tx, axis=1)
        By1 = np.nanmean(Ty, axis=1)
        beamLost = np.nonzero(np.sum(np.isnan(Tx), axis=1) * (1 + BPMsumError * SCrandnc(2, BPMsumError.shape)) > (
                    nParticles * SC.INJ.beamLostAt))
        Bx1[beamLost] = np.nan
        By1[beamLost] = np.nan
    else:
        Bx1 = T[0, nE]
        By1 = T[2, nE]
    Bx = np.cos(BPMroll) * Bx1 - np.sin(BPMroll) * By1
    By = np.sin(BPMroll) * Bx1 + np.cos(BPMroll) * By1
    Bx = (Bx - BPMoffset[:,0]) * (1 + BPMcalError[:, 0])
    By = (By - BPMoffset[:,1]) * (1 + BPMcalError[:, 1])
    Bx = Bx + BPMnoise[0, :]
    By = By + BPMnoise[1, :]
    B = np.array([Bx, By])
    return B
