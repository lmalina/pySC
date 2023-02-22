import numpy as np
import matplotlib.pyplot as plt

from pySC.core.SCgenBunches import SCgenBunches
from pySC.core.SCparticlesIn3D import SCparticlesIn3D
from pySC.at_wrapper import atpass

def SCgetBeamTransmission(SC, nParticles=None, nTurns=None, plotFlag=0, verbose=0):  # TODO Check if SC gets modified
    if nParticles is not None:
        SC.INJ.nParticles = nParticles
    if nTurns is not None:
        SC.INJ.nTurns = nTurns
    if verbose:
        print('Calculating maximum beam transmission for %d particles and %d turns: ' % (
        SC.INJ.nParticles, SC.INJ.nTurns), end='')
    Zin = SCgenBunches(SC)
    T = atpass(SC.RING, Zin, 1, SC.INJ.nTurns, len(SC.RING) + 1)
    if SC.INJ.nParticles > 1:
        M = SCparticlesIn3D(T, SC.INJ.nParticles)
        Tx = np.squeeze(M[0, :, :])
        maxTurns = np.where(np.sum(np.isnan(Tx), 1) > (SC.INJ.nParticles * SC.INJ.beamLostAt))[0][0] - 1
    else:
        Tx = T[0, :]
        maxTurns = np.where(np.isnan(Tx))[0][0] - 1
    if not maxTurns:
        maxTurns = SC.INJ.nTurns
        ERROR = 0
    lostCount = np.sum(np.isnan(Tx), 1) / SC.INJ.nParticles
    if plotFlag:
        plt.figure(12), plt.clf()
        plt.plot(lostCount)
        plt.plot([0, SC.INJ.nTurns], [SC.INJ.beamLostAt, SC.INJ.beamLostAt], 'k:')
        plt.xlim([0, SC.INJ.nTurns])
        plt.ylim([0, 1])
        plt.xlabel('Number of turns')
        plt.ylabel('EDF of lost count')
        plt.show()
    if verbose:
        print('%d turns and %.0f%% transmission.' % (maxTurns, 100 * (1 - lostCount[-1])))
    return maxTurns, lostCount

# SCgetBeamTransmission(SC,nParticles=1,nTurns=1000,plotFlag=1,verbose=1)
