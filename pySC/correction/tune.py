import numpy as np
import matplotlib.pyplot as plt

from pySC.core.beam import SCgetBeamTransmission
from pySC.core.lattice_setting import SCsetMags2SetPoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

def SCtuneScan(SC, qOrds, qSPvec, plotFlag=False, nParticles=None, nTurns=None, target=1, fullScan=0):
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    SC.INJ.nParticles = nParticles  # TODO
    SC.INJ.nTurns = nTurns
    maxTurns = np.empty((len(qSPvec[0]), len(qSPvec[1])))
    maxTurns[:] = np.nan
    finTrans = np.empty((len(qSPvec[0]), len(qSPvec[1]), nTurns))
    finTrans[:] = np.nan
    ERROR = 2
    qSP = []
    allInd = []
    tmp = spiral(max(len(qSPvec[0]), len(qSPvec[1])))
    idx = np.argsort(tmp.flatten())
    q1Ind, q2Ind = np.unravel_index(idx, (max(len(qSPvec[0]), len(qSPvec[1])), max(len(qSPvec[0]), len(qSPvec[1]))))
    for i in range(len(q1Ind)):
        q1 = q1Ind[i]
        q2 = q2Ind[i]
        ords = np.hstack(qOrds)
        setpoints = np.hstack((np.repeat(qSPvec[0][q1], len(qOrds[0])), np.repeat(qSPvec[1][q2], len(qOrds[1]))))
        SC = SCsetMags2SetPoints(SC, ords, False, 1, setpoints, method='rel')
        maxTurns[q1, q2], lostCount = SCgetBeamTransmission(SC, nParticles=nParticles, nTurns=nTurns)
        finTrans[q1, q2, :] = 1 - lostCount
        allInd.append([q1, q2])
        if plotFlag:
            plt.figure(185)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(100 * finTrans[:, :, -1])
            c1 = plt.colorbar(orientation='horizontal')
            c1.set_label('Beam transmission [%]')
            plt.clim(0, 100)
            plt.ylabel(SC.RING[qOrds[1][0]].FamName + ' [rel. to nom setpoint]')
            plt.xlabel(SC.RING[qOrds[0][0]].FamName + ' [rel. to nom setpoint]')
            tickInd = np.unique(np.round(np.hstack((np.linspace(1, len(qSPvec[0]), 5), len(qSPvec[0])))))
            plt.yticks(tickInd, qSPvec[1][tickInd])
            plt.xticks(tickInd, qSPvec[0][tickInd])
            plt.subplot(2, 2, 2)
            plt.imshow(maxTurns)
            c1 = plt.colorbar(orientation='horizontal')
            c1.set_label('Number of achieved turns')
            plt.clim(1, nTurns)
            plt.ylabel(SC.RING[qOrds[1][0]].FamName + ' [rel. to nom setpoint]')
            plt.xlabel(SC.RING[qOrds[0][0]].FamName + ' [rel. to nom setpoint]')
            tickInd = np.unique(np.round(np.hstack((np.linspace(1, len(qSPvec[0]), 5), len(qSPvec[0])))))
            plt.yticks(tickInd, qSPvec[1][tickInd])
            plt.xticks(tickInd, qSPvec[0][tickInd])
            plt.subplot(2, 2, [3, 4])
            plt.plot(lostCount)
            plt.plot([0, nTurns], [SC.INJ.beamLostAt, SC.INJ.beamLostAt], 'k:')
            plt.xlim(0, nTurns)
            plt.ylim(0, 1)
            plt.xlabel('Number of turns')
            plt.ylabel('EDF of lost count')
            plt.show()
        if not fullScan:
            if finTrans[q1, q2, -1] >= target:
                ERROR = 0
                qSP.append(qSPvec[0][q1])
                qSP.append(qSPvec[1][q2])
                LOGGER.debug(f'Transmission target reached with:\n  '
                             f'{SC.RING[qOrds[0][0]].FamName} SetPoint: {qSP[0]:.4f}\n '
                             f' {SC.RING[qOrds[1][0]].FamName} SetPoint: {qSP[1]:.4f}\n')
                return qSP, SC, maxTurns, finTrans, ERROR
    testTrans = np.zeros(len(allInd))
    testTurns = np.zeros(len(allInd))
    for i in range(len(allInd)):
        testTrans[i] = finTrans[allInd[i][0], allInd[i][1], -1]
        testTurns[i] = maxTurns[allInd[i][0], allInd[i][1]]
    a, b = np.sort(testTrans)[::-1], np.argsort(testTrans)[::-1]
    if a[0] == 0:
        a, b = np.sort(testTurns)[::-1], np.argsort(testTurns)[::-1]
        if a[0] == 0:
            raise RuntimeError('Fail, no transmission at all.\n')
        LOGGER.debug(f'No transmission at final turn at all. Best number of turns ({a[0]:d}) reached with:\n  '
                     f'{SC.RING[qOrds[0][0]].FamName} SetPoint: {qSPvec[0][allInd[b[0]][0]]:.4f}\n  '
                     f'{SC.RING[qOrds[1][0]].FamName} SetPoint: {qSPvec[1][allInd[b[0]][1]]:.4f}\n')
    else:
        LOGGER.debug(f'Transmission target not reached. Best value ({a[0]:d}) reached with:\n  '
                     f'{SC.RING[qOrds[0][0]].FamName} SetPoint: {qSPvec[0][allInd[b[0]][0]]:.4f}\n  '
                     f'{SC.RING[qOrds[1][0]].FamName} SetPoint: {qSPvec[1][allInd[b[0]][1]]:.4f}\n')
    qSP.append(qSPvec[0][allInd[b[0]][0]])
    qSP.append(qSPvec[1][allInd[b[0]][1]])
    if qSP[0] == qSPvec[0][q1Ind[0]] and qSP[1] == qSPvec[1][q2Ind[0]]:
        raise RuntimeError('No improvement possible.\n')
    else:
        ERROR = 1
    ords = np.hstack(qOrds)
    setpoints = np.hstack((np.repeat(qSP[0], len(qOrds[0])), np.repeat(qSP[1], len(qOrds[1]))))
    SC = SCsetMags2SetPoints(SC, ords, False, 1, setpoints, method='rel')
    return qSP, SC, maxTurns, finTrans, ERROR
