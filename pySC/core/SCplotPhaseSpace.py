import at
import matplotlib.pyplot as plt
import numpy as np

from pySC.core.SCgenBunches import SCgenBunches
from pySC.core.SCparticlesIn3D import SCparticlesIn3D


def SCplotPhaseSpace(SC, ord=1, plotCO=0, customBunch=[], nParticles=None, nTurns=None):
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    SC.INJ.nParticles = nParticles
    SC.INJ.nTurns = nTurns
    if len(customBunch) == 0:
        Zin = SCgenBunches(SC)
    else:
        Zin = customBunch
        SC.INJ.nParticles = Zin.shape[1]
    T = at.atpass(SC.RING, Zin, 1, SC.INJ.nTurns, ord)
    T[:, np.isnan(T[0, :])] = np.nan
    T3D = SCparticlesIn3D(T, SC.INJ.nParticles)
    labelStr = ['$\Delta x$ [$\mu$m]', '$\Delta x''$ [$\mu$rad]', '$\Delta y$ [$\mu$m]', '$\Delta y''$ [$\mu$rad]',
                '$\Delta S$ [m]', '$\delta E$ $[\%]$']
    titleStr = ['Horizontal', 'Vertical', 'Longitudinal']
    if SC.RING[SC.ORD.Cavity[0]].PassMethod == 'RFCavityPass':
        L0_tot = 0
        for i in range(len(SC.RING)):
            L0_tot = L0_tot + SC.RING[i].Length
        lengthSlippage = 299792458 * (
                    SC.RING[SC.ORD.Cavity[0]].HarmNumber / SC.RING[SC.ORD.Cavity[0]].Frequency - L0_tot / 299792458)
        T3D[6, :, :] = T3D[6, :, :] - lengthSlippage * np.arange(1, nTurns + 1)
        labelStr[5] = '$\Delta S_{act}$ [m]'
    if plotCO:
        CO = at.find_orbit6(SC.RING, ord)
        if np.isnan(CO[0]):
            startPointGuess = np.nanmean(np.nanmean(T3D, 2), 2)
            CO = at.find_orbit6(SC.RING, ord, startPointGuess)
            if np.isnan(CO[0]):
                CO = np.nan * np.ones(6)
    else:
        CO = np.nan * np.ones(6)
    T3D = T3D * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    SC.INJ.Z0 = SC.INJ.Z0 * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    CO = CO * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    T3D[[5, 6], :, :] = T3D[[6, 5], :, :]
    CO[[5, 6], :] = CO[[6, 5], :]
    SC.INJ.Z0[[5, 6]] = SC.INJ.Z0[[6, 5]]
    plt.figure(100)
    plt.clf()
    pVec = []
    legStr = []
    for nType in range(3):
        plt.subplot(1, 3, nType + 1);
        plt.hold(True)
        for nP in range(SC.INJ.nParticles):
            x = T3D[2 * nType, :, nP]
            y = T3D[2 * nType + 1, :, nP]
            plt.scatter(x, y, 10, np.arange(1, nTurns + 1))
        pVec.append(plt.plot(SC.INJ.Z0[2 * nType], SC.INJ.Z0[2 * nType + 1], 'O', 'MarkerSize', 15, 'LineWidth', 3))
        legStr.append('Injection point')
        if plotCO:
            pVec.append(plt.plot(CO[2 * nType], CO[2 * nType + 1], 'X', 'MarkerSize', 20, 'LineWidth', 3))
            legStr.append('Closed orbit')
        plt.gca().set_box('on')
        plt.xlabel(labelStr[2 * nType])
        plt.ylabel(labelStr[2 * nType + 1])
        plt.title(titleStr[nType] + ' @Ord: ' + str(ord))
    plt.legend(pVec, legStr)
    c = plt.colorbar()
    c.set_label('Number of turns')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.gcf().set_dpi(100)
    plt.gcf().set_facecolor('w')
    plt.show()
    return
