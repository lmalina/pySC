import matplotlib.pyplot as plt
import numpy as np
from pySC.utils.at_wrapper import atpass, findorbit6
from pySC.core.beam import SCgenBunches

SPEED_OF_LIGHT = 299792458


def SCplotPhaseSpace(SC, ord=np.zeros(1), customBunch=[], nParticles=None, nTurns=None, plotCO=False):
    if len(customBunch):
        Zin = customBunch[:]
        nParticles = Zin.shape[1]
    else:
        if nParticles is None:
            nParticles = SC.INJ.nParticles
        Zin = SCgenBunches(SC, nParticles=nParticles)
    if nTurns is None:
        nTurns = SC.INJ.nTurns

    T = atpass(SC.RING, Zin, nTurns, ord, keep_lattice=False)
    T[:, np.isnan(T[0, :])] = np.nan
    labelStr = ['$\Delta x$ [$\mu$m]', '$\Delta x''$ [$\mu$rad]', '$\Delta y$ [$\mu$m]', '$\Delta y''$ [$\mu$rad]',
                '$\Delta S$ [m]', '$\delta E$ $[\%]$']
    titleStr = ['Horizontal', 'Vertical', 'Longitudinal']
    if SC.RING[SC.ORD.RF[0]].PassMethod == 'RFCavityPass':
        L0_tot = 0
        for i in range(len(SC.RING)):
            L0_tot = L0_tot + SC.RING[i].Length
        lengthSlippage = SPEED_OF_LIGHT * (SC.RING[SC.ORD.RF[0]].HarmNumber / SC.RING[SC.ORD.RF[0]].Frequency - L0_tot / SPEED_OF_LIGHT)
        T[5, :, :, :] = T[5, :, :, :] - lengthSlippage * np.arange(nTurns)[np.newaxis, np.newaxis, :]
        labelStr[4] = '$\Delta S_{act}$ [m]'
    if plotCO:
        _, CO = findorbit6(SC.RING, ord)
        if np.isnan(CO[0, 0]):
            startPointGuess = np.nanmean(T, axis=(1, 2, 3))
            _, CO = findorbit6(SC.RING, ord, startPointGuess)
            if np.isnan(CO[0, 0]):
                CO = np.full(6, np.nan)
    else:
        CO = np.full(6, np.nan)
    T = T * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])[:, np.newaxis, np.newaxis, np.newaxis]
    Z0 = SC.INJ.Z0 * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    CO = CO * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    T[[4, 5], :, :, :] = T[[5, 4], :, :, :]
    CO[[4, 5]] = CO[[5, 4]]
    Z0[[4, 5]] = Z0[[5, 4]]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18.5, 10.5), dpi=100, facecolor="w")
    pVec = []
    legStr = []
    for nType in range(3):
        for nP in range(nParticles):
            x = T[2 * nType, nP, :, :]
            y = T[2 * nType + 1, nP, :, :]
            ax[nType].scatter(x, y, 10, np.arange(nTurns))
        pVec.append(ax[nType].plot(Z0[2 * nType], Z0[2 * nType + 1], 'o'))
        legStr.append('Injection point')
        if plotCO:
            pVec.append(ax[nType].plot(CO[2 * nType], CO[2 * nType + 1], 'x', MarkerSize=20, LineWidth=3))
            legStr.append('Closed orbit')
        # ax[nType].set_box('on')
        ax[nType].set_xlabel(labelStr[2 * nType])
        ax[nType].set_ylabel(labelStr[2 * nType + 1])
        plt.title(titleStr[nType] + ' @Ord: ' + str(ord))

    # plt.legend(pVec, legStr)
    # plt.colorbar()
    # c.set_label('Number of turns')
    plt.show()
