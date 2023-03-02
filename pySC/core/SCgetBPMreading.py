import numpy as np
from matplotlib import pyplot as plt

from pySC.classes import DotDict
from pySC.core.SCgenBunches import SCgenBunches
from pySC.utils.sc_tools import SCrandnc
from pySC.at_wrapper import atgetfieldvalues, atpass, findorbit6, findspos


def SCgetBPMreading(SC, BPMords=[], plotFunctionFlag=False):
    """
    lattice_pass
        r_out: (6, N, R, T) array containing output coordinates of N particles
          at R reference points for T turns.

    findorbit6
        orbit0:         (6,) closed orbit vector at the entrance of the
                        1-st element (x,px,y,py,dp,0)
        orbit:          (Nrefs, 6) closed orbit vector at each location
                        specified in ``refpts``
    """
    B1 = np.full((2, SC.INJ.nTurns * len(SC.ORD.BPM), SC.INJ.nShots), np.nan)
    for nShot in range(SC.INJ.nShots):
        if SC.INJ.trackMode == 'ORB':
            _ , TT = findorbit6(SC.RING, SC.ORD.BPM, keep_lattice=False)
        else:
            T = atpass(SC.RING, SCgenBunches(SC), SC.INJ.nTurns, SC.ORD.BPM, keep_lattice=False)
        B1[:, :, nShot] = calcBPMreading(SC, T, at_all_elements=plotFunctionFlag)

    B = np.nanmean(B1, 2)
    if plotFunctionFlag:
        B1 = np.full((2, SC.INJ.nTurns * len(SC.ORD.BPM), SC.INJ.nShots), np.nan)
        T1 = np.full((6, SC.INJ.nTurns * SC.INJ.nParticles * len(SC.RING), SC.INJ.nShots), np.nan)
        refOrds = np.arange(len(SC.RING))
        for nShot in range(SC.INJ.nShots):
            if SC.INJ.trackMode == 'ORB':
                T = findorbit6(SC.RING, refOrds)
            else:
                T = atpass(SC.RING, SCgenBunches(SC), SC.INJ.nTurns, refOrds, keep_lattice=False)
            T[:, np.isnan(T[0, :])] = np.nan
            B1[:, :, nShot] = calcBPMreading(SC, T, at_all_elements=plotFunctionFlag)
            T1[:, :, nShot] = T
        B = np.nanmean(B1, axis=2)
        _plot_bpm_reading(SC, B, T1)

    if SC.INJ.trackMode == 'PORB':   # ORB averaged over low amount of turns
        Bpseudo = np.full((2, len(SC.ORD.BPM)), np.nan)
        for nBPM in range(len(SC.ORD.BPM)):
            Bpseudo[:, nBPM] = np.nanmean(B[:, nBPM::len(SC.ORD.BPM)], 1)
        B = Bpseudo
    if len(BPMords) > 0:
        ind = np.where(np.isin(SC.ORD.BPM, BPMords))[0]
        if len(ind) != len(BPMords):
            print('Not all specified ordinates are registered BPMs.')
        if SC.INJ.trackMode == 'TBT':
            ind = np.arange(SC.INJ.nTurns) * len(SC.ORD.BPM) + ind
        B = B[:, ind]
    return B


def calcBPMreading(SC, T, nTurns, nParticles, at_all_elements=False):
    bpm_noise = np.array(atgetfieldvalues(SC.RING, SC.ORD.BPM, ('NoiseCO' if SC.INJ.trackMode == 'ORB' else "Noise")))
    # TODO for later 4D matrices and here no repetition
    bpm_offset = atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Offset') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportOffset')
    bpm_cal_error = atgetfieldvalues(SC.RING, SC.ORD.BPM, 'CalError')
    bpm_roll = atgetfieldvalues(SC.RING, SC.ORD.BPM, 'Roll') + atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SupportRoll')
    bpm_noise = bpm_noise * SCrandnc(2, (nTurns * len(SC.ORD.BPM), 2))  #  TODO check the order of dimensions
    bpm_sum_error = np.repeat(atgetfieldvalues(SC.RING, SC.ORD.BPM, 'SumError'), nTurns)
    if at_all_elements:
        Tx = T[0, :, SC.ORD.BPM, :]
        Ty = T[2, :, SC.ORD.BPM, :]
    else:
        Tx = T[0, :, :, :]
        Ty = T[2, :, :, :]
    # averaging over particles
    Bx1 = np.nanmean(Tx, axis=0)
    By1 = np.nanmean(Ty, axis=0)  # here it should still be 3D matrix
    beamLost = np.nonzero(np.sum(np.isnan(Tx), axis=1) * (1 + bpm_sum_error * SCrandnc(2, bpm_sum_error.shape)) > (
                nParticles * SC.INJ.beamLostAt))
    Bx1[beamLost] = np.nan
    By1[beamLost] = np.nan

    Bx = np.cos(bpm_roll) * Bx1 - np.sin(bpm_roll) * By1
    By = np.sin(bpm_roll) * Bx1 + np.cos(bpm_roll) * By1
    Bx = (Bx - bpm_offset[:, 0]) * (1 + bpm_cal_error[:, 0])
    By = (By - bpm_offset[:, 1]) * (1 + bpm_cal_error[:, 1])
    Bx = Bx + bpm_noise[0, :]
    By = By + bpm_noise[1, :]
    B = np.array([Bx, By])
    return B


def _plot_bpm_reading(SC, B, T):  # T is 5D matrix
    ApertureForPLotting = getRingAperture(SC)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=100, facecolor="w")

    tmpCol = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ylabelStr = ['$\Delta x$ [mm]', '$\Delta y$ [mm]']
    legStr = ['Particle trajectories', 'BPM reading', 'Aperture']
    sPos = findspos(SC.RING)
    sMax = sPos[-1]
    for nDim in range(2):
        x = np.reshape(np.arange(SC.INJ.nTurns) * sMax + sPos, 1, [])  # TODO nturns x bpms array of s positions
        x = np.repeat(x, SC.INJ.nParticles)
        for nS in range(SC.INJ.nShots):
            y = 1E3 * T[2 * nDim, :, :, :, nS]
            legVec = ax[nDim].plot(x, y, 'k')
        legVec = legVec[0]
        x = np.reshape(np.arange(SC.INJ.nTurns) * sMax + sPos[SC.ORD.BPM.keys()], 1,
                       [])  # TODO nturns x bpms array of s positions
        y = 1E3 * (B[nDim, :])
        legVec[1] = ax[nDim].plot(x, y, 'rO')
        if ApertureForPLotting is not None:
            apS = sPos[ApertureForPLotting.apOrds]
            x = np.reshape(np.arange(SC.INJ.nTurns) * sMax + apS, 1, [])
            y = 1E3 * np.repeat(ApertureForPLotting.apVals[nDim], SC.INJ.nTurns) # to mm
            legVec[2] = ax[nDim].plot(x, y[0, :], '-', color=tmpCol[0], linewidth=4)
            ax[nDim].plot(x, y[1, :], '-', color=tmpCol[0], linewidth=4)
        x = np.reshape(np.arange(SC.INJ.nTurns) * sMax, 1, [])
        for nT in range(SC.INJ.nTurns):
            ax[nDim].plot(x[nT] * [1, 1], 10 * [-1, 1], 'k:')
        ax[nDim].set_xlim([0, SC.INJ.nTurns * sPos[-1]])
        ax[nDim].set_box('on')  # True?
        ax[nDim].set_position(ax[nDim].get_position() + np.array([0, .07 * (nDim - 1), 0, 0]))
        ax[nDim].set_ylim([-0.5, 0.5])  # TODO  to config
        ax[nDim].set_ylabel(ylabelStr[nDim])
        ax[nDim].legend(legVec, legStr[0:len(legVec)])
    ax[nDim].set_xlabel('$s$ [m]')
    for item in ([ax[nDim].title, ax[nDim].xaxis.label, ax[nDim].yaxis.label] +
                 ax[nDim].get_xticklabels() + ax[nDim].get_yticklabels()):
        item.set_fontsize(18)
        item.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
    fig.set_facecolor('w')
    plt.show()
    # plt.pause(0.001)
    # plt.tight_layout()
    # plt.gcf().canvas.draw()
    # plt.gcf().canvas.flush_events()


def getRingAperture(SC):
    ap_ord = DotDict()
    for ord in range(len(SC.RING)):
        if hasattr(SC.RING[ord], 'EApertures') or hasattr(SC.RING[ord], 'RApertures'):
            ap_ord[ord] = (np.outer(getattr(SC.RING[ord], 'EApertures') * np.array([-1, 1]))
                           if hasattr(SC.RING[ord], 'EApertures')
                           else np.outer(
                getattr(SC.RING[ord], 'RApertures') * np.array([-1, 1])))  # TODO RApertures[2 * (nDim - 1) + [1, 2]]

    # ApertureForPLotting = [] if len(ap_ord) == 0 else {'apOrds': apOrds, 'apVals': apVals}
    return ap_ord
