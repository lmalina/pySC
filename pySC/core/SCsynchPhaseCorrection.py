import numpy as np
from scipy.optimize import root, minimize

from pySC.at_wrapper import findorbit6
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints


# TODO maybe translate again

def SCsynchPhaseCorrection(SC, cavOrd=None, nSteps=15, nTurns=20, plotResults=0, plotProgress=0, verbose=0):
    if cavOrd is None:
        cavOrd = SC.ORD.Cavity
    ERROR = 0
    deltaPhi = 0
    BPMshift = np.nan(1, nSteps)
    lamb = 299792458 / SC.RING[cavOrd].Frequency
    lambTestVec = 1 / 2 * lamb * np.linspace(-1, 1, nSteps)
    SC.INJ.nTurns = nTurns
    if verbose:
        print('Calibrate RF phase with: \n %d Particles \n %d Turns \n %d Shots \n %d Phase steps.\n\n' % (
            SC.INJ.nParticles, SC.INJ.nTurns, SC.INJ.nShots, nSteps))
    for nL in range(len(lambTestVec)):
        tmpSC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', lambTestVec[nL], 'add')
        BPMshift[nL], TBTdE = getTbTEnergyShift(tmpSC)
        if plotProgress:
            pass  # plotProgress(TBTdE, BPMshift, lambTestVec, nL)
    if not (max(BPMshift) > 0 > min(BPMshift)):
        print('Zero crossing not within data set.\n')
        ERROR = 1
        return deltaPhi, ERROR

    def sinFun(par, s):
        return par[0] * np.sin(2 * np.pi * (par[3] * s + par[1])) + par[2]

    def fomFun(par):
        return sum((sinFun(par, lambTestVec) - BPMshift) ** 2)

    for startPhase in [-np.pi, -np.pi / 2, -np.pi / 4, 0]:
        sol = minimize(fomFun, np.array(
            [max(BPMshift) - np.mean(BPMshift), startPhase, np.mean(BPMshift), 0.5 / max(lambTestVec)]))

        def solFun(x):
            return sinFun(sol, x)

        xp = np.linspace(lambTestVec[0], lambTestVec[-1], 100)
        if not (max(solFun(xp)) > 0 > min(solFun(xp))):
            print('Zero crossing not within fit function, trying different start point guess.\n')
        else:
            break
    if not (max(solFun(xp)) > 0 > min(solFun(xp))):
        print('Zero crossing not within fit function\n')
        ERROR = 2
    maxValInd = np.argmax(solFun(xp))
    deltaPhi = root(solFun, xp[maxValInd] - abs(lambTestVec[0]) / 2)  # TODO root
    if deltaPhi > lamb / 2:
        deltaPhi = deltaPhi - lamb
    elif deltaPhi < - lamb / 2:
        deltaPhi = deltaPhi + lamb
    if plotResults:
        pass  # plotFunction() TODO
    if np.isnan(deltaPhi):
        ERROR = 3
        print('SCrfCommissioning: ERROR (NaN phase)\n')
        return deltaPhi, ERROR
    if verbose:
        XCO = findorbit6(SC.RING)
        tmpSC = SCsetCavs2SetPoints(SC, cavOrd, 'TimeLag', deltaPhi, 'add')
        XCOfinal = findorbit6(tmpSC.RING)
        initial = np.remainder((XCO[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        final = np.remainder((XCOfinal[5] - SC.INJ.Z0[5]) / lamb * 360, 360)
        print(f'Phase correction step: {deltaPhi}.3f\n')
        print('>> Static phase error corrected from %.0fdeg to %.1fdeg\n' % (initial, final))
        return deltaPhi, ERROR


def getTbTEnergyShift(SC):
    B = SCgetBPMreading(SC)
    BB = np.reshape(B[0, :], [], SC.INJ.nTurns)
    dE = np.mean(BB - np.repmat(BB[:, 0], 1, SC.INJ.nTurns))
    x = np.arange(SC.INJ.nTurns)
    y = dE[1:]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    BPMshift = np.linalg.lstsq(x, y)[0]
    return BPMshift, dE

# def plotProgress(TBTdE, BPMshift, fTestVec, nE):
#     plt.figure(2);
#     plt.clf()
#     plt.subplot(2, 1, 1);
#     plt.hold(True)
#     plt.plot(TBTdE, 'o')
#     plt.plot(np.arange(len(TBTdE)) *BPMshift[nE], '--')
#     plt.xlabel('Number of turns');
#     plt.ylabel('$<\Delta x_\mathrm{TBT}>$ [m]');
#     plt.subplot(2, 1, 2);
#     plt.plot(fTestVec[0:nE], BPMshift[0:nE], 'o')
#     plt.xlabel('$\Delta \phi$ [m]');
#     plt.ylabel('$<\Delta x>$ [m/turn]');
#     plt.setp(plt.findall(plt.gcf(), '-property', 'FontSize'), 'FontSize', 18);
#     plt.setp(plt.findall(plt.gcf(), '-property', 'Interpreter'), 'Interpreter', 'latex');
#     plt.setp(plt.findall(plt.gcf(), '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
#     plt.gcf().set_color('w');
#     plt.draw()
#     plt.draw()
#
#
# def plotFunction():
#     plt.figure(87)
#     plt.clf()
#     plt.hold(True)
#     plt.plot((lambTestVec + SC.INJ.Z0[5]) / lamb * 360, 1E6 * BPMshift, 'o')
#     plt.plot((xp+SC.INJ.Z0[5]) / lamb * 360, 1E6 * solFun(xp))
#     plt.plot((SC.INJ.Z0[5]) / lamb * 2 * 180, 1E6 * (sol[0] * np.sin(2 * np.pi * (sol[3] * 0 + sol[1])) +sol[2]), 'rD', 'MarkerSize', 16)
#     plt.plot((deltaPhi) / lamb * 2 * 180, 0, 'kX', 'MarkerSize', 16)
#     plt.setp(plt.gca(), 'XLim', 180 *[-1 1], 'box', 'on');
#     plt.legend({'Measurement', 'Fit', 'Initial phase', 'Phase correction'})
#     plt.xlabel('RF phase [$^\circ$]');plt.ylabel('$<\Delta x>$ [$\mu$m/turn]');
#     plt.setp(plt.findall(plt.gcf(), '-property', 'FontSize'), 'FontSize', 18);
#     plt.setp(plt.findall(plt.gcf(), '-property', 'Interpreter'), 'Interpreter', 'latex');
#     plt.setp(plt.findall(plt.gcf(), '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
#     plt.gcf().set_color('w');plt.draw()
#
#            def inputCheck(SC, par):
#     if SC.INJ.trackMode != 'TBT':
#         raise ValueError('Trackmode should be turn-by-turn (''SC.INJ.trackmode=TBT'').')
#     if nSteps < 2 or nTurns < 2:
#         raise ValueError('Number of steps and number of turns must be larger than 2.')
#     if not hasattr(SC.RING[cavOrd], 'Frequency'):
#         raise ValueError('This is not a cavity (ord: %d)' % cavOrd)
#     if not any(SC.RING[cavOrd].PassMethod in ['CavityPass', 'RFCavityPass']):
#         print('Cavity (ord: %d) seemed to be switched off.' % cavOrd)
#
# # End
