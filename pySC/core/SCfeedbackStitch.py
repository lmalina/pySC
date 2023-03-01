import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCMs2SetPoints
from pySC.utils.feedback import isRepro, isTransmit, logLastBPM, goldenDonut


def SCfeedbackStitch(SC, Mplus, R0=np.zeros((2, 1)), nBPMs=4, maxsteps=30, nRepro=3, CMords=None, BPMords=None,
                     verbose=False):
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    BPMhist = -1 * np.ones(1, 100)
    B = SCgetBPMreading(SC, BPMords=BPMords)
    if not isSignal(B, nBPMs):
        if verbose:
            print('SCfeedbackStitch: Wiggling')
        pts = np.hstack((np.aray([[0], [0]]), goldenDonut(50E-6, 200E-6, 32)))
        dpts = np.diff(pts, axis=1)
        for nWiggleCM in range(1, 9):
            if verbose:
                print(f'SCfeedbackStitch: Number of magnets used for wiggling: {nWiggleCM}. \n')
            CMords = getLastCMords(B, nWiggleCM, CMords, BPMords)
            for i in range(len(dpts[0])):
                for ord in CMords:
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[0][i], 1, method='add')
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[1][i], 2, method='add')
                W = SCgetBPMreading(SC, BPMords=BPMords)
                BPMhist = logLastBPM(BPMhist, W)
                if isSignal(W, nBPMs):  # TODO double check. Seems a bit iffy
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    if isRepro(BPMhist, 3):
                        BPMhist[0:3] = -1  # void last hist
                        break
    B = SCgetBPMreading(SC, BPMords=BPMords)
    if not isSignal(B, nBPMs):
        raise RuntimeError('SCfeedbackStitch: FAIL Wiggling failed')

    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject...
        BPMhist = logLastBPM(BPMhist, B)
        lBPM = len(B[0])
        Bx1 = B[0][0:lBPM / 2]
        By1 = B[1][0:lBPM / 2]
        Bx2 = B[0][(lBPM / 2 + 1):]
        By2 = B[1][(lBPM / 2 + 1):]
        DELTABx = Bx2 - Bx1
        DELTABy = By2 - By1
        DELTABx[(nBPMs + 1):] = 0
        DELTABy[(nBPMs + 1):] = 0
        R = [Bx1 - R0[0], DELTABx, By1 - R0[1], DELTABy]
        R[np.isnan(R)] = 0
        dphi = Mplus * R
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], 1, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], 2,
                                   method='add')  # call correction subroutine.
        if isSetback(BPMhist):
            RuntimeError('SCfeedbackStitch: FAIL Setback')
        if isRepro(BPMhist, nRepro) and isTransmit(BPMhist):
            if verbose:
                print('SCfeedbackStitch: Success')
            return SC
    raise RuntimeError('SCfeedbackStitch: FAIL Reached maxsteps')


def getLastCMords(B, n, CMords, BPMords):
    dualCMords = np.intersect1d(CMords[0], CMords[1])  # Generate a list of CMs that can act in both planes.
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1]  # Get index of last reached BPM
    if len(lastBPMidx) == 0 or lastBPMidx > len(BPMords):  # If there is no beam loss in the first turn
        ords = dualCMords[-n:]  # ... just return the last n CMs
    else:  # in case the beam was lost in the first turn
        lastBPMord = BPMords[lastBPMidx]  # We can then find the ordinate of the last BPM.
        lastCMidx = np.where(dualCMords <= lastBPMord)[0][-1]  # Find the last CM upstream of the last BPM.
        ords = dualCMords[(lastCMidx - min(lastCMidx, n) + 1):lastCMidx]
    return ords

def isSetback(BPMhist):
    return BPMhist[0] != 0 and BPMhist[0] < BPMhist[2] and BPMhist[1] < BPMhist[2]

def isSignal(B, nBPMs):
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1]
    return lastBPMidx >= len(B[0]) / 2 + nBPMs
