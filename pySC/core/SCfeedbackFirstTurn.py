import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints


def SCfeedbackFirstTurn(SC, Mplus, R0=None, maxsteps=100, wiggleAfter=20, wiggleSteps=64, wiggleRange=[50E-6, 200E-6],
                        CMords=None, BPMords=None, verbose=0):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if verbose: print('SCfeedbackFirstTurn: Start')
    BPMhist = -1 * np.ones(1, 100)
    ERROR = 1
    nWiggleCM = 1
    for n in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject...
        if np.all(np.isnan(B)):
            if verbose: print('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
            ERROR = 2
            return
        correctionStep()  # call correction subroutine.
        if isRepro(BPMhist, 5) and isTransmit(BPMhist):
            if verbose: print('SCfeedbackFirstTurn: Success')
            ERROR = 0
            return
        elif isRepro(BPMhist, wiggleAfter):
            if verbose: print('SCfeedbackFirstTurn: Wiggling')
            CMidxsH = getLastCMsDim(par, B, 1, nWiggleCM)  # Last CMs in horz
            CMidxsV = getLastCMsDim(par, B, 2, nWiggleCM)  # Last CMs in vert
            CMordsH = CMords[0][CMidxsH]
            CMordsV = CMords[1][CMidxsV]
            pts = np.array([[0, 0], [0, 0]])
            pts = np.append(pts, goldenDonut(wiggleRange[0], wiggleRange[1], wiggleSteps), axis=1)
            dpts = np.diff(pts, 1, 1)
            for i in range(dpts.shape[1]):
                SPH = dpts[0, i] * np.ones((len(CMordsH), 1))  # Horizontal setpoint change
                SPV = dpts[1, i] * np.ones((len(CMordsV), 1))  # Vertical setpoint change
                SC = SCsetCMs2SetPoints(SC, CMordsH, SPH, 1, 'add')
                SC = SCsetCMs2SetPoints(SC, CMordsV, SPV, 2, 'add')
                W = SCgetBPMreading(SC, BPMords=BPMords)
                BPMhist = logLastBPM(BPMhist, W)
                if isNew(BPMhist):
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = logLastBPM(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    if isRepro(BPMhist, 3):
                        BPMhist[0:3] = -1  # void last hist
                        nWiggleCM = 0  # Reset Wiggler CM number
                        break  # Continue with feedback
            B = SCgetBPMreading(SC, BPMords=BPMords)
            correctionStep()
            B = SCgetBPMreading(SC, BPMords=BPMords)
            correctionStep()
            B = SCgetBPMreading(SC, BPMords=BPMords)
            correctionStep()
            nWiggleCM = nWiggleCM + 1
    if verbose: print('SCfeedbackFirstTurn: FAIL (maxsteps reached)')
    ERROR = 1
    return


def correctionStep():
    BPMhist = logLastBPM(BPMhist, B)
    R = np.array([B[0, :], B[1, :]])
    dR = R - R0
    dR[np.isnan(dR)] = 0
    dphi = Mplus @ (dR)
    lastCMh = getLastCMsDim(par, B, 1, 1)
    lastCMv = getLastCMsDim(par, B, 2, 1)
    dphi[lastCMh + 1:len(CMords[0])] = 0
    dphi[len(CMords[0]) + lastCMv:len(CMords[0]) + len(CMords[1])] = 0
    SC = SCsetCMs2SetPoints(SC, CMords[0], -dphi[0:len(CMords[0])], 1, 'add')
    SC = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):len(CMords[0]) + len(CMords[1])], 2, 'add')


def goldenDonut(r0, r1, Npts):
    out = np.zeros((2, Npts))  # initialize output array
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    theta = 0
    for n in range(Npts):
        out[:, n] = np.sqrt((r1 ** 2 - r0 ** 2) * n / (Npts - 1) + r0 ** 2) * np.array([np.cos(theta), np.sin(theta)])
        theta = theta + phi
    return out


def getLastCMsDim(par, B, dim, n):
    lastBPMidx = np.where(~np.isnan(B))[1][-1]
    if len(lastBPMidx) == 0:
        lastBPMidx = len(BPMords)  # the last one
    lastBPMord = BPMords[lastBPMidx]
    idxs = np.where(CMords[dim] <= lastBPMord)[0][-n:]
    return idxs


def logLastBPM(BPMhist, B):
    BPMhist = np.roll(BPMhist, 1)
    ord = getLastBPMord(B)
    if ord:
        BPMhist[0] = ord
    else:
        BPMhist[0] = 0
    return BPMhist


def getLastBPMord(B):
    ord = np.where(np.isnan(B))[1][0] - 1
    return ord


def isRepro(BPMhist, N):
    res = np.all(BPMhist[0:N] == BPMhist[0])
    return res


def isTransmit(BPMhist):
    res = BPMhist[0] == 0
    return res


def isNew(BPMhist):
    res = BPMhist[0] != BPMhist[1]
    return res
