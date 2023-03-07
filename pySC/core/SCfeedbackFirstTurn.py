import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCMs2SetPoints
from pySC.utils.feedback import isNew, isRepro, isTransmit, logLastBPM, goldenDonut


def SCfeedbackFirstTurn(SC, Mplus, R0=None, maxsteps=100, wiggleAfter=20, wiggleSteps=64, wiggleRange=np.array([50E-6, 200E-6]),
                        CMords=None, BPMords=None, verbose=0):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if verbose:
        print('SCfeedbackFirstTurn: Start')
    BPMhist = -1 * np.ones((1, 100))
    nWiggleCM = 1
    for n in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject...
        if np.all(np.isnan(B)):
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        SC, BPMhist = correctionStep(SC, BPMhist, BPMords, CMords, B, R0, Mplus)  # call correction subroutine.
        if isRepro(BPMhist, 5) and isTransmit(BPMhist):
            if verbose:
                print('SCfeedbackFirstTurn: Success')
            return SC
        elif isRepro(BPMhist, wiggleAfter):
            if verbose:
                print('SCfeedbackFirstTurn: Wiggling')
            CMidxsH = getLastCMsDim(B, nWiggleCM, BPMords, CMords[0])  # Last CMs in horz
            CMidxsV = getLastCMsDim(B, nWiggleCM, BPMords, CMords[1])  # Last CMs in vert
            CMordsH = CMords[0][CMidxsH]
            CMordsV = CMords[1][CMidxsV]
            pts = np.array([[0, 0], [0, 0]])
            pts = np.append(pts, goldenDonut(wiggleRange[0], wiggleRange[1], wiggleSteps), axis=1)
            dpts = np.diff(pts, 1, 1)
            for i in range(dpts.shape[1]):
                SPH = dpts[0, i] * np.ones((len(CMordsH), 1))  # Horizontal setpoint change
                SPV = dpts[1, i] * np.ones((len(CMordsV), 1))  # Vertical setpoint change
                SC, _ = SCsetCMs2SetPoints(SC, CMordsH, SPH, 1, method='add')
                SC, _ = SCsetCMs2SetPoints(SC, CMordsV, SPV, 2, method='add')
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
            SC, BPMhist = correctionStep(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            B = SCgetBPMreading(SC, BPMords=BPMords)
            SC, BPMhist = correctionStep(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            B = SCgetBPMreading(SC, BPMords=BPMords)
            SC, BPMhist = correctionStep(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            nWiggleCM = nWiggleCM + 1
    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')

def correctionStep(SC, BPMhist, BPMords, CMords, B, R0, Mplus):
    BPMhist = logLastBPM(BPMhist, B)
    R = B[:, :, 0].reshape(R0.shape)
    dR = R - R0
    dR[np.isnan(dR)] = 0
    dphi = Mplus @ dR
    lastCMh = getLastCMsDim(B, 1, BPMords, CMords[0])
    lastCMv = getLastCMsDim(B, 1, BPMords, CMords[1])
    dphi[lastCMh + 1:len(CMords[0])] = 0
    dphi[len(CMords[0]) + lastCMv:len(CMords[0]) + len(CMords[1])] = 0
    SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[len(CMords[0])], 1, method='add')
    SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):len(CMords[0]) + len(CMords[1])], 2, method='add')
    return SC, BPMhist

def getLastCMsDim(B, n, BPMords, CMords):
    BPMinds = np.where(~np.isnan(B))[1]
    if len(BPMinds) == 0:
        lastBPMidx = len(BPMords)  # the last one
    else:
        lastBPMidx = BPMinds[-1]
    lastBPMord = BPMords[lastBPMidx]
    idxs = np.where(CMords <= lastBPMord)[0][-n:]
    return idxs
