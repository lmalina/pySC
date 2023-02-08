import numpy as np
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints

def SCfeedbackBalance(SC,Mplus,eps=1e-5, R0=np.zeros((2, 1)), maxsteps=10, CMords=None, BPMords=None, verbose=False):
    if CMords is None:
        CMords = SC.ORD.CM[:]
    if BPMords is None:
        BPMords = SC.ORD.CM[:]
    if verbose:
        print('SCfeedbackBalance: Start')
    BPMindHist = -1 * np.ones(100)
    BRMShist = np.nan*np.ones(100)
    cnt=1
    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords) # Inject ...
        BPMindHist = logLastBPM(BPMindHist, B)
        lBPM = B.shape[1]
        Bx1 = B[0, 0:lBPM // 2]
        By1 = B[1, 0:lBPM // 2]
        Bx2 = B[0, lBPM // 2:]
        By2 = B[1, lBPM // 2:]
        DELTABx = Bx2 - Bx1
        DELTABy = By2 - By1
        R = np.vstack((Bx1 - R0[0, :], DELTABx, By1 - R0[1, :], DELTABy)).T
        R[np.isnan(R)] = 0
        dphi = Mplus @ R
        BRMShist = np.roll(BRMShist, 1)
        BRMShist[0] = np.sqrt(np.var(R, 1))
        SC = SCsetCMs2SetPoints(SC, CMords[0], -dphi[0:len(CMords[0])], 1, 'add')
        SC = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]) + 1:end], 2, 'add')
        if isSetback(BPMindHist):
            if verbose:
                print('SCfeedbackBalance: FAIL (setback)')
            return
        if not isTransmit(BPMindHist):
            if verbose:
                print('SCfeedbackBalance: FAIL (lost transmission)')
            return
        if isConverged(BRMShist,3,eps):
            if verbose:
                print('SCfeedbackBalance: Success (converged after %d steps)'%steps)
            return
        cnt = cnt+1
    if isStable(min(10, maxsteps), eps):
        if verbose:
            print('SCfeedbackBalance: Success (maxsteps reached)')
        return
    else:
        if verbose:
            print('SCfeedbackBalance: FAIL (maxsteps reached, unstable)')
        return


def isSetback(hist):
    return hist[0]<hist[1]


def isTransmit(hist):
    return hist[0]==0


def logLastBPM(hist,B):
    hist = np.roll(hist,1)
    ord = getLastBPMord(B)
    if ord:
        hist[0]=ord
    else:
        hist[0]=0
    return hist


def getLastBPMord(B):
    ord = np.where(np.isnan(B))[1]
    if len(ord)>0:
        return ord[0]-1
    else:
        return None


def isConverged(hist,n,eps):
    CV=np.var(hist[0:n],1)/np.std(hist[0:n])
    return CV<eps


def isStable(n,eps):
    CV=np.var(BRMShist[0:n],1)/np.std(BRMShist[0:n])
    return CV<eps

# End
 
