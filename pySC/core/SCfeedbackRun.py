import numpy as np
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints
from pySC.core.SCsetCavs2SetPoints import SCsetCavs2SetPoints


def SCfeedbackRun(SC,Mplus,R0=None,eps=1e-5,target=0,maxsteps=30,scaleDisp=0,CMords=None,BPMords=None,weight=None,verbose=0):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1],1))
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if weight is None:
        weight = np.ones((Mplus.shape[1],1))
    if verbose: print('SCfeedbackRun: Start')
    ERROR = 1
    BPMhist = np.nan*np.ones((1,100))
    cnt=1
    for steps in range(maxsteps):
        B = SCgetBPMreading(SC,BPMords=BPMords) # Inject ...
        R = np.array([B[0, :], B[1, :]])
        R[np.isnan(R)] = 0
        dphi = Mplus @ ((R - R0) * weight)
        if scaleDisp != 0:
            SC = SCsetCavs2SetPoints(SC, SC.ORD.Cavity, -scaleDisp * dphi[-1], add=True)
            dphi = dphi[:-1]
        SC = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], 1, add=True)
        SC = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], 2, add=True)
        BPMhist = np.roll(BPMhist, 1)
        BPMhist[0] = np.sqrt(np.mean(R ** 2, 1))
        ERROR = 0
        if np.any(np.isnan(B[0,:])):
            if verbose: print('SCfeedbackRun: FAIL (lost transmission)')
            ERROR = 2
            return
        if BPMhist[0]<target and isStable(min(10,maxsteps),eps):
            if verbose: print('SCfeedbackRun: Success (target reached)')
            ERROR = 0
            return
        if isConverged(3,eps):
            if verbose: print('SCfeedbackRun: Success (converged after %d steps)'%steps)
            ERROR = 0
            return
        cnt = cnt+1
    if isStable(min(10,maxsteps),eps) or maxsteps==1:
        if verbose: print('SCfeedbackRun: Success (maxsteps reached)')
        ERROR=0
        return
    else:
        if verbose: print('SCfeedbackRun: FAIL (maxsteps reached, unstable)')
        ERROR=1
        return

def isStable(n,eps):
    CV=np.var(BPMhist[:n],1)/np.std(BPMhist[:n])
    return CV<eps
def isConverged(n,eps):
    CV=np.var(BPMhist[:n],1)/np.std(BPMhist[:n])
    return CV<eps
 

# Test

#SCfeedbackRun(SC,Mplus,R0=np.zeros((Mplus.shape[1],1)),eps=1e-5,target=0,maxsteps=30,scaleDisp=0,CMords=None,BPMords=None,weight=None,verbose=1)

