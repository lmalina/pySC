import numpy as np
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints
from pySC.core.SCsetCavs2SetPoints import SCsetCavs2SetPoints
from pySC.utils.feedback import is_stable_or_converged


def SCfeedbackRun(SC, Mplus, R0=None, eps=1e-5, target=0, maxsteps=30, scaleDisp=0, CMords=None, BPMords=None,
                  weight=None, verbose=False, plotFunctionFlag=False):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if weight is None:
        weight = np.ones((Mplus.shape[1], 1))
    if verbose:
        print('SCfeedbackRun: Start')
    BPMhist = np.nan * np.ones((1, 100))
    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords, plotFunctionFlag=plotFunctionFlag)  # Inject ...
        R = np.array([B[0, :], B[1, :]])
        R[np.isnan(R)] = 0
        dphi = Mplus @ ((R - R0) * weight)
        if scaleDisp != 0:
            SC = SCsetCavs2SetPoints(SC, SC.ORD.Cavity, -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], 1, method="add")
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], 2, method="add")
        BPMhist = np.roll(BPMhist, 1)
        BPMhist[0] = np.sqrt(np.mean(R ** 2, 1))
        if np.any(np.isnan(B[0, :])):
            raise RuntimeError('SCfeedbackRun: FAIL (lost transmission)')

        if BPMhist[0] < target and is_stable_or_converged(min(10, maxsteps), eps, BPMhist):
            if verbose:
                print("SCfeedbackRun: Success (target reached)")
            return SC
        if is_stable_or_converged(3, eps, BPMhist):
            if verbose:
                print(f"SCfeedbackRun: Success (converged after {steps:d} steps)")
            return SC

    if is_stable_or_converged(min(10, maxsteps), eps, BPMhist) or maxsteps == 1:
        if verbose:
            print("SCfeedbackRun: Success (maxsteps reached)")
        return SC
    raise RuntimeError("SCfeedbackRun: FAIL (maxsteps reached, unstable)")
