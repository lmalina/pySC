import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints


def SCgetDispersion(SC,RFstep,BPMords=None,CAVords=None,nSteps=2):
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if CAVords is None:
        CAVords = SC.ORD.RF
    RFsteps = np.zeros((len(CAVords),nSteps))
    for nCav in range(len(CAVords)):
        RFsteps[nCav,:] = SC.RING[CAVords[nCav]].FrequencySetPoint + np.linspace(-RFstep,RFstep,nSteps)
    Bref = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1)
    if nSteps==2:
        SC = SCsetCavs2SetPoints(SC,CAVords,'Frequency',RFstep,'add')
        B = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1)
        eta = (B-Bref)/RFstep
    else:
        dB = np.zeros((nSteps,*np.shape(Bref)))
        for nStep in range(nSteps):
            SC = SCsetCavs2SetPoints(SC,CAVords,'Frequency',RFsteps[:,nStep],'abs')
            dB[nStep,:] = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1) - Bref
        eta = np.linalg.lstsq(np.linspace(-RFstep,RFstep,nSteps),dB)[0]
    return eta
