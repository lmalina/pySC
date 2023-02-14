import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetCMs2SetPoints import SCsetCMs2SetPoints


def SCfeedbackStitch(SC,Mplus,R0=np.zeros((2,1)),nBPMs=4,maxsteps=30,CMords=SC.ORD.CM,BPMords=SC.ORD.BPM,verbose=0,nRepro=3):
    ERROR = 1
    BPMhist = -1*np.ones(1,100)
    B = SCgetBPMreading(SC,BPMords=BPMords)
    if not isSignal(B,nBPMs):
        if verbose: print('SCfeedbackStitch: Wiggling')
        wiggle()
    B = SCgetBPMreading(SC,BPMords=BPMords)
    if not isSignal(B,nBPMs):
        if verbose: print('SCfeedbackStitch: FAIL Wiggling failed')
        ERROR=2
        return
    cnt=1
    for steps in range(maxsteps):
        B = SCgetBPMreading(SC,BPMords=BPMords) # Inject...
        correctionStep() # call correction subroutine.
        if isSetback(BPMhist):
            if verbose: print('SCfeedbackStitch: FAIL Setback')
            ERROR=3
            return
        if isRepro(BPMhist,nRepro) and isTransmit(BPMhist):
            if verbose: print('SCfeedbackStitch: Success')
            ERROR=0
            return # Success
        cnt = cnt+1
    if verbose: print('SCfeedbackStitch: FAIL Reached maxsteps')
    ERROR=1
    return

def correctionStep():
    BPMhist = logLastBPM(BPMhist,B)
    lBPM = len(B[0])
    Bx1 = B[0][0:lBPM/2]
    By1 = B[1][0:lBPM/2]
    Bx2 = B[0][(lBPM/2+1):]
    By2 = B[1][(lBPM/2+1):]
    DELTABx=Bx2-Bx1
    DELTABy=By2-By1
    DELTABx[(nBPMs+1):]=0
    DELTABy[(nBPMs+1):]=0
    R = [Bx1-R0[0],DELTABx,By1-R0[1],DELTABy]
    R[np.isnan(R)]=0
    dphi = Mplus * R
    SC = SCsetCMs2SetPoints(SC,CMords[0],-dphi[0:len(CMords[0])      ],1,'add')
    SC = SCsetCMs2SetPoints(SC,CMords[1],-dphi[  len(CMords[0])+1:end],2,'add')

def wiggle():
    pts = [[0;0] goldenDonut(50E-6,200E-6,32)]
    dpts = diff(pts,1,2)
    for nWiggleCM in [1 2 3 4 5 6 7 8]:
        if verbose: print('SCfeedbackStitch: Number of magnets used for wiggling: %d. \n',nWiggleCM)
        CMords = getLastCMords(B,nWiggleCM)
        for i in range(len(dpts[0])):
            for ord in CMords:
                SC = SCsetCMs2SetPoints(SC,ord,dpts[0][i],1,'add')
                SC = SCsetCMs2SetPoints(SC,ord,dpts[1][i],2,'add')
            W = SCgetBPMreading(SC,BPMords=BPMords)
            BPMhist = logLastBPM(BPMhist,W)
            if isSignal(W,nBPMs): # TODO double check. Seems a bit iffy
                BPMhist = logLastBPM(BPMhist,SCgetBPMreading(SC,BPMords=BPMords))
                BPMhist = logLastBPM(BPMhist,SCgetBPMreading(SC,BPMords=BPMords))
                BPMhist = logLastBPM(BPMhist,SCgetBPMreading(SC,BPMords=BPMords))
                if isRepro(BPMhist,3):
                    BPMhist[0:3] = -1 # void last hist
                    return

def getLastCMords(B,n):
    dualCMords = np.intersect1d(CMords[0],CMords[1]) # Generate a list of CMs that can act in both planes.
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1] # Get index of last reached BPM
    if len(lastBPMidx) == 0 or lastBPMidx > len(BPMords): # If there is no beam loss in the first turn
        ords = dualCMords[-n:] ... just return the last n CMs
    else: # in case the beam was lost in the first turn
        lastBPMord = BPMords[lastBPMidx] # We can then find the ordinate of the last BPM.
        lastCMidx  = np.where(dualCMords <= lastBPMord)[0][-1] # Find the last CM upstream of the last BPM.
        ords = dualCMords[(lastCMidx-min(lastCMidx,n)+1):lastCMidx]
    return ords

def goldenDonut(r0, r1, Npts):
    out = np.zeros((2,Npts)) # initialize output array
    phi = 2*np.pi/((1+np.sqrt(5))/2) # golden ratio
    theta = 0
    for n in range(Npts):
        out[:,n] = np.sqrt((r1**2-r0**2)*n/(Npts-1) + r0**2) * [np.cos(theta), np.sin(theta)]
        theta = theta + phi
    return out

def isNew(BPMhist):
    return BPMhist[0]!=BPMhist[1]

def isSetback(BPMhist):
    return BPMhist[0]!=0 and BPMhist[0]<BPMhist[2] and BPMhist[1]<BPMhist[2]

def isRepro(BPMhist,N):
    return all(BPMhist[0:N]==BPMhist[0])

def isTransmit(BPMhist):
    return BPMhist[0]==0

def isSignal(B,nBPMs):
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1]
    return lastBPMidx >= len(B[0])/2 + nBPMs

def logLastBPM(BPMhist,B):
    BPMhist = np.roll(BPMhist,1)
    ord = getLastBPMord(B)
    if ord:
        BPMhist[0]=ord
    else:
        BPMhist[0]=0
    return BPMhist

def getLastBPMord(B):
    ord = np.where(np.isnan(B[0]))[0][0]-1
    return ord