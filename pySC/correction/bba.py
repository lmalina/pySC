import matplotlib.pyplot as plt
import numpy as np

from pySC.utils.at_wrapper import findspos, atgetfieldvalues
from pySC.correction.orbit_trajectory import SCfeedbackRun
from pySC.core.beam import SCgetBPMreading
from pySC.utils.sc_tools import SCrandnc
from pySC.core.lattice_setting import SCsetCMs2SetPoints, SCsetMags2SetPoints, SCgetCMSetPoints
from pySC.utils import logging_tools
from pySC.core.classes import DotDict

LOGGER = logging_tools.get_logger(__name__)

def SCBBA(SC, BPMords, magOrds, **kwargs):
    par = DotDict(dict(mode=SC.INJ.trackMode, outlierRejectionAt=np.inf, nSteps= 10, fitOrder= 1, magOrder= 2,
                       magSPvec= [0.95, 1.05], magSPflag= 'rel', RMstruct= [], orbBumpWindow= 5,BBABPMtarget= 1E-3,
                       minBPMrangeAtBBABBPM= 500E-6, minBPMrangeOtherBPM= 100E-6, maxStdForFittedCenters= 600E-6,
                       nXPointsNeededAtMeasBPM= 3, maxNumOfDownstreamBPMs= len(SC.ORD.BPM), minSlopeForFit= 0.03,
                       maxTrajChangeAtInjection= [.9E-3, .9E-3],quadOrdPhaseAdvance= [ ],
                       quadStrengthPhaseAdvance= [0.95, 1.05], fakeMeasForFailures=False, dipCompensation= True,
                       skewQuadrupole= False, switchOffSext= False, useBPMreadingsForOrbBumpRef= False,
                       plotLines= False, plotResults=False))
    par.update(**kwargs)
    if BPMords.shape != magOrds.shape: # both in shape 2 x N
        raise ValueError('Input arrays for BPMs and magnets must be same size.')
    if not isinstance(par.magSPvec,list):
        par.magSPvec = [par.magSPvec]*len(magOrds)
    if par.mode not in ("TBT", "ORB"):
        raise ValueError(f"Unknown mode {par.mode}.")
    initOffsetErrors = _get_bpm_offset_from_mag(SC, BPMords, magOrds)
    errorFlags = np.full(BPMords.shape, np.nan)
    kickVec0  = par.maxTrajChangeAtInjection.reshape(2,1) * np.linspace(-1, 1, par.nSteps)
    initialZ0 = SC.INJ.Z0
    if par.mode == 'TBT' and SC.INJ.nTurns != 2:
        LOGGER.info('Setting number of turns to 2.')
        SC.INJ.nTurns = 2
    for jBPM in range(BPMords.shape[1]): # jBPM: Index of BPM adjacent to magnet for BBA
        for nDim in range(BPMords.shape[0]):
            LOGGER.debug(f'BBA-BPM {jBPM}/{BPMords.shape[1]}, nDim = {nDim}')
            SC0 = SC
            BPMind = np.where(BPMords[nDim,jBPM]==SC.ORD.BPM)[0][0]
            mOrd = magOrds[nDim,jBPM]
            if par.switchOffSext:
                SC = SCsetMags2SetPoints(SC,mOrd,skewness=False, order=2 ,setpoints=np.zeros(1) ,method='abs')
                SC = SCfeedbackRun(SC,par.RMstruct.MinvCO,BPMords=par.RMstruct.BPMords,CMords=par.RMstruct.CMords,
                                   target=0,maxsteps=50,scaleDisp=par.RMstruct.scaleDisp,eps=1E-6)
            if par.mode == 'ORB':
                CMords, CMvec = getOrbitBump(SC,mOrd,BPMords[nDim,jBPM],nDim,par)
                BPMpos,tmpTra = _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par, [CMords, CMvec])
            else:
                kickVec, BPMrange = _scale_injection_to_reach_bpm(SC, BPMind, nDim, initialZ0, kickVec0)
                if par.quadOrdPhaseAdvance and BPMrange < par.BBABPMtarget:
                    SC,kickVec = scanPhaseAdvance(SC,BPMind,nDim,initialZ0,kickVec0,par)
                BPMpos, tmpTra = _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par, [initialZ0, kickVec])
            OffsetChange,errorFlags[nDim,jBPM] = dataEvaluation(SC,BPMords,jBPM,BPMpos,tmpTra,nDim,mOrd,par)
            SC = SC0
            if  OffsetChange > par.outlierRejectionAt:
                OffsetChange = np.nan
                errorFlags[nDim,jBPM] = 6
            if not np.isnan(OffsetChange):
                SC.RING[BPMords[nDim,jBPM]].Offset[nDim] = SC.RING[BPMords[nDim,jBPM]].Offset[nDim] + OffsetChange
        if par.plotResults:
            plotBBAResults(SC,initOffsetErrors,errorFlags,jBPM,BPMords,magOrds)
    if par.fakeMeasForFailures:
        SC = _fake_measurement(SC, BPMords, magOrds, errorFlags)
    return SC,errorFlags


def _get_bpm_offset_from_mag(SC, BPMords, magOrds):
    offset = np.full(BPMords.shape, np.nan)
    for nDim in range(2):
        offset[nDim,:] = (atgetfieldvalues(SC.RING, BPMords[nDim, :], 'Offset', nDim)
                          + atgetfieldvalues(SC.RING, BPMords[nDim, :], 'SupportOffset', nDim)
                          - atgetfieldvalues(SC.RING, magOrds[nDim, :], 'MagnetOffset', nDim)
                          - atgetfieldvalues(SC.RING, magOrds[nDim, :], 'SupportOffset', nDim))
    return offset


def _fake_measurement(SC, BPMords, magOrds, errorFlags):
    finOffsetErrors = _get_bpm_offset_from_mag(SC, BPMords, magOrds)
    finOffsetErrors[errorFlags!=0] = np.nan
    LOGGER.info(f"Final offset error is {1E6*np.sqrt(np.nanmean(finOffsetErrors**2, axis=1))}"
                f" um (hor|ver) with {np.sum(errorFlags!=0,axis=1)} measurement failures -> being re-calculated now.\n")
    for nBPM in range(BPMords.shape[1]):
        for nDim in range(2):
            if errorFlags[nDim,nBPM]!=0:
                fakeBPMoffset = (SC.RING[magOrds[nDim,nBPM]].MagnetOffset[nDim]
                                 + SC.RING[magOrds[nDim,nBPM]].SupportOffset[nDim]
                                 - SC.RING[BPMords[nDim,nBPM]].SupportOffset[nDim]
                                 + np.sqrt(np.nanmean(np.square(finOffsetErrors[nDim,:]))) * SCrandnc(2))
                if not np.isnan(fakeBPMoffset):
                    SC.RING[BPMords[nDim,nBPM]].Offset[nDim] = fakeBPMoffset
                else:
                    LOGGER.info('BPM offset not reasigned, NaN.\n')
    return SC

def _data_measurement(SC, mOrd, BPMind, jBPM, nDim, par, varargin):
    if par.skewQuadrupole:
        skewness = True
        if nDim==1:
            measDim = 2  # TODO why the swap?
        else:
            measDim = 1
    else:
        skewness = False
        measDim = nDim
    if par.mode == 'ORB':
        CMords, CMvec = varargin
        nMsteps = CMvec[nDim].shape[0]
        tmpTra = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM]),len(SC.ORD.BPM)))
        BPMpos = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM])))
    else:
        initialZ0, kickVec = varargin
        nMsteps = kickVec.shape[1]
        tmpTra = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM]),par.maxNumOfDownstreamBPMs))
        BPMpos = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM])))
    for nQ in range(len(par.magSPvec[nDim,jBPM])):
        SC = SCsetMags2SetPoints(SC,mOrd,skewness,par.magOrder -1,  #  TODO remove -1 once the correct order is passed
                                 par.magSPvec[nDim,jBPM][nQ],method=par.magSPflag,dipCompensation=par.dipCompensation)
        for nKick in range(nMsteps):
            if par.mode == 'ORB':
                for nD in range(2):
                    SC, _ = SCsetCMs2SetPoints(SC,CMords[nD],CMvec[nD][nKick,:],nD,method='abs')
            elif par.mode == 'TBT':
                SC.INJ.Z0[2*nDim]   = initialZ0[2*nDim  ] + kickVec[2,nKick] # kick angle
                SC.INJ.Z0[2*nDim-1] = initialZ0[2*nDim-1] + kickVec[1,nKick] # offset
            B = SCgetBPMreading(SC)
            if par.plotLines:
                plotBBAstep(SC,BPMind,jBPM,nDim,nQ,mOrd,nKick,par)
            BPMpos[nKick,nQ] = B[nDim,BPMind]
            if par.mode == 'ORB':
                tmpTra[nKick,nQ,:] = B[measDim, : ]
            elif par.mode == 'TBT':
                tmpTra[nKick,nQ,:] = B[measDim, (BPMind+1):(BPMind+par.maxNumOfDownstreamBPMs) ]
    return BPMpos,tmpTra


def dataEvaluation(SC,BPMords,jBPM,BPMpos,tmpTra,nDim,mOrd,par):
    if par.plotLines:
        fig, ax = plt.subplots(num=56)
        p1 = ax.plot(0,1E6*SC.RING[mOrd].T2[2*nDim-1],0,'rD',MarkerSize=40,MarkerFaceColor='b')
    OffsetChange = np.nan
    Error        = 5
    tmpCenter = np.nan((1,(tmpTra.shape[1]-1)*par.maxNumOfDownstreamBPMs))
    tmpNorm   = np.nan((1,(tmpTra.shape[1]-1)*par.maxNumOfDownstreamBPMs))
    tmpRangeX = np.zeros((1,(tmpTra.shape[1]-1)*par.maxNumOfDownstreamBPMs))
    tmpRangeY = np.zeros((1,(tmpTra.shape[1]-1)*par.maxNumOfDownstreamBPMs))
    i = 0
    for nBPM in range(par.maxNumOfDownstreamBPMs):
        y0 = np.diff(tmpTra[:,:,nBPM],1,1)
        x0 = np.tile(np.mean(BPMpos,1),(y0.shape[1],1)).T
        for nKick in range(y0.shape[1]):
            i = i+1
            y = y0[:,nKick]
            x = x0[:,nKick]
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            if len(x)==0 or len(y)==0:
                continue
            tmpRangeX[i] = abs(np.min(x)-np.max(x))
            tmpRangeY[i] = abs(np.min(y)-np.max(y))
            sol = np.nan((1,2))
            if len(x)>=par.nXPointsNeededAtMeasBPM and tmpRangeX[i] > par.minBPMrangeAtBBABBPM and tmpRangeY[i] > par.minBPMrangeOtherBPM:
                if par.fitOrder==1:
                    sol = np.linalg.lstsq(np.vstack((np.ones(x.shape),x)).T,y)[0]
                    sol = sol[[1,0]]
                    if abs(sol[0]) < par.minSlopeForFit:
                        sol[0] = np.nan
                    tmpCenter[i] = -sol[1]/sol[0]
                    tmpNorm[i]   = 1/np.sqrt(np.sum((sol[0]*x+sol[1]-y)**2))
                else:
                    sol = np.polyfit(x,y,par.fitOrder)
                    if par.fitOrder==2:
                        tmpCenter[i] = - (sol[1]/(2*sol[0]))
                    else:
                        tmpCenter[i] = min(abs(np.roots(sol)))
                    tmpNorm[i]   = 1/np.linalg.norm(np.polyval(sol,x)-y)
            if par.plotLines:
                p2 = ax.plot(np.tile(jBPM+nBPM,x.shape),1E6*x,1E3*y,'ko')
                tmp = ax.plot(np.tile(jBPM+nBPM,x.shape),1E6*x,1E3*np.polyval(sol,x),'k-')
                p3 = tmp[0]
                p4 = plt.plot(jBPM+nBPM,1E6*tmpCenter[nBPM],0,'Or',MarkerSize=10)
    if np.max(tmpRangeX) < par.minBPMrangeAtBBABBPM:
        Error = 1
    elif np.max(tmpRangeY) < par.minBPMrangeOtherBPM:
        Error = 2
    elif np.std(tmpCenter,ddof=1) > par.maxStdForFittedCenters:
        Error = 3
    elif len(np.where(~np.isnan(tmpCenter))[0])==0:
        Error = 4
    else:
        OffsetChange = np.sum(tmpCenter*tmpNorm)/np.sum(tmpNorm)
        Error = 0
    if not par.dipCompensation and nDim==1 and SC.RING[mOrd].NomPolynomB[1]!=0:
        if 'BendingAngle' in SC.RING[mOrd].keys():
            B = SC.RING[mOrd].BendingAngle
        else:
            B = 0
        K = SC.RING[mOrd].NomPolynomB[1]
        L = SC.RING[mOrd].Length
        OffsetChange = OffsetChange + B/L/K
    if par.plotLines:
        p5 = plt.plot(0,1E6*OffsetChange,0,'kD',MarkerSize=30,MarkerFaceColor='r')
        p6 = plt.plot(0,1E6*(SC.RING[BPMords[nDim,jBPM]].Offset[nDim]+SC.RING[BPMords[nDim,jBPM]].SupportOffset[nDim]+OffsetChange),0,'kD',MarkerSize=30,MarkerFaceColor='g')
        plt.title(f'BBA-BPM: {jBPM:d} \n mOrd: {mOrd:d} \n mFam: {SC.RING[mOrd].FamName} \n nDim: {nDim:d} \n FinOffset = {1E6*np.abs(SC.RING[BPMords[nDim,jBPM]].Offset[nDim] + SC.RING[BPMords[nDim,jBPM]].SupportOffset[nDim] + OffsetChange - SC.RING[mOrd].MagnetOffset[nDim] - SC.RING[mOrd].SupportOffset[nDim]):3.0f} $\\mu m$')
        plt.legend((p1,p2,p3,p4,p5,p6),('Magnet center','Measured offset change','Line fit','Fitted BPM offset (individual)','Fitted BPM offset (mean)','Predicted magnet center'))
        plt.xlabel('Index of BPM')
        plt.ylabel('BBA-BPM offset [$\mu$m]')
        plt.zlabel('Offset change [mm]')
        plt.setp(plt.findall(plt.figure(56),'-property','TickLabelInterpreter'),TickLabelInterpreter='latex')
        plt.setp(plt.findall(plt.gcf(),'-property','Interpreter'),Interpreter='latex')
        plt.setp(plt.findall(plt.gcf(),'-property','FontSize'),FontSize=18)
        plt.gcf().color='w'
        plt.draw()
    return OffsetChange,Error


def _scale_injection_to_reach_bpm(SC, BPMind, nDim, initialZ0, kickVec0):
    fullTrans     = 0 # Transmission flag
    scalingFactor = 1 # Scaling factor for initial trajectory variation
    BPMrange      = 0 # Offset range at considered BPM
    while fullTrans == 0 and scalingFactor > 1E-6:
        SC.INJ.Z0 = initialZ0   # TODO not nice
        tmpBPMpos = np.full(kickVec0.shape[1], np.nan)
        for nK in range(kickVec0.shape[1]):
            SC.INJ.Z0[2*nDim]   = initialZ0[2*nDim  ] + scalingFactor * kickVec0[1,nK] # kick angle
            SC.INJ.Z0[2*nDim-1] = initialZ0[2*nDim-1] + scalingFactor * kickVec0[0,nK] # offset
            tmpBPMpos[nK] = SCgetBPMreading(SC)[nDim,BPMind]
        if np.any(np.isnan(tmpBPMpos)):
            scalingFactor = scalingFactor - 0.1
        else:
            fullTrans = 1
            BPMrange  = np.max(tmpBPMpos)-np.min(tmpBPMpos)
    if scalingFactor<1E-6:
        scalingFactor = 1
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
    kickVec       = scalingFactor * kickVec0
    LOGGER.debug(f'Initial trajectory variation scaled to [{100 * (kickVec[0] / kickVec0[0]):.0f}| {100 * (kickVec[-1] / kickVec0[-1]):.0f}]% '
                 f'of its initial value, BBA-BPM range {1E6*BPMrange:.0f} um.')
    return kickVec, BPMrange


def scanPhaseAdvance(SC,BPMind,nDim,initialZ0,kickVec0,par):
    mOrd = par.quadOrdPhaseAdvance
    qVec = par.quadStrengthPhaseAdvance
    q0   = SC.RING[mOrd].SetPointB[1]
    allBPMRange = np.zeros(len(qVec))
    for nQ in range(len(qVec)):
        LOGGER.debug(f'BBA-BPM range to small, try to change phase advance with quad ord {par.quadOrdPhaseAdvance} '
                     f'to {qVec[nQ]:.2f} of nom. SP.')
        SC = SCsetMags2SetPoints(SC, mOrd, False, 1, qVec[nQ], method='rel', dipCompensation=True)
        kickVec, BPMrange = _scale_injection_to_reach_bpm(SC, BPMind, nDim, initialZ0, kickVec0)
        allBPMRange[nQ] = BPMrange
        LOGGER.debug(f'Initial trajectory variation scaled to '
                     f'[{100 * (kickVec[0] / kickVec0[0]):.0f}|{100 * (kickVec[-1] / kickVec0[-1]):.0f}]% '
                     f'of its initial value, BBA-BPM range {1E6 * BPMrange:.0f}um.')
        if BPMrange >= par.BBABPMtarget:
            LOGGER.debug(f'Change phase advance with quad ord {mOrd} successful. BBA-BPM range = {1E6 * BPMrange:.0f}um.')
            return SC, kickVec

    if BPMrange<max(allBPMRange):
        LOGGER.debug(f'Changing phase advance of quad with ord {mOrd} NOT succesfull, '
                     f'returning to best value with BBA-BPM range = {1E6 * max(allBPMRange):.0f}um.')
        return SCsetMags2SetPoints(SC, mOrd, False, 1, np.max(qVec), method='rel', dipCompensation=True), kickVec
    LOGGER.debug(f'Changing phase advance of quad with ord {mOrd} NOT succesfull, returning to initial setpoint.')
    return SCsetMags2SetPoints(SC, mOrd, False, 1, q0, method='abs',dipCompensation=True), kickVec


def getOrbitBump(SC, mOrd, BPMord, nDim, par):
    tmpCMind = np.where(par.RMstruct.CMords[0]==mOrd)[0]
    if tmpCMind.size != 0:
        par.RMstruct.RM = np.delete(par.RMstruct.RM,tmpCMind,1)
        par.RMstruct.CMords[0] = np.delete(par.RMstruct.CMords[0],tmpCMind)
    tmpBPMind = np.where(BPMord==par.RMstruct.BPMords)[0]
    if par.useBPMreadingsForOrbBumpRef:
        R0 = SCgetBPMreading(SC)
        R0[nDim,tmpBPMind] = R0[nDim,tmpBPMind] + par.BBABPMtarget
    else:
        R0 = np.zeros((2,len(par.RMstruct.BPMords)))
        R0[nDim,tmpBPMind] = par.BBABPMtarget
    W0 = np.ones((2,len(par.RMstruct.BPMords)))
    W0[nDim,max(1,tmpBPMind-par.orbBumpWindow):(tmpBPMind-1)] = 0
    W0[nDim,(tmpBPMind+1):min(len(par.RMstruct.BPMords),tmpBPMind+par.orbBumpWindow)] = 0
    CUR = SCfeedbackRun(SC,par.RMstruct.MinvCO,
                                    reference=np.vstack((R0[0,:],R0[1,:])),
                                    target=0,
                                    maxsteps=50,
                                    scaleDisp=par.RMstruct.scaleDisp,
                                    BPMords=par.RMstruct.BPMords,
                                    CMords=par.RMstruct.CMords,
                                    eps=1E-6)
    for nDim in range(2):
        for nCM in range(len(par.RMstruct.CMords[nDim])):
            CMvec0[nDim][nCM] = SCgetCMSetPoints(SC,par.RMstruct.CMords[nDim][nCM],skewness=nDim)
            deltaCM[nDim][nCM] = SCgetCMSetPoints(SC,par.RMstruct.CMords[nDim][nCM],skewness=nDim) - SCgetCMSetPoints(CUR,par.RMstruct.CMords[nDim][nCM],skewness=nDim)
    factor = np.linspace(-1,1,par.nSteps)
    for nDim in range(2):
        for nStep in range(par.nSteps):
            CMvec[nDim][nStep,:] = CMvec0[nDim] + factor[nStep] * deltaCM[nDim]
    CMords = par.RMstruct.CMords
    return CMords,CMvec

def plotBBAstep(SC,BPMind,jBPM,nDim,nQ,mOrd,nKick,par):
    sPos = findspos(SC.RING)
    xLim = sPos[mOrd]+[-10 10]
    yLim = 1.3*[-1 1]
    if nQ==1 and nKick==1:
        plt.figure(99)
        plt.clf()
    B,T=SCgetBPMreading(SC)
    #T=SCparticlesIn3D(T,SC.INJ.nParticles)
    T=T[:,:,1]
    plt.figure(99)
    plt.subplot(len(par.magSPvec[nDim,jBPM]),1,nQ)
    plt.hold(True)
    plt.plot(sPos[SC.ORD.BPM],1E3*B[nDim,0:len(SC.ORD.BPM)],marker='o')
    plt.plot(sPos[SC.ORD.BPM[BPMind]],1E3*B[nDim,BPMind],marker='o',markersize=10,markerfacecolor='k')
    plt.plot(sPos,1E3*T[2*nDim-1,0:len(SC.RING)],linestyle='-')
    plt.rectangle([sPos[mOrd],-1,sPos[mOrd+1]-sPos[mOrd],1 ],facecolor=[0,0.4470,0.7410])
    plt.xlim(xLim)
    plt.ylim(yLim)

def plotBBAResults(SC,initOffsetErrors,errorFlags,jBPM,BPMords,magOrds):
    fom0  = initOffsetErrors
    fom = _get_bpm_offset_from_mag(SC, BPMords, magOrds)
    fom[:,jBPM+1:] = np.nan
    if BPMords.shape[1]==1:
        nSteps = 1
    else:
        nSteps = 1.1 * np.max(np.abs(fom0)) * np.linspace(-1, 1, np.floor(BPMords.shape[1]/3))
    plt.figure(90)
    plt.clf()
    tmpCol=plt.gca().colororder
    plt.subplot(3,1,1)
    for nDim in range(size(BPMords,1)):
        a,b = np.histogram(fom[nDim,:],nSteps)
        plt.plot(1E6*b,a,linewidth=2)
    a,b = np.histogram(fom0(:), nSteps)
    plt.plot(1E6*b,a,'k-',linewidth=2)
    if size(BPMords,1)>1:
        plt.legend({sprintf('Horizontal rms: $%.0f\\mu m$',1E6*sqrt(mean(fom(1,:).^2,'omitnan'))),sprintf('Vertical rms: $%.0f\\mu m$',1E6*sqrt(mean(fom(2,:).^2,'omitnan'))),sprintf('Initial rms: $%.0f\\mu m$',1E6*sqrt(mean(fom0(:).^2,'omitnan')))},'Interpreter','latex')
    plt.xlabel('Final BPM offset w.r.t. magnet [$\mu$m]')
    plt.ylabel('Number of counts')
    plt.set(gca,'box','on')
    plt.subplot(3,1,2)
    plt.hold(True)
    p=np.zeros(1,4)
    for nDim in range(size(BPMords,1)):
        x = np.where(np.in1d(SC.ORD.BPM,BPMords[nDim,:]))
        if any(~isnan(fom(nDim,errorFlags(nDim,:)==0))):
            p[nDim]=plt.plot(x(errorFlags(nDim,:)==0),1E6*abs(fom(nDim,errorFlags(nDim,:)==0)),marker='O',linewidth=2,color=tmpCol[nDim,:])
        if any(~isnan(fom(nDim,errorFlags(nDim,:)~=0))):
            p[2+nDim]=plt.plot(x(errorFlags(nDim,:)~=0),1E6*abs(fom(nDim,errorFlags(nDim,:)~=0)),marker='X',linewidth=2,color=tmpCol[nDim,:])
    plt.ylabel('Final offset [$\mu$m]')
    plt.xlabel('Index of BPM')
    plt.set(gca,'XLim',[1 length(SC.ORD.BPM)],'box','on')
    legStr = {'Horizontal','Vertical','Horizontal failed','Vertical failed'}
    plt.legend(p(p~=0),legStr{p~=0})
    plt.subplot(3,1,3)
    plt.hold(True)
    p=np.zeros(1,4)
    x = np.where(np.in1d(SC.ORD.BPM,BPMords[nDim,:]))
    for nDim in range(size(BPMords,1)):
        plt.plot(x,1E6*(fom0(nDim,:)-fom(nDim,:)),marker='d',linewidth=2)
    plt.ylabel('Offsets change [$\mu$m]')
    plt.xlabel('Index of BPM')
    plt.set(gca,'XLim',[1 length(SC.ORD.BPM)],'box','on')
    plt.legend({'Horizontal','Vertical'})
    plt.set(gcf,'color','w')
    plt.set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
    plt.set(findall(gcf,'-property','Interpreter'),'Interpreter','latex')
    plt.set(findall(gcf,'-property','FontSize'),'FontSize',18)
    plt.draw()



