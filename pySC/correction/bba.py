import matplotlib.pyplot as plt
import numpy as np

from pySC.utils.at_wrapper import findspos
from pySC.correction.orbit_trajectory import SCfeedbackRun
from pySC.core.beam import SCgetBPMreading
from pySC.utils.sc_tools import SCrandnc
from pySC.core.lattice_setting import SCsetCMs2SetPoints, SCsetMags2SetPoints, SCgetCMSetPoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

def SCBBA(SC,BPMords,magOrds,**kwargs):
    """
    SCBBA(SC,BPMords,magOrds,**kwargs)
    SC:        SC structure
    BPMords:   BPMs adjacent to magnets for BBA
    magOrds:   Magnets adjacent to BPMs for BBA
    kwargs:    Optional arguments
    """
    # Optional arguments
    mode = kwargs.get('mode', SC.INJ.trackMode)
    outlierRejectionAt = kwargs.get('outlierRejectionAt', np.inf)
    fakeMeasForFailures = kwargs.get('fakeMeasForFailures', 0)
    dipCompensation = kwargs.get('dipCompensation', True)
    nSteps = kwargs.get('nSteps', 10)
    fitOrder = kwargs.get('fitOrder', 1)
    magOrder = kwargs.get('magOrder', 2)
    magSPvec = kwargs.get('magSPvec', [0.95, 1.05])
    magSPflag = kwargs.get('magSPflag', 'rel')
    skewQuadrupole = kwargs.get('skewQuadrupole', 0)
    switchOffSext = kwargs.get('switchOffSext', 0)
    RMstruct = kwargs.get('RMstruct', [])
    orbBumpWindow = kwargs.get('orbBumpWindow', 5)
    useBPMreadingsForOrbBumpRef = kwargs.get('useBPMreadingsForOrbBumpRef', 0)
    BBABPMtarget = kwargs.get('BBABPMtarget', 1E-3)
    minBPMrangeAtBBABBPM = kwargs.get('minBPMrangeAtBBABBPM', 500E-6)
    minBPMrangeOtherBPM = kwargs.get('minBPMrangeOtherBPM', 100E-6)
    maxStdForFittedCenters = kwargs.get('maxStdForFittedCenters', 600E-6)
    nXPointsNeededAtMeasBPM = kwargs.get('nXPointsNeededAtMeasBPM', 3)
    maxNumOfDownstreamBPMs = kwargs.get('maxNumOfDownstreamBPMs', len(SC.ORD.BPM))
    minSlopeForFit = kwargs.get('minSlopeForFit', 0.03)
    maxTrajChangeAtInjection = kwargs.get('maxTrajChangeAtInjection', [.9E-3, .9E-3])
    quadOrdPhaseAdvance = kwargs.get('quadOrdPhaseAdvance', [ ])
    quadStrengthPhaseAdvance = kwargs.get('quadStrengthPhaseAdvance', [0.95, 1.05])
    plotLines = kwargs.get('plotLines', 0)
    plotResults = kwargs.get('plotResults', 0)
    # Check input
    if BPMords.shape != magOrds.shape:
        raise ValueError('Input arrays for BPMs and magnets must be same size.')
    if not isinstance(magSPvec,list):
        magSPvec = [magSPvec]*len(magOrds)
    initOffsetErrors = getBPMoffsetFromMag(SC,BPMords,magOrds)
    errorFlags = np.nan*np.ones(BPMords.shape)
    kickVec0  = maxTrajChangeAtInjection.reshape(2,1) * np.linspace(-1,1,nSteps)
    initialZ0 = SC.INJ.Z0
    if mode == 'TBT' and SC.INJ.nTurns != 2:
        LOGGER.info('Setting number of turns to 2.')
        SC.INJ.nTurns = 2
    for jBPM in range(BPMords.shape[1]): # jBPM: Index of BPM adjacent to magnet for BBA
        for nDim in range(BPMords.shape[0]):
            LOGGER.debug(f'BBA-BPM {jBPM}/{BPMords.shape[1]}, nDim = {nDim}')
            SC0 = SC
            BPMind = np.where(BPMords[nDim,jBPM]==SC.ORD.BPM)[0][0]
            mOrd = magOrds[nDim,jBPM]
            if switchOffSext:
                SC = SCsetMags2SetPoints(SC,mOrd,skewness=False , order=2 ,setpoints=np.zeros(1) ,method='abs')
                [SC,_] = SCfeedbackRun(SC,RMstruct.MinvCO,target=0,maxsteps=50,scaleDisp=RMstruct.scaleDisp,BPMords=RMstruct.BPMords,CMords=RMstruct.CMords,eps=1E-6)
            if mode == 'ORB':
                [CMords,CMvec] = getOrbitBump(SC,mOrd,BPMords[nDim,jBPM],nDim,RMstruct,orbBumpWindow,useBPMreadingsForOrbBumpRef)
                [BPMpos,tmpTra] = dataMeasurement(SC,mOrd,BPMind,jBPM,nDim,CMords,CMvec,magSPvec[jBPM],magSPflag,magOrder,dipCompensation,plotLines)
            elif mode == 'TBT':
                [kickVec, BPMrange] = scaleInjectionToReachBPM(SC,BPMind,nDim,initialZ0,kickVec0,BBABPMtarget,minBPMrangeAtBBABBPM,minBPMrangeOtherBPM)
                if quadOrdPhaseAdvance and BPMrange < BBABPMtarget:
                    [SC,kickVec] = scanPhaseAdvance(SC,BPMind,nDim,initialZ0,kickVec0,quadOrdPhaseAdvance,quadStrengthPhaseAdvance)
                [BPMpos,tmpTra] = dataMeasurement(SC,mOrd,BPMind,jBPM,nDim,initialZ0,kickVec,magSPvec[jBPM],magSPflag,magOrder,dipCompensation,plotLines)
            [OffsetChange,errorFlags[nDim,jBPM]] = dataEvaluation(BPMpos,tmpTra,nDim,mOrd,fitOrder,maxStdForFittedCenters,nXPointsNeededAtMeasBPM,maxNumOfDownstreamBPMs,minSlopeForFit)
            SC = SC0
            if  OffsetChange > outlierRejectionAt:
                OffsetChange          = np.nan
                errorFlags[nDim,jBPM] = 6
            if not np.isnan(OffsetChange):
                SC.RING[BPMords[nDim,jBPM]].Offset[nDim] = SC.RING[BPMords[nDim,jBPM]].Offset[nDim] + OffsetChange
        if plotResults:
            plotBBAResults(SC,initOffsetErrors,errorFlags,jBPM,BPMords,magOrds)
    if fakeMeasForFailures:
        SC = fakeMeasurement(SC,BPMords,magOrds,errorFlags)
    return SC,errorFlags


def dataMeasurement(SC,mOrd,BPMind,jBPM,nDim,par,varargin):
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
        CMords = varargin[0]
        CMvec  = varargin[1]
        nMsteps = CMvec[nDim].shape[0]
        tmpTra = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM]),len(SC.ORD.BPM)))
        BPMpos = np.nan((nMsteps,len(par.magSPvec[nDim,jBPM])))
    elif par.mode == 'TBT':
        initialZ0 = varargin[0]
        kickVec   = varargin[1]
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
        plt.figure(56)
        plt.clf()
        plt.hold(True)
        p1 = plt.plot(0,1E6*SC.RING[mOrd].T2[2*nDim-1],0,'rD',MarkerSize=40,MarkerFaceColor='b')
        plt.hold(True)
        plt.box('on')
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
                p2 = plt.plot(np.tile(jBPM+nBPM,x.shape),1E6*x,1E3*y,'ko')
                tmp = plt.plot(np.tile(jBPM+nBPM,x.shape),1E6*x,1E3*np.polyval(sol,x),'k-')
                p3 = tmp[0]
                p4 = plt.plot(jBPM+nBPM,1E6*tmpCenter[nBPM],0,'Or',MarkerSize=10)
    if (np.max(tmpRangeX) < par.minBPMrangeAtBBABBPM):
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


def scaleInjectionToReachBPM(SC,BPMind,nDim,initialZ0,kickVec0,par):
    fullTrans     = 0 # Transmission flag
    scalingFactor = 1 # Scaling factor for initial trajectory variation
    BPMrange      = 0 # Offset range at considered BPM
    while fullTrans == 0 and scalingFactor > 1E-6:
        SC.INJ.Z0 = initialZ0
        tmpBPMpos = []
        for nK in [1,size(kickVec0,2)]:
            SC.INJ.Z0[2*nDim]   = initialZ0[2*nDim  ] + scalingFactor * kickVec0[2,nK] # kick angle
            SC.INJ.Z0[2*nDim-1] = initialZ0[2*nDim-1] + scalingFactor * kickVec0[1,nK] # offset
            B = SCgetBPMreading(SC)
            tmpBPMpos.append(B[nDim,BPMind])
        if any(isnan(tmpBPMpos)):
            scalingFactor = scalingFactor - 0.1
        else:
            fullTrans = 1
            BPMrange  = abs(max(tmpBPMpos)-min(tmpBPMpos))
    if scalingFactor<1E-6:
        scalingFactor = 1
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
    kickVec       = scalingFactor * kickVec0
    LOGGER.debug('Initial trajectory variation scaled to [%.0f|%.0f]%% of its initial value, BBA-BPM range %.0fum.' % (100*(kickVec([1 end])./kickVec0([1 end])),1E6*BPMrange))
    return kickVec, BPMrange


def scanPhaseAdvance(SC,BPMind,nDim,initialZ0,kickVec0,par):
    mOrd = par.quadOrdPhaseAdvance
    qVec = par.quadStrengthPhaseAdvance
    q0   = SC.RING[mOrd].SetPointB[1]
    allBPMRange = np.zeros(len(qVec))
    for nQ in range(len(qVec)):
        LOGGER.debug(f'BBA-BPM range to small, try to change phase advance with quad ord {par.quadOrdPhaseAdvance} to {qVec[nQ]:.2f} of nom. SP.')
        SC = SCsetMags2SetPoints(SC,mOrd,False,1,qVec[nQ],method='rel', dipCompensation=True)
        [kickVec, BPMrange] = scaleInjectionToReachBPM(SC,BPMind,nDim,initialZ0,kickVec0,par)
        allBPMRange[nQ] = BPMrange
        LOGGER.debug(f'Initial trajectory variation scaled to '
                     f'[{100 * (kickVec[0] / kickVec0[0]):.0f}|{100 * (kickVec[-1] / kickVec0[-1]):.0f}]% '
                     f'of its initial value, BBA-BPM range {1E6 * BPMrange:.0f}um.')
        if not ( BPMrange < par.BBABPMtarget ):
            break
    if BPMrange < par.BBABPMtarget:
        if BPMrange<max(allBPMRange):
            nBest = np.argmax(allBPMRange)
            SC = SCsetMags2SetPoints(SC,mOrd,False,1,qVec[nBest], method='rel', dipCompensation=True)
            LOGGER.debug(
                f'Changing phase advance of quad with ord {mOrd} NOT succesfull, returning to best value with BBA-BPM range = {1E6 * max(allBPMRange):.0f}um.')
        else:
            SC = SCsetMags2SetPoints(SC,mOrd,False,1,q0, method='abs',dipCompensation=True)
            LOGGER.debug(
                f'Changing phase advance of quad with ord {mOrd} NOT succesfull, returning to initial setpoint.')
    else:
        LOGGER.debug(
            f'Change phase advance of quad with ord {mOrd} successful. BBA-BPM range = {1E6 * BPMrange:.0f}um.')
    return SC,kickVec

def getOrbitBump(SC,mOrd,BPMord,nDim,par):
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
    CUR,_ = SCfeedbackRun(SC,par.RMstruct.MinvCO,
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
    sPos = findspos(SC.RING,np.arange(len(SC.RING)))
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
    fom =     getBPMoffsetFromMag(SC,BPMords,magOrds)
    fom[:,jBPM+1:end] = np.nan
    if size(BPMords,2)==1:
        nSteps = 1
    else:
        nSteps = 1.1*max(abs(fom0(:)))*np.linspace(-1,1,floor(size(BPMords,2)/3))
    plt.figure(90)
    plt.clf()
    tmpCol=plt.gca().colororder
    plt.subplot(3,1,1)
    for nDim in range(size(BPMords,1)):
        a,b = np.histogram(fom[nDim,:],nSteps)
        plt.plot(1E6*b,a,linewidth=2)
    a,b = np.histogram(fom0(:),nSteps)
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

def getBPMoffsetFromMag(SC,BPMords,magOrds):
    offset = np.nan(len(BPMords))
    for nDim in range(len(BPMords,1)):
        for nBPM in range(len(BPMords,2)):
            offset[nDim,nBPM] = SC.RING[BPMords[nDim,nBPM]].Offset[nDim] + SC.RING[BPMords[nDim,nBPM]].SupportOffset[nDim] - SC.RING[magOrds[nDim,nBPM]].MagnetOffset[nDim] - SC.RING[magOrds[nDim,nBPM]].SupportOffset[nDim]
    return offset

def fakeMeasurement(SC,BPMords,magOrds,errorFlags):
    finOffsetErrors = getBPMoffsetFromMag(SC,BPMords,magOrds)
    finOffsetErrors[errorFlags!=0] = np.nan
    LOGGER.info(f"Final offset error is {1E6*np.sqrt(np.nanmean(finOffsetErrors**2,2))}"
                f" um (hor|ver) with {sum(errorFlags!=0,2)} measurement failures -> being re-calculated now.\n")
    for nBPM in range(len(BPMords,2)):
        for nDim in range(2):
            if errorFlags[nDim,nBPM]!=0:
                fakeBPMoffset = SC.RING[magOrds[nDim,nBPM]].MagnetOffset[nDim] + SC.RING[magOrds[nDim,nBPM]].SupportOffset[nDim] - SC.RING[BPMords[nDim,nBPM]].SupportOffset[nDim] + np.sqrt(np.mean(finOffsetErrors(nDim,:).^2,'omitnan')) * SCrandnc(2)
                if not np.isnan(fakeBPMoffset):
                    SC.RING[BPMords[nDim,nBPM]].Offset[nDim] = fakeBPMoffset
                else:
                    LOGGER.info('BPM offset not reasigned, NaN.\n')