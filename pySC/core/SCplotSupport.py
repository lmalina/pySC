import matplotlib.pyplot as plt
import numpy as np

from pySC.core.SCgetSupportOffset import SCgetSupportOffset
from pySC.core.SCgetSupportRoll import SCgetSupportRoll


def SCplotSupport(SC,fontSize=12,shiftAxes=0.03,xLim=[0, findspos(SC.RING,len(SC.RING)+1)]):
    if not hasattr(SC.ORD,'Magnet'):
        raise Exception('Magnets must be registered. Use SCregisterMagnets.')
    elif not hasattr(SC.ORD,'BPM'):
        raise Exception('BPMs must be registered. Use SCregisterBPMs.')
    C = findspos(SC.RING,len(SC.RING)+1)
    s = np.linspace(xLim[0],xLim[1],100*(xLim[1]-xLim[0]))
    sPos = findspos(SC.RING,range(1,len(SC.RING)+1))
    offSupportLine  = SCgetSupportOffset(SC,s)
    rollSupportLine = SCgetSupportRoll(SC,s)
    i=0
    for ord in SC.ORD.Magnet:
        if sPos[ord-1]>=xLim[0] and sPos[ord-1]<=xLim[1]:
            i=i+1
            magOrds[i] = ord
            offMagSupport[:,i]=SC.RING[ord-1].SupportOffset
            rollMagSupport[:,i]=SC.RING[ord-1].SupportRoll
            offMagInd[:,i]=SC.RING[ord-1].MagnetOffset
            rollMagInd[:,i]=SC.RING[ord-1].MagnetRoll
            offMagTot[:,i]=SC.RING[ord-1].T2([1, 3, 6])
            rollMagTot[:,i]=SC.RING[ord-1].MagnetRoll + SC.RING[ord-1].SupportRoll
    for type in ['Section','Plinth','Girder']:
        if hasattr(SC.ORD,type):
            i=1
            for ordPair in SC.ORD.(type):
                if (sPos[ordPair[0]-1]>=xLim[0] and sPos[ordPair[0]-1]<=xLim[1]) or (sPos[ordPair[1]-1]>=xLim[0] and sPos[ordPair[1]-1]<=xLim[1]):
                    SuppOrds.(type)[:,i] = ordPair
                    SuppOff.(type).a[:,i]=SC.RING[ordPair[0]-1].(type+'Offset')
                    SuppOff.(type).b[:,i]=SC.RING[ordPair[1]-1].(type+'Offset')
                    SuppRoll.(type)[:,i]=SC.RING[ordPair[0]-1].(type+'Roll')
                    i = i+1
    i=0
    for ord in SC.ORD.BPM:
        if sPos[ord-1]>=xLim[0] and sPos[ord-1]<=xLim[1]:
            i=i+1
            BPMords[i] = ord
    sBPM         = findspos(SC.RING,BPMords)
    offBPM       = np.array(atgetfieldvalues(SC.RING[BPMords-1],'Offset'))
    offBPMStruct = np.array(atgetfieldvalues(SC.RING[BPMords-1],'SupportOffset'))
    offBPM[      :,2] = 0 # Longitudinal offsets not supported for BPMs
    offBPMStruct[:,2] = 0 # Longitudinal offsets not supported for BPMs
    rollBPM       = np.array(atgetfieldvalues(SC.RING[BPMords-1],'Roll',{1,1}))
    rollBPMStruct = np.array(atgetfieldvalues(SC.RING[BPMords-1],'SupportRoll',{1,1}))
    rollBPM[      :,1:2] = 0 # Pitch and yaw not supported for BPMs
    rollBPMStruct[:,1:2] = 0 # Pitch and yaw not supported for BPMs
    plt.figure(1213);plt.clf();tmpCol=plt.gca().get_color_cycle();ax=[]
    yLabOffStr  = ['$\Delta x$ [$\mu$m]','$\Delta y$ [$\mu$m]','$\Delta z$ [$\mu$m]']
    yLabRollStr = ['$a_z$ [$\mu$rad]','$a_x$ [$\mu$rad]','$a_y$ [$\mu$rad]']
    titlteOffStr = ['Horizontal Offsets','Vertical Offsets','Longitudinal Offsets']
    titlteRollStr  = ['Roll (roll around z-axis)','Pitch (roll around x-axis)','Yaw (roll around y-axis)']
    lineSpec.Plinth  = {'Color','r','LineWidth',4}
    lineSpec.Section = {'Color',tmpCol[7,:], 'LineWidth',2, 'LineStyle',':'}
    lineSpec.Girder  = {'Color',tmpCol[5,:], 'LineWidth',4}
    for nDim in range(3):
        ax[3*(nDim-1)+1,1]=plt.subplot(12,2,2*4*(nDim-1)+ [1 3]);plt.hold(True)
        pVec=[];legStr=[]
        plt.plot(s,1E6*offSupportLine[nDim,:])
        plt.plot(sPos[magOrds-1],1E6*offMagSupport[nDim,:],'D',Color=tmpCol[1,:])
        pVec.append(plt.plot([-2, -1],[0, 0],'-D',Color=tmpCol[1,:]))
        legStr.append('Overall support structure')
        pVec.append(plt.plot(sPos[magOrds-1],1E6*offMagInd[nDim,:],'kx',MarkerSize=8))
        legStr.append('Individual Magnet')
        for type in ['Section','Plinth','Girder']:
            if hasattr(SuppOrds,type):
                for i in range(SuppOrds.(type).shape[1]):
                    if diff(findspos(SC.RING,SuppOrds.(type)[:,i]))<0:
                        for nCase in range(2):
                            if nCase==1:
                                splot  = findspos(SC.RING,[SuppOrds.(type)[0,i], len(SC.RING)])
                                sint   = [findspos(SC.RING,SuppOrds.(type)[0,i]),findspos(SC.RING,SuppOrds.(type)[1,i])+C]
                            else:
                                splot  = findspos(SC.RING,[1, SuppOrds.(type)[1,i]])
                                sint   = [-findspos(SC.RING,SuppOrds.(type)[1,i]),findspos(SC.RING,SuppOrds.(type)[1,i])]
                            offInt = np.interp(splot,[sint[0], sint[1]],[SuppOff.(type).a[nDim,i], SuppOff.(type).b[nDim,i]])
                            plt.plot(splot,1E6*offInt,lineSpec.(type){:})
                    else:
                        plt.plot(findspos(SC.RING,SuppOrds.(type)[:,i]),1E6*[SuppOff.(type).a[nDim,i], SuppOff.(type).b[nDim,i]],lineSpec.(type){:})
                pVec.append(plt.plot([-2, -1],[0, 0],lineSpec.(type){:}))
                legStr.append('Individual '+type)
        plt.legend(pVec,legStr)
        plt.set(gca,'xlim',xLim,'box','on')
        plt.ylabel(yLabOffStr[nDim])
        plt.title(titlteOffStr[nDim])
        ax[3*(nDim-1)+2,1]=plt.subplot(12,2,2*4*(nDim-1)+ 5);plt.hold(True)
        plt.plot(sPos[magOrds-1],1E6*offMagTot[nDim,:],'kO-')
        plt.legend('Overall magnet offset')
        plt.set(gca,'xlim',xLim,'box','on')
        plt.ylabel(yLabOffStr[nDim])
        ax[3*(nDim-1)+3,1]=plt.subplot(12,2,2*4*(nDim-1)+ 7);plt.hold(True)
        plt.plot(sBPM,1E6*offBPM[:,nDim],'O',Color=tmpCol[2,:],MarkerSize=6)
        plt.plot(sBPM,1E6*offBPMStruct[:,nDim],'-',Color=tmpCol[2,:])
        plt.legend({'Random BPM offset','BPM support offset'})
        plt.set(gca,'xlim',xLim,'box','on')
        plt.ylabel(yLabOffStr[nDim])
        if nDim==3:
            plt.xlabel('$s$ [m]')
        ax[3*(nDim-1)+1,2]=plt.subplot(12,2,2*4*(nDim-1)+ [2 4]);plt.hold(True)
        pVec=[];legStr=[]
        plt.plot(s,1E6*rollSupportLine[nDim,:],Color=tmpCol[1,:])
        plt.plot(sPos[magOrds-1],1E6*rollMagSupport[nDim,:],'D',Color=tmpCol[1,:])
        pVec.append(plt.plot([-2, -1],[0, 0],'-D',Color=tmpCol[1,:]))
        legStr.append('Overall support structure')
        pVec.append(plt.plot(sPos[magOrds-1],1E6*rollMagInd[nDim,:],'kx',MarkerSize=8))
        legStr.append('Individual Magnet')
        for type in ['Section','Plinth','Girder']:
            if hasattr(SuppOrds,type):
                for i in range(SuppOrds.(type).shape[1]):
                    if diff(findspos(SC.RING,SuppOrds.(type)[:,i]))<0:
                        for nCase in range(2):
                            if nCase==1:
                                splot  = findspos(SC.RING,[SuppOrds.(type)[0,i], len(SC.RING)])
                                sint   = [findspos(SC.RING,SuppOrds.(type)[0,i]),findspos(SC.RING,SuppOrds.(type)[1,i])+C]
                            else:
                                splot  = findspos(SC.RING,[1, SuppOrds.(type)[1,i]])
                                sint   = [-findspos(SC.RING,SuppOrds.(type)[1,i]),findspos(SC.RING,SuppOrds.(type)[1,i])]
                            rollInt = np.interp(splot,[sint[0], sint[1]],SuppRoll.(type)[nDim,i]*[1, 1])
                            plt.plot(splot,1E6*rollInt,lineSpec.(type){:})
                    else:
                        plt.plot(findspos(SC.RING,SuppOrds.(type)[:,i]),1E6*SuppRoll.(type)[nDim,i]*[1, 1],lineSpec.(type){:})
                pVec.append(plt.plot([-2, -1],[0, 0],lineSpec.(type){:}))
                legStr.append('Individual '+type)
        plt.legend(pVec,legStr)
        plt.set(gca,'xlim',xLim,'box','on','YAxisLocation','right')#,'XTickLabel',''
        plt.ylabel(yLabRollStr[nDim])
        plt.title(titlteRollStr[nDim])
        ax[3*(nDim-1)+2,2]=plt.subplot(12,2,2*4*(nDim-1)+ 6);plt.hold(True)
        plt.plot(sPos[magOrds-1],1E6*rollMagTot[nDim,:],'kO-')
        plt.legend('Overall magnet roll')
        plt.set(gca,'xlim',xLim,'box','on','YAxisLocation','right')#,'XTickLabel',''
        plt.ylabel(yLabRollStr[nDim])
        ax[3*(nDim-1)+3,2]=plt.subplot(12,2,2*4*(nDim-1)+ 8);plt.hold(True)
        plt.plot(sBPM,1E6*rollBPM[:,nDim],'O',Color=tmpCol[2,:],MarkerSize=6)
        plt.plot(sBPM,1E6*rollBPMStruct[:,nDim],'-',Color=tmpCol[2,:])
        plt.legend({'Random BPM roll','BPM support roll'})
        plt.set(gca,'xlim',xLim,'box','on','YAxisLocation','right')
        plt.ylabel(yLabRollStr[nDim])
        if nDim==3:
            plt.xlabel('$s$ [m]')
    plt.linkaxes(ax,'x')
    for nDim in range(3):
        for nAx in range(3):
            for n in range(2):
                plt.set(ax[nAx+3*(nDim-1),n],'Position',plt.get(ax[nAx+3*(nDim-1),n],'Position') - ((nDim-1)-0.4*(nAx-1))*[0, shiftAxes, 0, 0])
    plt.set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
    plt.set(findall(gcf,'-property','Interpreter'),'Interpreter','latex')
    plt.set(findall(gcf,'-property','FontSize'),'FontSize',fontSize)
    plt.set(gcf,'color','w')
    plt.draw()