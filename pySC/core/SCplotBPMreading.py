import numpy as np
import matplotlib.pyplot as plt

from pySC.core.SCparticlesIn3D import SCparticlesIn3D


def SCplotBPMreading(SC,B,T):
    ApertureForPLotting = getRingAperture(SC)
    if SC.INJ.trackMode == 'ORB':
        SC.INJ.nTurns     = 1
        SC.INJ.nParticles = 1
    fig = plt.figure(23)
    fig.clf()
    tmpCol = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ylabelStr = ['$\Delta x$ [mm]','$\Delta y$ [mm]']
    legStr = ['Particle trajectories','BPM reading','Aperture']
    sPos = np.array([s.s for s in SC.RING])
    sMax = sPos[-1]
    for nDim in range(2):
        ax = fig.add_subplot(2,1,nDim+1)
        ax.hold(True)
        x = np.reshape([0:SC.INJ.nTurns-1]*sMax+sPos,1,[])
        x = np.repeat(x,SC.INJ.nParticles)
        for nS in range(SC.INJ.nShots):
            M = SCparticlesIn3D(T[:,:,nS],SC.INJ.nParticles)
            y = 1E3*np.squeeze(M[2*nDim,:,:])
            legVec=ax.plot(x,y,'k')
        legVec=legVec[0]
        x    = np.reshape([0:SC.INJ.nTurns-1]*sMax+sPos[SC.ORD.BPM],1,[])
        y    = 1E3*(B[nDim,:])
        legVec[1] = ax.plot(x,y,'rO')
        if ApertureForPLotting != []:
            apS = sPos[ApertureForPLotting.apOrds]
            x = np.reshape([0:SC.INJ.nTurns-1]*sMax+apS,1,[])
            y = 1E3*np.repeat(ApertureForPLotting.apVals[nDim],SC.INJ.nTurns)
            legVec[2]=ax.plot(x,y[0,:],'Color',tmpCol[0],'LineWidth',4)
            ax.plot(x,y[1,:],'Color',tmpCol[0],'LineWidth',4)
        x = np.reshape([0:SC.INJ.nTurns-1]*sMax,1,[])
        for nT in range(SC.INJ.nTurns):
            ax.plot(x[nT]*[1 1],10*[-1 1],'k:')
        ax.set_xlim([0 SC.INJ.nTurns*sPos[-1]])
        ax.set_box('on')
        ax.set_position(ax.get_position()+[0 .07*(nDim-1) 0 0])
        ax.set_ylim(.5*[-1 1])
        ax.set_ylabel(ylabelStr[nDim])
        ax.legend(legVec,legStr[0:len(legVec)])
    ax.set_xlabel('$s$ [m]')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
        item.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))
    fig.set_facecolor('w')
    plt.draw()
    plt.show()
    plt.pause(0.001)
    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

    # TODO translate getRingAperture method
