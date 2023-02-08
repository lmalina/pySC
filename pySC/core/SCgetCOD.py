import numpy as np
import matplotlib.pyplot as plt

def SCgetCOD(SC,ords=None,plot=False):
    if ords is None:
        ords = SC.ORD.Magnet
    T = findorbit6(SC.RING,ords)
    if any(np.isnan(T)):
        print('Closed orbit could not be found.')
        COD = np.nan(2,len(ords))
        return COD
    magOffset=np.zeros((2,len(ords)))
    for i,ord in enumerate(ords):
        magOffset[:,i] = SC.RING[ord].T2([1,3])
    COD = T[[1,3],:] - magOffset
    if plot:
        sPos = findspos(SC.RING,ords)
        ylabelStr = ['$\Delta x$ [mm]','$\Delta y$ [mm]']
        plt.figure(784);plt.clf()
        for nDim in range(2):
            ax = plt.subplot(2,1,nDim+1);plt.hold(True)
            plt.plot(sPos,1E3*COD[nDim,:],linewidth=2)
            plt.plot(sPos,1E3*magOffset[nDim,:],linewidth=2)
            plt.plot(sPos,1E3*T[2*nDim,:],linewidth=2)
            plt.legend(['COD in magnets','Magnet offset','Orbit'])
            plt.xlabel('s [m]');
            plt.ylabel(ylabelStr[nDim]);
            plt.gca().set_box(True)
            plt.gca().set_xlim([min(sPos),max(sPos)])
        plt.gcf().set_size_inches(8,12)
        plt.gcf().set_dpi(100)
        plt.gcf().set_facecolor('w')
        plt.gcf().set_edgecolor('w')
        plt.gcf().tight_layout()
        plt.draw()
        plt.pause(0.01)
        plt.gca().set_xlim([min(sPos),max(sPos)])
        plt.draw()
        plt.pause(0.01)
    return COD
# End
 
