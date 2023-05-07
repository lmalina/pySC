import numpy as np
import matplotlib.pyplot as plt
from pySC.at_wrapper import findspos


def SCplotCMstrengths(SC):
    fieldNames = ['SetPointB','SetPointA']; # {hor,ver}
    plt.figure(86)
    CMs=[]
    CMval=[[],[]]

    for nDim in range(2):
        CMs.append(findspos(SC.RING,SC.ORD.CM[nDim]))
        nCM=1;
        for ord in SC.ORD.CM[nDim]:
            if SC.RING[ord].PassMethod == 'CorrectorPass':
                # CMval[nDim][nCM] = SC.RING[ord].KickAngle[nDim];
                CMval[nDim].append(1E6*SC.RING[ord].KickAngle[nDim])
            else:
                # CMval[nDim][nCM] = SC.RING[ord][fieldNames[nDim]][1] * SC.RING[ord].Length;
                CMval[nDim].append(1E6*(SC.RING[ord].SetPointA[0] if nDim else SC.RING[ord].SetPointB[0]) * SC.RING[ord].Length)
            nCM=nCM+1
    
    plt.subplot(2,1,1)
    for nDim in range(2):
        plt.bar(CMs[nDim],CMval[nDim])
    #plt.ylabel('CM strengh [$\mu$rad]');plt.xlabel('s [m]');
    # plt.title([sprintf('$\\Theta_x=%.0f\\mu$rad rms,            ', 1E6*np.sqrt(np.mean(CMval[1]**2))),...
    #     sprintf('$\\Theta_y=%.0f\\mu$rad rms', 1E6*np.sqrt(np.mean(CMval[2]**2)))],'Interpreter','latex')
    # plt.gca().box('on')
    plt.subplot(2,1,2)
    for nDim in range(2):
        count, bins_count = np.histogram(CMval[nDim], bins=len(CMval[nDim]))
        cdf = np.cumsum(count / sum(count))
        plt.plot(bins_count[1:], cdf)

    
        # plt.xlabel('CM strengh [$\mu$rad]');plt.ylabel('CDF');
        # plt.legend({'Horizontal','Vertical'})
        # plt.gca().ylim([0 1]).box('on')
        # plt.set(findall(plt.gcf, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
        # plt.set(findall(plt.gcf, '-property', 'Interpreter'), 'Interpreter', 'latex');
        # plt.set(findall(plt.gcf, '-property', 'FontSize'), 'FontSize', 18);
        # plt.gcf().color('w');
    plt.show()
# End
 
