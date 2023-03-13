import numpy as np
import matplotlib.pyplot as plt
from pySC.at_wrapper import findspos


def SCplotCMstrengths(SC,varargin):
    fieldNames = {'SetPointB','SetPointA'}; # {hor,ver}
    plt.figure(86);plt.clf()
    for nDim in range(2):
        CMs[nDim] = findspos(SC.RING,SC.ORD.CM[nDim])';
        nCM=1;
        for ord in SC.ORD.CM[nDim]:
            if SC.RING[ord].PassMethod == 'CorrectorPass':
                CMval[nDim][nCM] = SC.RING[ord].KickAngle[nDim];
            else:
                CMval[nDim][nCM] = SC.RING[ord][fieldNames[nDim]][1] * SC.RING[ord].Length;
            nCM=nCM+1;
        plt.subplot(2,1,1);plt.hold(True)
        plt.bar(CMs[1],1E6*CMval[1]);
        plt.bar(CMs[2],1E6*CMval[2]);
        plt.ylabel('CM strengh [$\mu$rad]');plt.xlabel('s [m]');
        plt.title([sprintf('$\\Theta_x=%.0f\\mu$rad rms,            ', 1E6*np.sqrt(np.mean(CMval[1]**2))),...
            sprintf('$\\Theta_y=%.0f\\mu$rad rms', 1E6*np.sqrt(np.mean(CMval[2]**2)))],'Interpreter','latex')
        plt.gca().box('on')
        plt.subplot(2,1,2);plt.hold(True)
        [ah,bh]= plt.hist(1E6 * (CMval[1]), length(CMval[1]));
        [av,bv]= plt.hist(1E6 * (CMval[2]), length(CMval[2]));
        plt.stairs(bh,np.cumsum(ah)/sum(ah),'LineWidth',2);
        plt.stairs(bv,np.cumsum(av)/sum(av),'LineWidth',2);
        plt.xlabel('CM strengh [$\mu$rad]');plt.ylabel('CDF');
        plt.legend({'Horizontal','Vertical'})
        plt.gca().ylim([0 1]).box('on')
        plt.set(findall(plt.gcf, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
        plt.set(findall(plt.gcf, '-property', 'Interpreter'), 'Interpreter', 'latex');
        plt.set(findall(plt.gcf, '-property', 'FontSize'), 'FontSize', 18);
        plt.gcf().color('w');
        plt.draw()
# End
 
