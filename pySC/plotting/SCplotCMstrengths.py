import numpy as np
import matplotlib.pyplot as plt
from pySC.utils.at_wrapper import findspos


def SCplotCMstrengths(SC):
    f, ax = plt.subplots(nrows=2, num=86)
    CMs = []
    CMval = [[], []]
    s_pos = findspos(SC.RING)
    for nDim in range(2):
        CMs.append(s_pos[SC.ORD.CM[nDim]])
        for ord in SC.ORD.CM[nDim]:
            if SC.RING[ord].PassMethod == 'CorrectorPass':
                CMval[nDim].append(1E6 * SC.RING[ord].KickAngle[nDim])
            else:
                CMval[nDim].append(1E6 * SC.RING[ord].Length *
                                   (SC.RING[ord].SetPointA[0] if nDim else SC.RING[ord].SetPointB[0]))
        ax[0].bar(CMs[nDim], CMval[nDim])
        count, bins_count = np.histogram(CMval[nDim], bins=len(CMval[nDim]))
        cdf = np.cumsum(count / sum(count))
        ax[1].plot(bins_count[1:], cdf)

    ax[0].set_ylabel(r'CM strength [$\mu$rad]')
    ax[0].set_xlabel('s [m]')
    # ax[0].set_title(f'$\\Theta_x={1E6*np.sqrt(np.mean(CMval[1]**2)):.0f}\\mu$rad rms,            '
    #                 f'\n$\\Theta_y={1E6*np.sqrt(np.mean(CMval[2]**2)):.0f}\\mu$rad rms',
    #                 Interpreter='latex')
    ax[1].set_xlabel(r'CM strength [$\mu$rad]')
    ax[1].set_ylabel('CDF')
    ax[1].legend(['Horizontal', 'Vertical'])
    ax[1].set_ylim([0, 1])
    # plt.set(findall(plt.gcf, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex')
    # plt.set(findall(plt.gcf, '-property', 'Interpreter'), 'Interpreter', 'latex')
    # plt.set(findall(plt.gcf, '-property', 'FontSize'), 'FontSize', 18)
    # plt.gcf().color('w')
    plt.show()
