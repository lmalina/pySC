import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pySC.at_wrapper import twissline, atlinopt

def SCplotLattice(SC, transferLine=0, nSectors=1, oList=[], plotIdealRing=1, sRange=[], plotMagNames=0, fontSize=16):
    sPos = np.cumsum(np.array([el.Length for el in SC.RING]))
    if len(oList) == 0:
        oList = np.where(sPos <= (sPos[-1] / nSectors))[0]
    if len(sRange) > 0:
        oList = np.intersect1d(np.where(sPos >= sRange[0]), np.where(sPos <= sRange[1]))
    if transferLine:
        if not hasattr(SC.RING[0], 'TD'):
            print('Transfer line lattice did not contain initial parameters needed for beta function calculation!')
            beta = np.nan * np.ones((2, len(oList)))
            disp = np.nan * np.ones((1, len(oList)))
        else:
            TD = twissline(SC.IDEALRING, 0, SC.IDEALRING[0].TD, oList, 'chrom', 1E-8)
            beta = np.reshape([TD.beta], (2, len(oList)))
            disp = np.reshape([TD.Dispersion], (4, len(oList)))
    else:
        if plotIdealRing:
            ld, _, _ = atlinopt(SC.IDEALRING, 1e-3, oList)
        else:
            ld, _, _ = atlinopt(SC.RING, 1e-3, oList)
        beta = np.reshape([ld.beta], (2, len(oList)))
        disp = np.reshape([ld.Dispersion], (4, len(oList)))
    DIP = []
    QUAD = []
    SEXT = []
    OCT = []
    SKEW = []
    for ord in oList:
        if hasattr(SC.RING[ord], 'BendingAngle') and SC.RING[ord].BendingAngle != 0:
            DIP.append(ord)
        if hasattr(SC.RING[ord], 'NomPolynomB'):
            if any(np.where(SC.RING[ord].NomPolynomB == 2)):
                QUAD.append(ord)
            if any(np.where(SC.RING[ord].NomPolynomB == 3)):
                SEXT.append(ord)
            if any(np.where(SC.RING[ord].NomPolynomB == 4)):
                OCT.append(ord)
    if hasattr(SC, 'ORD') and hasattr(SC.ORD, 'SkewQuad'):
        SKEW = np.intersect1d(SC.ORD.SkewQuad, oList)
    ApertureForPlotting = {'apOrds': [], 'apVals': {0: [], 1: []}}
    for nEl in oList:
        if hasattr(SC.RING[nEl], 'EApertures') or hasattr(SC.RING[nEl], 'RApertures'):
            ApertureForPlotting['apOrds'].append(nEl)
            for nDim in range(2):
                if hasattr(SC.RING[nEl], 'EApertures'):
                    ApertureForPlotting['apVals'][nDim].append(SC.RING[nEl].EApertures[nDim] * np.array([-1, 1]))
                else:
                    ApertureForPlotting['apVals'][nDim].append(SC.RING[nEl].RApertures[2 * (nDim - 1) + [1, 2]])
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(sPos[oList], 1E2 * disp[1, :], color='k', linewidth=2)
    ax[0].set_ylabel('$\eta_x$ [cm]')
    ax[0].set_ylim(ax[0].get_ylim() * np.array([1, 1.5]))
    ax[0].set_xlim([sPos[min(oList)], sPos[max(oList)]])
    ax[0].set_box(True)
    ax[0].set_title('Beta Functions and Dispersion')
    ax[0].legend(['Hor. Beta', 'Ver. Beta', 'Hor. Disp.'], loc='N', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    ax[1].set_title('Aperture and Magnets')
    ax[1].set_ylabel('Aperture [mm]')
    ax[1].set_xlim([sPos[min(oList)], sPos[max(oList)]])
    ax[1].set_box(True)
    if len(ApertureForPlotting['apOrds']) > 0:
        apS = sPos[ApertureForPlotting['apOrds']]
        lStyle = ['-', ':']
        for nDim in range(2):
            ax[1].plot(apS, 1E3 * np.array(ApertureForPlotting['apVals'][nDim])[0, :],
                       color=ax[0].get_lines()[nDim].get_color(), linewidth=4, linestyle=lStyle[nDim])
            ax[1].plot(apS, 1E3 * np.array(ApertureForPlotting['apVals'][nDim])[1, :],
                       color=ax[0].get_lines()[nDim].get_color(), linewidth=4, linestyle=lStyle[nDim])
    scale = 1E3 * max([max(np.abs(np.array(ApertureForPlotting['apVals'][0]).flatten())),
                       max(np.abs(np.array(ApertureForPlotting['apVals'][1]).flatten()))])
    if len(scale) == 0:
        scale = 10
    for nM in range(len(OCT)):
        ax[1].add_patch(patches.Rectangle((sPos[OCT[nM]], -scale), sPos[OCT[nM] + 1] - sPos[OCT[nM]], scale,
                                          facecolor=ax[0].get_lines()[5].get_color()))
        if plotMagNames:
            ax[1].text(sPos[OCT[nM]], -(1.2 + 0.1 * (-1) ** (nM)) * scale, SC.RING[OCT[nM]].FamName)
    for nM in range(len(SEXT)):
        ax[1].add_patch(patches.Rectangle((sPos[SEXT[nM]], 0), sPos[SEXT[nM] + 1] - sPos[SEXT[nM]], scale,
                                          facecolor=ax[0].get_lines()[4].get_color()))
        if plotMagNames:
            ax[1].text(sPos[SEXT[nM]], (1.2 + 0.1 * (-1) ** (nM)) * scale, SC.RING[SEXT[nM]].FamName)
    for nM in range(len(DIP)):
        ax[1].add_patch(
            patches.Rectangle((sPos[DIP[nM]], 0), sPos[DIP[nM] + 1] - sPos[DIP[nM]], scale / 2, facecolor='k'))
    for nM in range(len(QUAD)):
        ax[1].add_patch(patches.Rectangle((sPos[QUAD[nM]], -scale / 2), sPos[QUAD[nM] + 1] - sPos[QUAD[nM]], scale / 2,
                                          facecolor=ax[0].get_lines()[2].get_color()))
        if plotMagNames:
            ax[1].text(sPos[QUAD[nM]], (-.7 + 0.1 * (-1) ** (nM)) * scale, SC.RING[QUAD[nM]].FamName)
    ax[1].set_ylim(ax[1].get_ylim() * np.array([1, 1.3]))
    ax[1].legend(['Hor. Ap.', 'Ver. Ap.', 'Oct', 'Sext', 'Dip', 'Quad'], loc='N', bbox_to_anchor=(0.5, 1.05), ncol=6,
                 frameon=False)
    if hasattr(SC, 'ORD') and hasattr(SC.ORD, 'CM'):
        for nDim in range(2):
            for ord in np.intersect1d(SC.ORD.CM[nDim], oList):
                if SC.RING[ord].PassMethod == 'CorrectorPass':
                    ax[2].bar(sPos[ord], (-1) ** (nDim - 1) * 4, color=ax[0].get_lines()[nDim].get_color(),
                              width=(max(sPos[oList]) - min(sPos[oList])) / 100)
                else:
                    ax[2].add_patch(patches.Rectangle((sPos[ord], 0 - 4 * (nDim - 1)), sPos[ord + 1] - sPos[ord], 4,
                                                      facecolor=ax[0].get_lines()[nDim].get_color()))
    for nM in range(len(SKEW)):
        ax[2].add_patch(patches.Rectangle((sPos[SKEW[nM]], -2), sPos[SKEW[nM] + 1] - sPos[SKEW[nM]], 4,
                                          facecolor=ax[0].get_lines()[4].get_color()))
    if hasattr(SC, 'ORD') and hasattr(SC.ORD, 'BPM'):
        for ord in np.intersect1d(SC.ORD.BPM, oList):
            ax[2].add_patch(
                patches.Rectangle((sPos[ord] - np.diff(sPos[oList]) / 300, -3), np.diff(sPos[oList]) / 150, 6,
                                  facecolor='k'))
    ax[2].set_ylim(ax[2].get_ylim() * np.array([1, 1.3]))
    ax[2].legend(['HCM', 'VCM', 'SKEW', 'BPM'], loc='N', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
    ax[2].set_title('BPMs and CMs')
    ax[2].set_xlabel('$s$ [m]')
    ax[2].set_xlim([sPos[min(oList)], sPos[max(oList)]])
    ax[2].set_box(True)
    ax[2].set_yticklabels([])
    plt.tight_layout()
    plt.show()
    return

# SCplotLattice(SC,transferLine=1,nSectors=4,oList=range(0,len(SC.RING),2),plotIdealRing=0,sRange=[0,100],plotMagNames=1,fontSize=16)
