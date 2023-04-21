import at
import at.plot
import matplotlib.pyplot as plt
import numpy as np


def SCplotSupport(SC,
                  fontSize: int = 8,
                  shiftAxis: float = 0.03,
                  xLim=None):
    """
    SCplotSupport
    =============

    NAME
    ----
    SCplotSupport - Plots the offset and rolls of magnets, the support structure and BPMs

    SYNOPSIS
    --------
    `SCplotSupport(SC)`

    DESCRIPTION
    -----------
    This function plots the overall offsets [dx,dy,dz] and rolls [az,ax,ay] of all magnets and BPMs,
    as well as the individual contributions from different support structures (if registered).
    Please note that the apperance of the figure significanlty depends on the lattice (magnitude of
    errors or lattice size) and the used computer (screen size, Matlab version). The user might have
    to adjust plot apperance properties.

    INPUTS
    ------
    `SC`:: SC base structure


    OPTIONS
    -------
    The following options can be given as name/value-pairs:

    `'fontSize'` (12)::
      Figure font size.
    `'shiftAxes'` (0.03)::
      Axes are reanranged for grouping. Depending on screen resolution this value may be adjusted.
    `'xLim'` ([0 findspos(SC.RING,length(SC.RING)+1)])::
      Plot limits.

    SEE ALSO
    --------
    *SCregisterSupport*, *SCgetSupportOffset*, *SCgetSupportRoll*, *SCupdateSupport*
    """

    if not hasattr(SC.ORD, 'Magnet'):
        raise IndexError('Magnets must be registered. Use ''SCregisterMagnets''.')

    if not hasattr(SC.ORD, 'BPM'):
        raise IndexError('BPM must be registered. Use ''SCregisterBPMs''.')

    if fontSize:
        # adjust font size
        plt.rcParams.update({'font.size': fontSize})

    # get ring lattice
    ring = SC.RING

    # Get s - positions along the lattice
    allind = np.array(range(len(ring) + 1))
    sPos = at.get_s_pos(ring, allind)
    C = sPos[-1]
    if not xLim:
        xLim = (0, C)

    s = np.linspace(xLim[0], xLim[1], 100 * int((xLim[1]-xLim[0])/2))  # s locations to compute

    # default xLim
    if xLim:
        ring.s_range = xLim
    else:
        ring.s_range = [0, C]

    # Magnet offsets and rolls
    offSupportLine, rollSupportLine = SC.support_offset_and_roll(s)
    magOrds=[]
    offMagSupport=[]
    rollMagSupport=[]
    offMagInd=[]
    rollMagInd=[]
    offMagTot=[]
    rollMagTot=[]
    for count, ord in enumerate(SC.ORD.Magnet):
        if sPos[ord]>=xLim[0] and sPos[ord]<=xLim[1]:
            magOrds.append(ord)  # Magnets in range
            offMagSupport.append(SC.RING[ord].SupportOffset)  # Support structure offset
            rollMagSupport.append(SC.RING[ord].SupportRoll)  # Support structure roll
            offMagInd.append(SC.RING[ord].MagnetOffset)  # Get individual magnet offset
            rollMagInd.append(SC.RING[ord].MagnetRoll)  # Get individual magnet roll
            offMagTot.append(SC.RING[ord].T2[[0, 2, 5]])  # Get overall magnet offset
            rollMagTot.append(SC.RING[ord].MagnetRoll+SC.RING[ord].SupportRoll)  # Get overall magnet roll

    offMagSupport = np.array(offMagSupport).transpose()
    rollMagSupport = np.array(rollMagSupport).transpose()
    offMagInd = np.array(offMagInd).transpose()
    rollMagInd = np.array(rollMagInd).transpose()
    offMagTot = np.array(offMagTot).transpose()
    rollMagTot = np.array(rollMagTot).transpose()


    # Loop over individual support structure types
    datadict = {'Ords': [], 'Off': {'a': [], 'b': []}, 'Roll': []}
    supdict = {'Section': datadict, 'Plinth': datadict, 'Girder': datadict}
    for sup_type in supdict.keys():
        if hasattr(SC.ORD, sup_type):  # Check if support structure is registered
            for ordPair in SC.ORD[sup_type].transpose():
                if len(ordPair) != 0:
                    if (sPos[ordPair[0]] >= xLim[0] and sPos[ordPair[0]] <= xLim[1]) or \
                        (sPos[ordPair[1]] >= xLim[0] and sPos[ordPair[1]] <= xLim[1]):
                        # Structures in range
                        supdict[sup_type]['Ords'].append(ordPair)
                        # Get girder start and ending offsets
                        supdict[sup_type]['Off']['a'].append(SC.RING[ordPair[0]].__dict__[sup_type + 'Offset'])
                        supdict[sup_type]['Off']['b'].append(SC.RING[ordPair[1]].__dict__[sup_type + 'Offset'])
                        # Get girder rolls
                        supdict[sup_type]['Roll'].append(SC.RING[ordPair[0]].__dict__[sup_type + 'Roll'])

    # Find BPMs within plotting range
    # BPM offsets
    BPMords=[]
    sBPM=[]
    offBPM=[]
    offBPMStruct=[]
    rollBPM=[]
    rollBPMStruct=[]
    for count, ord in enumerate(SC.ORD.BPM):
        if sPos[ord] >= xLim[0] and sPos[ord] <= xLim[1]:
            BPMords.append(ord)  # BPMs in range
            sBPM.append(sPos[ord])
            offBPM.append(np.append(SC.RING[ord].Offset, 0.0))  # append 3x1 array x,y,s Longitudinal offsets not supported for BPMs
            offBPMStruct.append(np.append(SC.RING[ord].SupportOffset, 0.0))  # append 3x1 array x,y,s Longitudinal offsets not supported for BPMs
            rollBPM.append(np.append(SC.RING[ord].Roll, [0.0, 0.0]))  # append 3x1 array roll,pitch,yaw, Pitch and yaw not supported for BPMs
            rollBPMStruct.append(np.append(SC.RING[ord].SupportRoll, [0.0, 0.0]))  # append 3x1 array roll,pitch,yaw, Pitch and yaw not supported for BPMs

    BPMords = np.array(BPMords).transpose()
    sBPM = np.array(sBPM).transpose()
    offBPM = np.array(offBPM).transpose()
    rollBPM = np.array(rollBPM).transpose()
    offBPMStruct = np.array(offBPMStruct).transpose()
    rollBPMStruct = np.array(rollBPMStruct).transpose()

    # create figure
    fig, ax = plt.subplots(nrows=9, ncols=2, num=1213, sharex=True, figsize=(10, 15))

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    tmpCol = plt.rcParams['axes.prop_cycle'].by_key()['color']

    yLabOffStr = ['$\Delta x$ [$\mu$m]', '$\Delta y$ [$\mu$m]', '$\Delta z$ [$\mu$m]']
    yLabRollStr = ['$a_z$ [$\mu$rad]', '$a_x$ [$\mu$rad]', '$a_y$ [$\mu$rad]']
    titlteOffStr = ['Horizontal Offsets', 'Vertical Offsets', 'Longitudinal Offsets']
    titlteRollStr = ['Roll (roll around z-axis)', 'Pitch (roll around x-axis)', 'Yaw (roll around y-axis)']

    lineSpec={}
    lineSpec['Plinth'] = {'color': 'r', 'linewidth': 4}
    lineSpec['Section'] = {'color': tmpCol[6], 'linewidth': 2, 'linestyle': ':'}
    lineSpec['Girder'] = {'color': tmpCol[4], 'linewidth': 4}

    # ax.plot(s, offSupportLine[0, :])
    # plt.show()

    for nDim in range(3):
        presax = ax[3*nDim, 0]
        presax.stairs(1e6*offSupportLine[nDim, :-1], s)
        presax.plot(sPos[magOrds], 1e6 * offMagSupport[nDim, :], 'D', color=tmpCol[0])
        presax.plot([-2, -1], [0, 0], 'D', color=tmpCol[0], label='Overall supports')
        presax.plot(sPos[magOrds], 1e6 * offMagInd[nDim, :], 'kx', ms=8,
                    label='Individual Magnet')
        # loop over support structure
        for sup_type in supdict.keys():
            for i in range(len(supdict[sup_type]['Ords'])): # loop supports
                val = supdict[sup_type]['Ords'][i]

                # check if support structure spans over injection point
                ssup = at.get_s_pos(SC.RING, val)
                if ssup[1] - ssup[0] < 0:
                    # interpolate between last support structure and enf of ring
                    for nCase in range(1):
                        if nCase==0:
                            splot = at.get_s_pos(SC.RING, val[0] + [len(SC.RING)])
                            sint = splot[0]+[at.get_s_pos(SC.RING, val[1])+C]
                        else:
                            splot = at.get_s_pos(SC.RING, [1] + val[1])
                            sint = -splot[-1] + [splot[-1]]
                        offInt = np.interp(sint, [supdict[sup_type]['Off']['a'][i][nDim],
                                                  supdict[sup_type]['Off']['b'][i][nDim]], splot)
                        presax.plot(splot, 1e6*offInt, lineSpec[sup_type])
                else:
                    splot = at.get_s_pos(SC.RING, val)

                    presax.plot(splot, [1e6*supdict[sup_type]['Off']['a'][i][nDim],
                                        1e6*supdict[sup_type]['Off']['b'][i][nDim]],
                                **lineSpec[sup_type])

            presax.plot([-2, -1], [0, 0], label='Individual ' + sup_type)

        # legend and axis stuff
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabOffStr[nDim])
        presax.set_title(titlteOffStr[nDim])

        # plot overall magnet offset
        presax = ax[3*nDim+1, 0]
        presax.plot(sPos[magOrds], 1e6*offMagTot[nDim, :], 'ko-', label='Overall magnet offset')
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabOffStr[nDim])

        # plot BPM offset
        presax = ax[3 * nDim + 2, 0]
        # plot random offset
        presax.plot(sBPM, 1e6 * offBPM[nDim, :], 'o', color=tmpCol[1], ms=6, label='Random BPM offset')
        # plot BPM offset from support structure
        presax.plot(sBPM, 1e6 * offBPMStruct[nDim, :], '-', color=tmpCol[1], label='BPM support offset')
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabOffStr[nDim])
        if nDim == 2:
            presax.set_xlabel('$s$ [m]')


        # plot individual roll contribution
        presax = ax[3 * nDim, 1]
        presax.stairs(1e6 * rollSupportLine[nDim, :-1], s)
        presax.plot(sPos[magOrds], 1e6 * rollMagSupport[nDim, :], 'D', color=tmpCol[0])
        presax.plot([-2, -1], [0, 0], 'D', color=tmpCol[0], label='Overall supports')
        presax.plot(sPos[magOrds], 1e6 * rollMagInd[nDim, :], 'kx', ms=8,
                    label='Individual Magnet')
        # loop over support structure
        for sup_type in supdict.keys():
            for i in range(len(supdict[sup_type]['Ords'])):  # loop supports
                val = supdict[sup_type]['Ords'][i]
                # check if support structure spans over injection point
                ssup = at.get_s_pos(SC.RING, val)
                if ssup[1] - ssup[0] < 0:
                    # interpolate between last support structure and enf of ring
                    for nCase in range(1):
                        if nCase == 0:
                            splot = at.get_s_pos(SC.RING, val[0] + [len(SC.RING)])
                            sint = splot[0] + [at.get_s_pos(SC.RING, val[1]) + C]
                        else:
                            splot = at.get_s_pos(SC.RING, [1] + val[1])
                            sint = -splot[-1] + [splot[-1]]
                        offInt = np.interp(sint, [supdict[sup_type]['Roll'][i][nDim],
                                                  supdict[sup_type]['Roll'][i][nDim]], splot)
                        presax.plot(splot, 1e6(offInt), lineSpec[sup_type])
                else:
                    splot = at.get_s_pos(SC.RING, val)
                    presax.plot(splot, [1e6 * supdict[sup_type]['Roll'][i][nDim],
                                        1e6 * supdict[sup_type]['Roll'][i][nDim]],
                                **lineSpec[sup_type])

            presax.plot([-2, -1], [0, 0], label='Individual ' + sup_type)

        # legend and axis stuff
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabRollStr[nDim])
        presax.set_title(titlteRollStr[nDim])

        # plot overall magnet offset
        presax = ax[3 * nDim + 1, 1]
        presax.plot(sPos[magOrds], 1e6 * rollMagTot[nDim, :], 'ko-',label='Overall magnet roll')
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabRollStr[nDim])

        # plot BPM offset
        presax = ax[3 * nDim + 2, 1]
        # plot random offset
        presax.plot(sBPM, 1e6 * rollBPM[nDim, :], 'o', color=tmpCol[1], ms=6, label='Random BPM roll')
        # plot BPM offset from support structure
        presax.plot(sBPM, 1e6 * rollBPMStruct[nDim, :], '-', color=tmpCol[1], label='BPM support roll')
        presax.legend()
        presax.set_xlim(xLim)
        presax.set_ylabel(yLabRollStr[nDim])
        if nDim == 2:
            presax.set_xlabel('$s$ [m]')

    return



    pass

class ord:
    CM=[]
    BPM=[]
    SkewQuad=[]


if __name__=='__main__':

    file='/machfs/liuzzo/EBS/beamdyn/matlab/optics/sr/S28F_all_BM_27Mar2022/betamodel.mat'
    lattice_variable_name = 'betamodel'
    # file = '../scfodo.mat'
    # lattice_variable_name = 'r'
    from pySC.classes import SimulatedComissioning

    RING = at.load_lattice(file, mat_key=lattice_variable_name)

    sc = SimulatedComissioning(RING)


    sc.IDEALRING = at.load_lattice(file, mat_key=lattice_variable_name)
    sc.ORD.Magnet = sc.IDEALRING.get_refpts(at.Quadrupole)
    sc.ORD.BPM = sc.IDEALRING.get_refpts(at.Monitor)
    sc.ORD.CM.append(sc.IDEALRING.get_refpts('S[HFDJI]*'))
    sc.ORD.CM.append(sc.IDEALRING.get_refpts('S[HFDJI]*'))
    sc.ORD.SkewQuad = sc.IDEALRING.get_refpts(at.Quadrupole)

    # test simple input
    SCplotSupport(sc)

    plt.show()

    pass
