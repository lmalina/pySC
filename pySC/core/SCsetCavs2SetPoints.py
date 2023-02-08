import numpy as np

from pySC.core.SCupdateCAVs import SCupdateCAVs


def SCsetCavs2SetPoints(SC,CAVords,type,setpoints,mode='abs'):
    if len(setpoints)==1:
        setpoints = np.ones(len(CAVords))*setpoints
    i = 0
    for ord in CAVords:
        if mode == 'abs':
            SC.RING[ord][type+'SetPoint'] = setpoints[i]
        elif mode == 'rel':
            SC.RING[ord][type+'SetPoint'] = setpoints[i] * SC.RING[ord][type+'SetPoint']
        elif mode == 'add':
            SC.RING[ord][type+'SetPoint'] = setpoints[i] + SC.RING[ord][type+'SetPoint']
        else:
            print('Unsupported setpoint type.')
        i = i + 1
    SC = SCupdateCAVs(SC,CAVords)
    return SC

