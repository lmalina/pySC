import numpy as np


def SCgetCMSetPoints(SC,CMords,nDim):
    setpoints = np.nan*np.ones(len(CMords))
    for idx in range(len(CMords)):
        if SC.RING[CMords[idx]].PassMethod == 'CorrectorPass':
            normBy = [1,1]
        else:
            normBy = [-1,1]*SC.RING[CMords[idx]].Length # positive setpoint -> positive kick -> negative horizontal field
        if nDim==1:
            setpoints[idx] = SC.RING[CMords[idx]].SetPointB[0] * normBy[nDim]
        else:
            setpoints[idx] = SC.RING[CMords[idx]].SetPointA[0] * normBy[nDim]
    return setpoints
# End
