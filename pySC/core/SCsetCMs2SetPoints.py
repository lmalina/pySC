import numpy as np

from pySC.core.SCupdateMagnets import SCupdateMagnets


def SCsetCMs2SetPoints(SC, CMords, setpoints, nDim, method='abs'):
    if len(setpoints) == 1:
        setpoints = np.repeat(setpoints, len(CMords))
    i = 0
    for ord in CMords:
        curAB = np.array([SC.RING[ord].SetPointB, SC.RING[ord].SetPointA]).T
        if SC.RING[ord].PassMethod == 'CorrectorPass':
            normBy = np.array([1, 1])
        else:
            normBy = np.array([-1, 1]) * SC.RING[ord].Length  # positive setpoint -> positive kick -> negative horizontal field
        if method == 'abs':
            setpoints[i] = setpoints[i]
        elif method == 'rel':
            setpoints[i] = setpoints[i] * curAB[0, nDim] * normBy[nDim]
        elif method == 'add':
            setpoints[i] = setpoints[i] + curAB[0, nDim] * normBy[nDim]
        else:
            raise ValueError('Unsupported method: ''%s''. Allowed are ''abs'',''rel'' and ''add''.' % method)
        if hasattr(SC.RING[ord], 'CMlimit') and abs(setpoints[i]) > abs(SC.RING[ord].CMlimit[nDim]):
            print('CM (ord: %d / dim: %d) is clipping' % (ord, nDim))
            setpoints[i] = np.sign(setpoints[i]) * SC.RING[ord].CMlimit[nDim]
        if nDim == 1:
            SC.RING[ord].SetPointB[0] = setpoints[i] / normBy[nDim]
        else:
            SC.RING[ord].SetPointA[0] = setpoints[i] / normBy[nDim]
        i = i + 1
    SC = SCupdateMagnets(SC, CMords)
    return SC, setpoints
# End
