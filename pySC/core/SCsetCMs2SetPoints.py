import numpy as np

from pySC.core.SCregisterUpdate import SCupdateMagnets


def SCsetCMs2SetPoints(SC, CMords, setpoints, nDim, method='abs'):
    # TODO correct accessing SC.RING.attr.subattr/elements
    valid_methods = ("abs", "rel", "add")
    if method not in valid_methods:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {valid_methods}.')
    if len(setpoints) == 1:
        setpoints = np.repeat(setpoints, len(CMords))
    for i, ord in enumerate(CMords):
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

        if hasattr(SC.RING[ord], 'CMlimit') and abs(setpoints[i]) > abs(SC.RING[ord].CMlimit[nDim]):
            print(f'CM (ord: {ord} / dim: {nDim}) is clipping')
            setpoints[i] = np.sign(setpoints[i]) * SC.RING[ord].CMlimit[nDim]
        if nDim == 1:
            SC.RING[ord].SetPointB[0] = setpoints[i] / normBy[nDim]
        else:
            SC.RING[ord].SetPointA[0] = setpoints[i] / normBy[nDim]
    SC = SCupdateMagnets(SC, CMords)
    return SC, setpoints

