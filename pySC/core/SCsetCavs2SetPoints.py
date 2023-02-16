import numpy as np

from pySC.core.SCupdateCAVs import SCupdateCAVs


def SCsetCavs2SetPoints(SC, CAVords, type, setpoints, method='abs'):
    setpoint_str = f"{type}SetPoint"
    valid_methods = ("abs", "rel", "add")
    if method not in valid_methods:
        raise ValueError(f'Unsupported setpoint method: {method}. Allowed options are: {valid_methods}.')
    if len(setpoints) == 1:
        setpoints = np.ones(len(CAVords))*setpoints
    for i, ord in enumerate(CAVords):
        new_setpoint = setpoints[i]
        if method == 'rel':
            new_setpoint *= getattr(SC.RING[ord], setpoint_str)
        if method == 'add':
            new_setpoint += getattr(SC.RING[ord], setpoint_str)
        setattr(SC.RING[ord], setpoint_str, new_setpoint)
    SC = SCupdateCAVs(SC, CAVords)
    return SC

