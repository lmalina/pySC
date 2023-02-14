import numpy as np


def SCregisterCAVs(SC, CAVords, **kwargs):
    if 'Cavity' in SC.ORD:
        SC.ORD.Cavity = np.sort(np.unique(np.concatenate((SC.ORD.Cavity, CAVords))))
    else:
        SC.ORD.Cavity = np.sort(CAVords)
    fields = ['Voltage', 'Frequency', 'TimeLag']
    if "RF" not in SC.SIG.keys():
        SC.SIG.RF = dict()
    for ord in CAVords:
        for field in fields:
            setattr(SC.RING[ord], f"{field}SetPoint", getattr(SC.RING[ord], field))
            setattr(SC.RING[ord], f"{field}Offset", 0)
            setattr(SC.RING[ord], f"{field}CalError", 0)

        SC.SIG.RF[ord] = kwargs # TODO unify SC.SIG and SC.ORD (Cavity vs RF)
    return SC
