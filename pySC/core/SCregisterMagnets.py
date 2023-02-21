import numpy as np
from pySC.classes import DotDict, SimulatedComissioning
from numpy import ndarray


def SCregisterMagnets(SC: SimulatedComissioning, MAGords: ndarray, **kwargs) -> SimulatedComissioning:
    keywords = ['HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf']  # TODO MasterOf should be np.array
    nvpairs = {key: value for key, value in kwargs.items() if key not in keywords}
    SC.ORD.Magnet = np.unique(np.concatenate((SC.ORD.Magnet, MAGords)))  # TODO unify with Mag in SC.SIG
    if 'SkewQuad' in kwargs.keys():
        SC.ORD.SkewQuad = np.unique(np.concatenate((SC.ORD.SkewQuad, MAGords)))
    if 'HCM' in kwargs.keys():
        SC.ORD.HCM = np.unique(np.concatenate((SC.ORD.HCM, MAGords)))
    if 'VCM' in kwargs.keys():
        SC.ORD.VCM = np.unique(np.concatenate((SC.ORD.VCM, MAGords)))
    for ord in MAGords:
        if ord not in SC.SIG.Mag.keys():
            SC.SIG.Mag[ord] = DotDict()
        SC.SIG.Mag[ord].update(nvpairs)

        SC.RING[ord].NomPolynomB = SC.RING[ord].PolynomB[:]
        SC.RING[ord].NomPolynomA = SC.RING[ord].PolynomA[:]
        SC.RING[ord].SetPointB = SC.RING[ord].PolynomB[:]
        SC.RING[ord].SetPointA = SC.RING[ord].PolynomA[:]
        SC.RING[ord].CalErrorB = np.zeros(len(SC.RING[ord].PolynomB))
        SC.RING[ord].CalErrorA = np.zeros(len(SC.RING[ord].PolynomA))
        SC.RING[ord].MagnetOffset = np.zeros(3)
        SC.RING[ord].SupportOffset = np.zeros(3)
        SC.RING[ord].MagnetRoll = np.zeros(3)
        SC.RING[ord].SupportRoll = np.zeros(3)
        SC.RING[ord].T1 = np.zeros(6)
        SC.RING[ord].T2 = np.zeros(6)
        SC = setOptional(SC, ord, MAGords, **kwargs)
    return SC


def setOptional(SC, ord, MAGords, **kwargs):
    if 'CF' in kwargs.keys():
        SC.RING[ord].CombinedFunction = 1
    if "HCM" in kwargs.keys() or "VCM" in kwargs.keys():
        SC.RING[ord].CMlimit = np.zeros(2)
    if 'HCM' in kwargs.keys():
        SC.RING[ord].CMlimit[0] = kwargs["HCM"]
    if 'VCM' in kwargs.keys():
        SC.RING[ord].CMlimit[1] = kwargs['VCM']
    if 'SkewQuad' in kwargs.keys():
        SC.RING[ord].SkewQuadLimit = kwargs['SkewQuad']
    if 'MasterOf' in kwargs.keys():  # TODO
        SC.RING[ord].MasterOf = kwargs['MasterOf'][:, ord==MAGords].T
    return SC

