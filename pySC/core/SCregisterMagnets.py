import numpy as np


def SCregisterMagnets(SC, MAGords, **kwargs):
    keywords = ['HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf']  # TODO MasterOf should be np.array
    nvpairs = {key: value for key, value in kwargs.items() if key not in keywords}
    if "Mag" not in SC.SIG.keys():
        SC.SIG.Mag = dict()
    for ord in MAGords:
        SC.RING[ord].NomPolynomB = SC.RING[ord].PolynomB
        SC.RING[ord].NomPolynomA = SC.RING[ord].PolynomA
        SC.RING[ord].SetPointB = SC.RING[ord].PolynomB
        SC.RING[ord].SetPointA = SC.RING[ord].PolynomA
        SC.RING[ord].CalErrorB = np.zeros(len(SC.RING[ord].PolynomB))
        SC.RING[ord].CalErrorA = np.zeros(len(SC.RING[ord].PolynomA))
        SC.RING[ord].MagnetOffset  = np.zeros(3)
        SC.RING[ord].SupportOffset = np.zeros(3)
        SC.RING[ord].MagnetRoll  = np.zeros(3)
        SC.RING[ord].SupportRoll = np.zeros(3)
        SC.RING[ord].T1 = np.zeros(6)
        SC.RING[ord].T2 = np.zeros(6)
        SC = setOptional(SC, ord, MAGords, **kwargs)
        SC.SIG.Mag[ord] = nvpairs

    return storeOrds(SC, MAGords, kwargs)


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
    if 'MasterOf' in kwargs.keys():
        raise NotImplementedError("Impement it first!")
        # SC.RING[ord].MasterOf = varargin['MasterOf'][:, ord==MAGords].T
    return SC

def storeOrds(SC, MAGords, kwargs):
    if 'Magnet' in SC.ORD:   # TODO unify with Mag in SC.SIG
        SC.ORD.Magnet = np.sort(np.unique(np.concatenate((SC.ORD.Magnet, MAGords))))
    else:
        SC.ORD.Magnet = MAGords[:]
    if 'SkewQuad' in kwargs.keys():
        if 'SkewQuad' in SC.ORD:
            SC.ORD.SkewQuad = np.sort(np.unique(np.concatenate((SC.ORD.SkewQuad,MAGords))))
        else:
            SC.ORD.SkewQuad = MAGords[:]
    if ('HCM' in kwargs.keys() or 'VCM' in kwargs.keys()) and "CM" not in SC.ORD.keys():
        SC.ORD.CM = [np.zeros(0), np.zeros(0)]
    if 'HCM' in kwargs.keys():
        SC.ORD.CM[0] = np.sort(np.unique(np.concatenate((SC.ORD.CM[0],MAGords))))
    if 'VCM' in kwargs:
        SC.ORD.CM[1] = np.sort(np.unique(np.concatenate((SC.ORD.CM[1],MAGords))))
    return SC
