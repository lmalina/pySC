import numpy as np


def SCregisterMagnets(SC,MAGords,**varargin):
    keywords = ['HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf']  # TODO MasterOf should be np.array
    nvpairs = {key: value for key, value in varargin.items() if key not in keywords}
    if "Mag" not in SC.SIG.keys():
        SC.SIG["Mag"] = dict()
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
        SC = setOptional(SC,ord,MAGords,**varargin)
        SC['SIG']['Mag'][ord] = varargin

    return storeOrds(SC, MAGords, varargin)



def setOptional(SC,ord,MAGords,**varargin):
    if 'CF' in varargin.keys():
        SC.RING[ord].CombinedFunction = 1
    if "HCM" in varargin.keys() or "VCM" in varargin.keys():
        SC.RING[ord].CMlimit = np.zeros(2)
    if 'HCM' in varargin.keys():
        SC.RING[ord].CMlimit[0] = varargin["HCM"]
    if 'VCM' in varargin.keys():
        SC.RING[ord].CMlimit[1] = varargin['VCM']
    if 'SkewQuad' in varargin.keys():
        SC.RING[ord].SkewQuadLimit = varargin['SkewQuad']
    if 'MasterOf' in varargin.keys():
        raise NotImplementedError("Impement it first!")
        # SC.RING[ord].MasterOf = varargin['MasterOf'][:, ord==MAGords].T
    return SC

def storeOrds(SC,MAGords,varargin):
    if 'ORD' in SC and 'Magnet' in SC['ORD']:   # TODO unify with Mag in SC.SIG
        SC.ORD.Magnet = np.sort(np.unique(np.concatenate((SC.ORD.Magnet, MAGords))))
    else:
        SC['ORD']['Magnet'] = MAGords[:]
    if 'SkewQuad' in varargin.keys():
        if 'SkewQuad' in SC.ORD:
            SC.ORD.SkewQuad = np.sort(np.unique(np.concatenate((SC.ORD.SkewQuad,MAGords))))
        else:
            SC.ORD.SkewQuad = MAGords[:]
    if ('HCM' in varargin.keys() or 'VCM' in varargin.keys()) and "CM" not in SC.ORD.keys():
        SC.ORD.CM = [np.zeros(0), np.zeros(0)]
    if 'HCM' in varargin.keys():
        SC.ORD.CM[0] = np.sort(np.unique(np.concatenate((SC.ORD.CM[0],MAGords))))
    if 'VCM' in varargin:
        SC.ORD.CM[1] = np.sort(np.unique(np.concatenate((SC.ORD.CM[1],MAGords))))
    return SC

# End
