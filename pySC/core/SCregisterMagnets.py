import numpy as np


def SCregisterMagnets(SC,MAGords,varargin):
    keywords = {'HCM','VCM','CF','SkewQuad','MasterOf'}
    [nvpairs] = getSigmaPairs(keywords,varargin)
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
        SC.RING[ord].T1 = np.zeros(6,1)
        SC.RING[ord].T2 = np.zeros(6,1)
        SC = setOptional(SC,ord,MAGords,varargin)
        for i in range(0,len(nvpairs),2):
            SC.SIG.Mag[ord][nvpairs[i]] = nvpairs[i+1]
    SC = storeOrds(SC,MAGords,varargin)

def getSigmaPairs(keywords,varargin):
    nvpairs=[]
    for n in range(0,len(varargin),2):
        if varargin[n] not in keywords:
            if isinstance(varargin[n+1],list):
                nvpairs.append(varargin[n])
                nvpairs.append(varargin[n+1])
            else:
                nvpairs.append(varargin[n])
                nvpairs.append(varargin[n+1])
    return nvpairs

def setOptional(SC,ord,MAGords,varargin):
    if 'CF' in varargin:
        SC.RING[ord].CombinedFunction = 1
    if 'HCM' in varargin:
        SC.RING[ord].CMlimit[0] = varargin[varargin.index('HCM')+1]
    if 'VCM' in varargin:
        SC.RING[ord].CMlimit[1] = varargin[varargin.index('VCM')+1]
    if 'SkewQuad' in varargin:
        SC.RING[ord].SkewQuadLimit = varargin[varargin.index('SkewQuad')+1]
    if 'MasterOf' in varargin:
        SC.RING[ord].MasterOf = varargin[varargin.index('MasterOf')+1][:,ord==MAGords].T
    return SC

def storeOrds(SC,MAGords,varargin):
    if 'Magnet' in SC.ORD:
        SC.ORD.Magnet = np.sort(np.unique(np.concatenate((SC.ORD.Magnet,MAGords))))
    else:
        SC.ORD.Magnet = MAGords[:]
    if 'SkewQuad' in varargin:
        if 'SkewQuad' in SC.ORD:
            SC.ORD.SkewQuad = np.sort(np.unique(np.concatenate((SC.ORD.SkewQuad,MAGords))))
        else:
            SC.ORD.SkewQuad = MAGords[:]
    if 'HCM' in varargin:
        if 'CM' in SC.ORD:
            SC.ORD.CM[0] = np.sort(np.unique(np.concatenate((SC.ORD.CM[0],MAGords))))
        else:
            SC.ORD.CM[0] = MAGords[:]
    if 'VCM' in varargin:
        if 'CM' in SC.ORD and len(SC.ORD.CM)==2:
            SC.ORD.CM[1] = np.sort(np.unique(np.concatenate((SC.ORD.CM[1],MAGords))))
        else:
            SC.ORD.CM[1] = MAGords[:]
    return SC

# End
