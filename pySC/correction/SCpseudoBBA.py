import numpy as np

from pySC.utils.sc_tools import SCrandnc


def SCpseudoBBA(SC, BPMords, MagOrds, postBBAoffset, sigma=2):
    # TODO this looks fishy ... assumes BPMs attached to quads?
    #  at the same time two separate 2D arrays?
    if len(postBBAoffset) == 1:
        postBBAoffset = np.tile(postBBAoffset, (2,np.size(BPMords, axis=1)))
    for nBPM in range(np.size(BPMords, axis=1)):
        for nDim in range(2):
            SC.RING[BPMords[nDim][nBPM]].Offset[nDim] = (SC.RING[MagOrds[nDim][nBPM]].MagnetOffset[nDim]
                                                         + SC.RING[MagOrds[nDim][nBPM]].SupportOffset[nDim]
                                                         - SC.RING[BPMords[nDim][nBPM]].SupportOffset[nDim]
                                                         + postBBAoffset[nDim][nBPM] * SCrandnc(sigma))
    return SC
