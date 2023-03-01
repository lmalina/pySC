import numpy as np

from pySC.utils.sc_utils import SCrandnc


def SCpseudoBBA(SC, BPMords, MagOrds, postBBAoffset, sigma=2):
    if len(postBBAoffset) == 1:
        postBBAoffset = np.repeat(postBBAoffset, 2 * np.size(BPMords, 1))
    for nBPM in range(np.size(BPMords, 1)):
        for nDim in range(2):
            SC.RING[BPMords[nBPM, nDim]].Offset[nDim] = SC.RING[MagOrds[nBPM, nDim]].MagnetOffset[nDim] + \
                                                        SC.RING[MagOrds[nBPM, nDim]].SupportOffset[nDim] - \
                                                        SC.RING[BPMords[nBPM, nDim]].SupportOffset[nDim] + \
                                                        postBBAoffset[nDim, nBPM] * SCrandnc(sigma)
    return SC
