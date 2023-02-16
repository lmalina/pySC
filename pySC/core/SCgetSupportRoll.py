import numpy as np

from pySC import findspos
from pySC.core.SCgetSupportOffset import SCgetSupportOffset


def SCgetSupportRoll(SC, s):
    s0 = findspos(SC.RING, range(len(SC.RING)))
    roll0 = np.zeros((3, len(s0)))
    lengths = np.zeros(len(SC.RING))
    for n in range(len(SC.RING)):
        lengths[n] = SC.RING[n].Length
    C = sum(lengths)
    off0 = SCgetSupportOffset(SC, s0)
    supportOrds = getSupportOrds(SC)
    for type in ['Section', 'Plinth', 'Girder']:
        if type in supportOrds:
            for nEl in range(len(supportOrds[type])):
                ords = supportOrds[type][nEl]
                if ords[1] - ords[0] > 0:
                    roll0[0, ords[0]:ords[1]] = roll0[0, ords[0]:ords[1]] + SC.RING[ords[0]][type + 'Roll'][0]
                    roll0[1, ords[0]:ords[1]] = (off0[1, ords[1]] - off0[1, ords[0]]) / (s0[ords[1]] - s0[ords[0]])
                    roll0[2, ords[0]:ords[1]] = (off0[0, ords[1]] - off0[0, ords[0]]) / (s0[ords[1]] - s0[ords[0]])
                else:
                    roll0[0, ords[0]:] = roll0[0, ords[0]:] + SC.RING[ords[0]][type + 'Roll'][0]
                    roll0[0, :ords[1]] = roll0[0, :ords[1]] + SC.RING[ords[0]][type + 'Roll'][0]
                    roll0[1, ords[0]:] = (off0[1, ords[1]] - off0[1, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                    roll0[1, :ords[1]] = (off0[1, ords[1]] - off0[1, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                    roll0[2, ords[0]:] = (off0[0, ords[1]] - off0[0, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                    roll0[2, :ords[1]] = (off0[0, ords[1]] - off0[0, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
    b = np.unique(s0)
    roll = np.empty((3, len(s)))
    roll[0, :] = np.interp(s, s0[b], roll0[0, b])
    roll[1, :] = np.interp(s, s0[b], roll0[1, b])
    roll[2, :] = np.interp(s, s0[b], roll0[2, b])
    return roll


def getSupportOrds(SC):
    supportOrds = {}
    for type in ['Section', 'Plinth', 'Girder']:
        if type in SC.ORD:
            for i in range(SC.ORD[type].shape[1]):
                supportOrds[type][i] = SC.ORD[type][:, i]
    return supportOrds

# End
