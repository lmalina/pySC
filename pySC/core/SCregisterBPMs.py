import numpy as np
from pySC.classes import DotDict

def SCregisterBPMs(SC, BPMords, **kwargs):
    if 'ORD' in SC and 'BPM' in SC['ORD']:
        SC['ORD']['BPM'] = np.sort(np.unique(np.concatenate((SC['ORD']['BPM'], BPMords))))
    else:
        SC['ORD']['BPM'] = BPMords[:]
    if "BPM" not in SC.SIG.keys():
        SC.SIG["BPM"] = DotDict()
    for ord in BPMords:
        SC['SIG']['BPM'][ord] = kwargs

        SC['RING'][ord].Noise = np.zeros(2)
        SC['RING'][ord].NoiseCO = np.zeros(2)
        SC['RING'][ord].Offset = np.zeros(2)
        SC['RING'][ord].SupportOffset = np.zeros(2)
        SC['RING'][ord].Roll = 0
        SC['RING'][ord].SupportRoll = 0
        SC['RING'][ord].CalError = np.zeros(2)
        SC['RING'][ord].SumError = 0
    return SC
# End
