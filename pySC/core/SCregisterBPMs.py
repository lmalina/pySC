import numpy as np
import matplotlib.pyplot as plt


def SCregisterBPMs(SC,BPMords,varargin):
    if 'ORD' in SC and 'BPM' in SC['ORD']:
        SC['ORD']['BPM'] = np.sort(np.unique(np.concatenate((SC['ORD']['BPM'],BPMords))))
    else:
        SC['ORD']['BPM'] = BPMords
    for ord in BPMords:
        if len(varargin) > 0:
            for i in range(0,len(varargin),2):
                SC['SIG']['BPM'][ord][varargin[i]] = varargin[i+1]
        SC['RING'][ord]['Noise'] = np.zeros(2)
        SC['RING'][ord]['NoiseCO'] = np.zeros(2)
        SC['RING'][ord]['Offset'] = np.zeros(2)
        SC['RING'][ord]['SupportOffset'] = np.zeros(2)
        SC['RING'][ord]['Roll'] = 0
        SC['RING'][ord]['SupportRoll'] = 0
        SC['RING'][ord]['CalError'] = np.zeros(2)
        SC['RING'][ord]['SumError'] = 0
    return SC
# End
