import numpy as np
import matplotlib.pyplot as plt

def SCregisterCAVs(SC,CAVords,varargin):
    if 'ORD' in SC and 'Cavity' in SC['ORD']:
        SC['ORD']['Cavity'] = np.sort(np.unique(np.concatenate((SC['ORD']['Cavity'],CAVords))))
    else:
        SC['ORD']['Cavity'] = np.sort(CAVords)
    fields = ['Voltage','Frequency','TimeLag']
    for ord in CAVords:
        for field in fields:
            SC['RING'][ord][field+'SetPoint'] = SC['RING'][ord][field]
            SC['RING'][ord][field+'Offset'] = 0
            SC['RING'][ord][field+'CalError'] = 0
        if len(varargin) > 0:
            for i in range(0,len(varargin),2):
                SC['SIG']['RF'][ord][varargin[i]] = varargin[i+1]
    return SC
# End
# Test

# SC = SCregisterCAVs(SC,CAVords)
# End
