import numpy as np
import matplotlib.pyplot as plt

def SCrandnc(c,m=1,n=1,normalize=False):
    out = np.random.randn(m,n)
    outindex = np.where(abs(out)>abs(c))
    while len(outindex[0])>0:
        out[outindex] = np.random.randn(len(outindex[0]))
        outindex = np.where(abs(out)>abs(c))
    if normalize:
        print('Not yet implemented.')
    return out

# Test

# c = 1
# m = 100
# n = 100
#
# out = SCrandnc(c,m,n)
#
# plt.hist(out.flatten(),bins=100)
# plt.show()

# End
 
