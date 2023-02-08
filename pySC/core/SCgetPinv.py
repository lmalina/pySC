import numpy as np
import matplotlib.pyplot as plt

def SCgetPinv(M,N=0,alpha=[],damping=[],plot=0):
    U,S,V = np.linalg.svd(M)
    SVs = np.diag(S)
    D = np.zeros(S.shape)
    if len(alpha)>0:
        D[0:len(SVs),0:len(SVs)] = np.diag(SVs / (SVs * SVs + alpha**2))
    else:
        D[0:len(SVs),0:len(SVs)] = np.diag(1./SVs)
    if len(damping)>0:
        D = damping * D
    if N!=0:
        keep = len(SVs)-N
        D[keep+1:len(SVs),keep+1:len(SVs)] = 0
    Minv = V * D.T * U.T
    if plot:
        plt.figure(66)
        plt.subplot(1,2,1)
        plt.semilogy(np.diag(S)/np.max(np.diag(S)),'o--')
        plt.xlabel('Number of SV')
        plt.ylabel('$\sigma/\sigma_0$')
        plt.subplot(1,2,2)
        plt.plot(np.diag(S)*np.diag(D),'o--')
        plt.xlabel('Number of SV')
        plt.ylabel('$\sigma * \sigma^+$')
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.gcf().set_size_inches(12,4)
        plt.gcf().set_dpi(100)
        plt.gcf().set_facecolor('w')
        plt.show()
    return Minv
# End
# Test

# M = np.random.rand(10,10)
# Minv = SCgetPinv(M,N=0,alpha=0.1,damping=0.1,plot=1)
# End
 
