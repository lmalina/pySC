import numpy as np


def SCgetSupportOffset(SC,s):
    s0 = np.cumsum(np.array([SC.RING[i].Length for i in range(len(SC.RING))]))
    off0 = np.zeros((3,len(s0)))
    for n in range(len(SC.RING)):
        lengths[n] = SC.RING[n].Length
    C = sum(lengths)
    sposMID = np.cumsum(lengths)-lengths/2
    for type in ['Section','Plinth','Girder']:
        if type in SC.ORD:
            ord1=SC.ORD[type][0,:] # Beginning ordinates
            ord2=SC.ORD[type][1,:] # End ordinates
            s1=sposMID[ord1]
            s2=sposMID[ord2]
            tmpoff1=np.zeros((3,len(ord1)))
            tmpoff2=np.zeros((3,len(ord2)))
            for i in range(len(ord1)):
                tmpoff1[:,i] = off0[:,ord1[i]] + SC.RING[ord1[i]][type+'Offset']
                tmpoff2[:,i] = off0[:,ord2[i]] + SC.RING[ord2[i]][type+'Offset']
            off0[0,:] =  limp(off0[0,:],s0,C,s1,ord1,tmpoff1[0,:],s2,ord2,tmpoff2[0,:])
            off0[1,:] =  limp(off0[1,:],s0,C,s1,ord1,tmpoff1[1,:],s2,ord2,tmpoff2[1,:])
            off0[2,:] =  limp(off0[2,:],s0,C,s1,ord1,tmpoff1[2,:],s2,ord2,tmpoff2[2,:])
    if not np.array_equal(s,s0):
        b = np.unique(s0,return_index=True)[1]
        off[0,:] = np.interp(s,s0[b],off0[0,b])
        off[1,:] = np.interp(s,s0[b],off0[1,b])
        off[2,:] = np.interp(s,s0[b],off0[2,b])
    else:
        off = off0
    return off

def limp(off,s,C,s1,ord1,f1,s2,ord2,f2):
    for n in range(len(s1)):
        if s1[n]==s2[n]: # Sampling points have same s-position
            if f1[n]!=f2[n]:
                raise ValueError('Something went wrong.')
            ind = ord1[n]:ord2[n]
            off[ind] = f1[n]
        elif s1[n]<s2[n]: # Standard interpolation
            ind = ord1[n]:ord2[n]
            off[ind] = np.interp([s1[n] s2[n]],[f1[n] f2[n]],s[ind])
        else: # Injection is within sampling points
            ind1 = 1:ord2[n]
            ind2 = ord1[n]:len(off)
            off[ind1] = np.interp([s1[n] s2[n]+C],[f1[n] f2[n]],C+s[ind1])
            off[ind2] = np.interp([s1[n] s2[n]+C],[f1[n] f2[n]],s[ind2])
    return off
# End
 
