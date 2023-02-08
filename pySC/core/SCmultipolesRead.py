import numpy as np


def SCmultipolesRead(fname):
    f=open(fname,'r')
    tab=np.array(f.read().split()).astype(float)
    f.close()
    if(len(tab)%3!=0):
        print('Incorrect table size.')
        return
    AB = tab.reshape((-1,3))[:,1:]
    idx=np.where(AB==1)
    if len(idx[0])!=1:
        print('Nominal order could not be (uniquely) determined. Continuing with idx=1.')
        idx=1
    order,type = idx[0][0],idx[1][0]
    if type>2:
        print('Ill-defined magnet type.')
        return
    return AB, order, type
# End

# Test

# fname = '../data/SCmultipoles.dat'
# AB, order, type = SCmultipolesRead(fname)
# print(AB)
# print(order)
# print(type)

# End

