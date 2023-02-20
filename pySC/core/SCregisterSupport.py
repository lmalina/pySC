import numpy as np
from pySC.constants import SUPPORT_TYPES
from pySC.classes import DotDict, SimulatedComissioning
from numpy import ndarray


def SCregisterSupport(SC: SimulatedComissioning, support_ords: ndarray, support_type: str,  **kwargs) -> SimulatedComissioning:
    if support_type not in SUPPORT_TYPES:
        raise ValueError(f'Unknown support type ``{support_type}`` found. Allowed are {SUPPORT_TYPES}.')
    if not len(support_ords) or support_ords.shape[0] != 2:
        raise ValueError('Ordinates must be a 2xn array of ordinates.')
    # checkInput(args)
    SC.ORD[support_type] = update_double_ordinates(SC.ORD[support_type], support_ords)
    for ord in np.ravel(support_ords):
        setattr(SC.RING[ord], f"{support_type}Offset", np.zeros(3))  # [x,y,z]
        setattr(SC.RING[ord], f"{support_type}Roll", np.zeros(3))  # [az,ax,ay]
        SC.SIG.Support[ord] = DotDict()
    for ord_pair in support_ords.T:
        for key, value in kwargs.items():
            if isinstance(value, list):
                if value[0].ndim == 1:
                    SC.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value
                else:
                    SC.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = [value[0][0, :], value[1]]
                    SC.SIG.Support[ord_pair[1]][f"{support_type}{key}"] = [value[0][1, :], value[1]]

            else:
                if value.ndim == 1:
                    SC.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value
                else:
                    SC.SIG.Support[ord_pair[0]][f"{support_type}{key}"] = value[0, :]
                    SC.SIG.Support[ord_pair[1]][f"{support_type}{key}"] = value[1, :]
    return SC


def update_double_ordinates(ord1, ord2):
    con = np.concatenate((ord1, ord2), axis=1)
    con = con[:, np.lexsort((con[0, :], con[1, :]))]
    return con[:, np.where(np.sum(np.abs(np.diff(con, axis=1)), axis=0))[0]]

def checkInput(args):
    pass
    #     if any(np.diff(args[1],axis=0)<0):
#         print('%d ''%s'' endpoint(s) might be upstream of startpoint(s).'%(sum(np.diff(args[1],1)<0),args[0]))
#     if 'Offset' in args:
#         offset = args[args.index('Offset')+1]
#         if isinstance(offset,list):
#             if len(offset[1])!=1:
#                 raise ValueError('Sigma cutoff must be a single value.')
#             offset = offset[0]
#         if len(offset[0])!=3 or (len(offset)!=1 and len(offset)!=2):
#             print('Support structure offset uncertainty of ''%s'' must be given as [1x3] (start end endpoints get same offset errors) or [2x3] (start end endpoints get independent offset errors) array.'%args[0])
#     if 'Roll' in args:
#         roll = args[args.index('Roll')+1]
#         if isinstance(roll,list):
#             if len(roll[1])!=1:
#                 raise ValueError('Sigma cutoff must be a single value.')
#             roll = roll[0]
#         if len(roll)!=3:
#             print('''%s roll uncertainty must be [1x3] array [az,ax,ay] of roll (around z-axis), pitch (roll around x-axis) and yaw (roll around y-axis) angle.'''%args[0])
# # End
