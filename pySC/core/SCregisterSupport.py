import numpy as np


def SCregisterSupport(SC,**args):
    keywords = ['Girder', 'Plinth', 'Section']
    stripped_dict={key: value for key, value in args.items() if key not in keywords}
    type, ords = [[key, value] for key, value in args.items() if key in keywords][0]
    #checkInput(args)  #
    SC.ORD[type] = ords  #  TODO similar to register BPMs
    if "Support" not in SC.SIG.keys():
        SC.SIG["Support"] = dict()
    for ord in np.ravel(ords):
        setattr(SC.RING[ord], f"{type}Offset", np.zeros(3))  # [x,y,z]
        setattr(SC.RING[ord], f"{type}Roll", np.zeros(3))  # [az,ax,ay]
        SC.SIG["Support"][ord] = dict()
    for ordPair in ords.T:
        for key in stripped_dict.keys():
            if isinstance(stripped_dict[key], list):
                if stripped_dict[key][0].ndim == 1:
                    SC.SIG.Support[ordPair[0]][f"{type}{key}"]= stripped_dict[key]
                else:
                    SC.SIG.Support[ordPair[0]][f"{type}{key}"]=[stripped_dict[key][0][0, :], stripped_dict[key][1]]
                    SC.SIG.Support[ordPair[1]][f"{type}{key}"]= [stripped_dict[key][0][1, :], stripped_dict[key][1]]

            else:
                if stripped_dict[key].ndim == 1:
                    SC.SIG.Support[ordPair[0]][f"{type}{key}"]=stripped_dict[key]
                else:
                    SC.SIG.Support[ordPair[0]][f"{type}{key}"]=stripped_dict[key][0, :]
                    SC.SIG.Support[ordPair[1]][f"{type}{key}"]= stripped_dict[key][1, :]
    return SC

def checkInput(args):

    if first_key:=list(args.keys())[0] not in ['Girder','Plinth','Section']:
        raise ValueError('Unsupported structure type. Allowed are ''Girder'', ''Plinth'' and ''Section''.')
    if len(args[first_key])==0 or args[first_key].shape[0]!=2:
        raise ValueError('Ordinates must be a 2xn array of ordinates.')
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

