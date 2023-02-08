import numpy as np


def SCregisterSupport(SC,*args):
    if len(args)<2:
        return
    checkInput(args)
    Nele = len(SC.RING)
    ords = (args[1]-1)%Nele+1
    type = args[0]
    SC.ORD[type] = ords
    for ordPair in ords:
        for n in range(2):
            SC.RING[ordPair[n]][type+'Offset'] = [0,0,0] # [x,y,z]
            SC.RING[ordPair[n]][type+'Roll'] = [0,0,0] # [az,ax,ay]
        for i in range(3,len(args),2):
            if isinstance(args[i+1][0],list):
                SC.SIG.Support[ordPair[0]][type+args[i]] = {args[i+1][0][0],args[i+1][1]}
                if len(args[i+1][0])==2:
                    SC.SIG.Support[ordPair[1]][type+args[i]] = {args[i+1][0][1],args[i+1][1]}
            else:
                SC.SIG.Support[ordPair[0]][type+args[i]] = args[i+1][0]
                if len(args[i+1])==2:
                    SC.SIG.Support[ordPair[1]][type+args[i]] = args[i+1][1]

def checkInput(args):
    if args[0] not in ['Girder','Plinth','Section']:
        raise ValueError('Unsupported structure type. Allowed are ''Girder'', ''Plinth'' and ''Section''.')
    if len(args[1])==0 or len(args[1][0])!=2:
        raise ValueError('Ordinates must be a 2xn array of ordinates.')
    if len(args)%2:
        raise ValueError('Optional input must be given as name-value pairs.')
    if any(np.diff(args[1],1)<0):
        print('%d ''%s'' endpoint(s) might be upstream of startpoint(s).'%(sum(np.diff(args[1],1)<0),args[0]))
    if 'Offset' in args:
        offset = args[args.index('Offset')+1]
        if isinstance(offset,list):
            if len(offset[1])!=1:
                raise ValueError('Sigma cutoff must be a single value.')
            offset = offset[0]
        if len(offset[0])!=3 or (len(offset)!=1 and len(offset)!=2):
            print('Support structure offset uncertainty of ''%s'' must be given as [1x3] (start end endpoints get same offset errors) or [2x3] (start end endpoints get independent offset errors) array.'%args[0])
    if 'Roll' in args:
        roll = args[args.index('Roll')+1]
        if isinstance(roll,list):
            if len(roll[1])!=1:
                raise ValueError('Sigma cutoff must be a single value.')
            roll = roll[0]
        if len(roll)!=3:
            print('''%s roll uncertainty must be [1x3] array [az,ax,ay] of roll (around z-axis), pitch (roll around x-axis) and yaw (roll around y-axis) angle.'''%args[0])
# End

