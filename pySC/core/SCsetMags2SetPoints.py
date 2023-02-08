import numpy as np


def SCsetMags2SetPoints(SC,MAGords,type,order,setpoints,method='abs',dipCompensation=0):
    if len(setpoints)==1:
        setpoints = np.repeat(setpoints,len(MAGords))
    i=0
    for ord in MAGords:
        nomAB = np.array([SC.RING[ord].NomPolynomA,SC.RING[ord].NomPolynomB])
        curAB = np.array([SC.RING[ord].SetPointA  ,SC.RING[ord].SetPointB  ])
        if method=='abs':
            setpoints[i] = setpoints[i]
        elif method=='rel':
            setpoints[i] = setpoints[i] * nomAB[order,type]
        elif method=='add':
            setpoints[i] = setpoints[i] + curAB[order,type]
        else:
            print('Unsupported setpoint flag.')
        setpoints[i] = checkClipping(SC,ord,type,order,setpoints[i])
        if dipCompensation and order==2:
            SC = dipCompensation(SC,ord,setpoints[i])
        if type==1:
            SC.RING[ord].SetPointA[order] = setpoints[i]
        else:
            SC.RING[ord].SetPointB[order] = setpoints[i]
        SC = SCupdateMagnets(SC,ord)
        i = i + 1
    return SC

def dipCompensation(SC,ord,setpoint):
    if not (hasattr(SC.RING[ord],'BendingAngle') and SC.RING[ord].BendingAngle != 0 and ord in SC.ORD.CM[0]):
        return SC
    idealKickDifference =  ( ( setpoint - ( SC.RING[ord].SetPointB[2]-SC.RING[ord].NomPolynomB[2] ) ) / SC.RING[ord].NomPolynomB[2] - 1) * SC.RING[ord].BendingAngle / SC.RING[ord].Length
    SC,_ = SCsetCMs2SetPoints(SC,ord, idealKickDifference*SC.RING[ord].Length ,1,'add')
    return SC

def checkClipping(SC,ord,type,order,setpoint):
    if not (type==1 and order==2):
        return setpoint
    if hasattr(SC.RING[ord],'SkewQuadLimit') and abs(setpoint)>abs(SC.RING[ord].SkewQuadLimit):
        print('SC:SkewLim','Skew quadrupole (ord: %d) is clipping' % ord)
        setpoint = np.sign(setpoint) * SC.RING[ord].SkewQuadLimit
    return setpoint
# Test

# SC = SCinit()
# SC = SCsetMags2SetPoints(SC,SC.ORD.QF,1,2,0.1)
# SC = SCsetMags2SetPoints(SC,SC.ORD.QD,1,2,0.1)
# SC = SCsetMags2SetPoints(SC,SC.ORD.QF,1,2,0.1,'rel')
# SC = SCsetMags2SetPoints(SC,SC.ORD.QD,1,2,0.1,'rel')
# SC = SCsetMags2SetPoints(SC,SC.ORD.QF,1,2,0.1,'add')
# SC = SCsetMags2SetPoints(SC,SC.ORD.QD,1,2,0.1,'add')
