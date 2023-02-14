import numpy as np


def SCgetTransformation(dx, dy, dz, ax, ay, az, magTheta, magLength, refPoint='center'):
    if refPoint not in ('center', 'entrance'):
        raise ValueError(f'Unsupported reference point {refPoint}. Allowed are ''center'' or ''entrance''.')
    d0Vector = np.array([dx, dy, dz])
    xAxis = np.array([1, 0, 0])
    yAxis = np.array([0, 1, 0])
    zAxis = np.array([0, 0, 1])

    R0 = np.array([[np.cos(ay) * np.cos(az), -np.cos(ay) * np.sin(az), np.sin(ay)],
                   [np.cos(az) * np.sin(ax) * np.sin(ay) + np.cos(ax) * np.sin(az),
                    np.cos(ax) * np.cos(az) - np.sin(ax) * np.sin(ay) * np.sin(az), -np.cos(ay) * np.sin(ax)],
                   [-np.cos(ax) * np.cos(az) * np.sin(ay) + np.sin(ax) * np.sin(az),
                    np.cos(az) * np.sin(ax) + np.cos(ax) * np.sin(ay) * np.sin(az), np.cos(ax) * np.cos(ay)]])
    if refPoint == 'center':
        RB2 = np.array([[np.cos(magTheta / 2), 0, -np.sin(magTheta / 2)],
                        [0, 1, 0],
                        [np.sin(magTheta / 2), 0, np.cos(magTheta / 2)]])
        RX = np.dot(RB2, np.dot(R0, RB2.T))
        if magTheta == 0:
            OO0 = (magLength / 2) * zAxis
            P0P = -(magLength / 2) * np.dot(RX, zAxis)
        else:
            Rc = magLength / magTheta
            OO0 = Rc * np.sin(magTheta / 2) * np.dot(RB2, zAxis)
            P0P = -Rc * np.sin(magTheta / 2) * np.dot(RX, np.dot(RB2, zAxis))
        OP = OO0 + P0P + np.dot(RB2, d0Vector)
    else:
        RX = R0
        OP = d0Vector

    for face in range(2):
        if face == 0:
            R = RX
            XaxiSxyz = np.dot(R, xAxis)
            YaxiSxyz = np.dot(R, yAxis)
            ZaxiSxyz = np.dot(R, zAxis)
            LD = np.dot(ZaxiSxyz, OP)
            tmp = OP
        else:
            RB = np.array([[np.cos(magTheta), 0, -np.sin(magTheta)],
                           [0, 1, 0],
                           [np.sin(magTheta), 0, np.cos(magTheta)]])
            R = np.dot(RB.T, np.dot(RX.T, RB))
            XaxiSxyz = np.dot(RB, xAxis)
            YaxiSxyz = np.dot(RB, yAxis)
            ZaxiSxyz = np.dot(RB, zAxis)
            if magTheta == 0:
                OPp = np.array([0, 0, magLength])
            else:
                Rc = magLength / magTheta
                OPp = np.array([Rc * (np.cos(magTheta) - 1), 0, magLength * np.sin(magTheta) / magTheta])
            OOp = np.dot(RX, OPp) + OP
            OpPp = (OPp - OOp)
            LD = np.dot(ZaxiSxyz, OpPp)
            tmp = OpPp
        tD0 = np.array([-np.dot(tmp, XaxiSxyz), 0, -np.dot(tmp, YaxiSxyz), 0, 0, 0])
        T0 = np.array([LD * R[2, 0] / R[2, 2], R[2, 0], LD * R[2, 1] / R[2, 2], R[2, 1], 0, LD / R[2, 2]])
        T = T0 + tD0
        LinMat = np.array(
            [[R[1, 1] / R[2, 2], LD * R[1, 1] / R[2, 2] ** 2, -R[0, 1] / R[2, 2], -LD * R[0, 1] / R[2, 2] ** 2, 0, 0],
             [0, R[0, 0], 0, R[1, 0], R[2, 0], 0],
             [-R[1, 0] / R[2, 2], -LD * R[1, 0] / R[2, 2] ** 2, R[0, 0] / R[2, 2], LD * R[0, 0] / R[2, 2] ** 2, 0, 0],
             [0, R[0, 1], 0, R[1, 1], R[2, 1], 0],
             [0, 0, 0, 0, 1, 0],
             [-R[0, 2] / R[2, 2], -LD * R[0, 2] / R[2, 2] ** 2, -R[1, 2] / R[2, 2], -LD * R[1, 2] / R[2, 2] ** 2, 0,
              1]])
        if face == 0:
            R1 = LinMat
            T1 = np.dot(np.linalg.inv(R1), T)
        else:
            R2 = LinMat
            T2 = T
    return T1, T2, R1, R2


def sc_get_transformation(dx, dy, dz, ax, ay, az, magTheta, magLength, refPoint='center'):
    if refPoint not in ('center', 'entrance'):
        raise ValueError('Unsupported reference point. Allowed are ''center'' or ''entrance''.')
    d0Vector = np.array([dx, dy, dz])
    xAxis = np.array([1, 0, 0])
    yAxis = np.array([0, 1, 0])
    zAxis = np.array([0, 0, 1])

    R0 = np.array([[np.cos(ay) * np.cos(az), -np.cos(ay) * np.sin(az), np.sin(ay)],
                   [np.cos(az) * np.sin(ax) * np.sin(ay) + np.cos(ax) * np.sin(az),
                    np.cos(ax) * np.cos(az) - np.sin(ax) * np.sin(ay) * np.sin(az), -np.cos(ay) * np.sin(ax)],
                   [-np.cos(ax) * np.cos(az) * np.sin(ay) + np.sin(ax) * np.sin(az),
                    np.cos(az) * np.sin(ax) + np.cos(ax) * np.sin(ay) * np.sin(az), np.cos(ax) * np.cos(ay)]])
    if refPoint == 'center':
        RB2 = np.array([[np.cos(magTheta / 2), 0, -np.sin(magTheta / 2)],
                        [0, 1, 0],
                        [np.sin(magTheta / 2), 0, np.cos(magTheta / 2)]])
        RX = np.dot(RB2, np.dot(R0, RB2.T))
        if magTheta == 0:
            OO0 = (magLength / 2) * zAxis
            P0P = -(magLength / 2) * np.dot(RX, zAxis)
        else:
            Rc = magLength / magTheta
            OO0 = Rc * np.sin(magTheta / 2) * np.dot(RB2, zAxis)
            P0P = -Rc * np.sin(magTheta / 2) * np.dot(RX, np.dot(RB2, zAxis))
        OP = OO0 + P0P + np.dot(RB2, d0Vector)
    else:
        RX = R0
        OP = d0Vector

    for face in range(2):
        if face == 0:
            R = RX
            XaxiSxyz = np.dot(R, xAxis)
            YaxiSxyz = np.dot(R, yAxis)
            ZaxiSxyz = np.dot(R, zAxis)
            LD = np.dot(ZaxiSxyz, OP)
            tmp = OP
        else:
            RB = np.array([[np.cos(magTheta), 0, -np.sin(magTheta)],
                           [0, 1, 0],
                           [np.sin(magTheta), 0, np.cos(magTheta)]])
            R = np.dot(RB.T, np.dot(RX.T, RB))
            XaxiSxyz = np.dot(RB, xAxis)
            YaxiSxyz = np.dot(RB, yAxis)
            ZaxiSxyz = np.dot(RB, zAxis)
            if magTheta == 0:
                OPp = np.array([0, 0, magLength])
            else:
                Rc = magLength / magTheta
                OPp = np.array([Rc * (np.cos(magTheta) - 1), 0, magLength * np.sin(magTheta) / magTheta])
            OOp = np.dot(RX, OPp) + OP
            OpPp = (OPp - OOp)
            LD = np.dot(ZaxiSxyz, OpPp)
            tmp = OpPp
        tD0 = np.array([-np.dot(tmp, XaxiSxyz), 0, -np.dot(tmp, YaxiSxyz), 0, 0, 0])
        T0 = np.array([LD * R[2, 0] / R[2, 2], R[2, 0], LD * R[2, 1] / R[2, 2], R[2, 1], 0, LD / R[2, 2]])
        T = T0 + tD0
        LinMat = np.array(
            [[R[1, 1] / R[2, 2], LD * R[1, 1] / R[2, 2] ** 2, -R[0, 1] / R[2, 2], -LD * R[0, 1] / R[2, 2] ** 2, 0, 0],
             [0, R[0, 0], 0, R[1, 0], R[2, 0], 0],
             [-R[1, 0] / R[2, 2], -LD * R[1, 0] / R[2, 2] ** 2, R[0, 0] / R[2, 2], LD * R[0, 0] / R[2, 2] ** 2, 0, 0],
             [0, R[0, 1], 0, R[1, 1], R[2, 1], 0],
             [0, 0, 0, 0, 1, 0],
             [-R[0, 2] / R[2, 2], -LD * R[0, 2] / R[2, 2] ** 2, -R[1, 2] / R[2, 2], -LD * R[1, 2] / R[2, 2] ** 2, 0,
              1]])
        if face == 0:
            R1 = LinMat
            T1 = np.dot(np.linalg.inv(R1), T)
        else:
            R2 = LinMat
            T2 = T
    return T1, T2, R1, R2
