import pytest
import numpy as np
from numpy.testing import assert_allclose

from pySC.utils.sc_tools import update_transformation, rotation
from pySC.core.classes import DotDict


def test_compare_transformation(element):
    print("\n")
    print(element)
    ref_element = element.deepcopy()
    magLength = element.Length
    magTheta = element.BendingAngle if hasattr(element, 'BendingAngle') else 0
    magnet_offsets = element.SupportOffset + element.MagnetOffset
    magnet_rolls = np.roll(element.MagnetRoll + element.SupportRoll, -1)  # z,x,y -> x,y,z
    T1, T2, R1, R2 = _sc_get_transformation(magnet_offsets, magnet_rolls, magTheta, magLength, refPoint="entrance")
    ref_element = update_transformation(ref_element)
    assert_allclose(ref_element.T1, T1)
    assert_allclose(ref_element.R1, R1)
    assert_allclose(ref_element.R2, R2)
    assert_allclose(ref_element.T2, T2)


def _sc_get_transformation(offsets, rolls, magTheta, magLength, refPoint='center'):
    xAxis = np.array([1, 0, 0])
    yAxis = np.array([0, 1, 0])
    zAxis = np.array([0, 0, 1])
    RX = rotation(rolls)
    OP = offsets[:]
    if refPoint == 'center':
        RB2 = rotation([0, -magTheta/2, 0])
        RX = np.dot(RB2, np.dot(RX, RB2.T))
        OP = np.dot(RB2, OP) + np.dot(np.eye(3)-RX, np.dot(RB2, zAxis)) * magLength * (1/2 if magTheta == 0 else np.sin(magTheta / 2) / magTheta)

    for face in range(2):
        if face == 0:
            R = RX
            XaxiSxyz = np.dot(R, xAxis)
            YaxiSxyz = np.dot(R, yAxis)
            ZaxiSxyz = np.dot(R, zAxis)
            LD = np.dot(ZaxiSxyz, OP)
            tmp = OP
        else:
            RB = rotation([0, -magTheta, 0])
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


@pytest.fixture
def element():
    return DotDict(dict(
        Length=1, BendingAngle=0.01, SupportOffset=1e-3*np.random.randn(3), MagnetOffset=1e-3 * np.random.randn(3),
        SupportRoll=1e-3*np.random.randn(3), MagnetRoll=1e-3*np.random.randn(3)))
