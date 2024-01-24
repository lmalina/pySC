import copy
import at
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from tests.test_at_wrapper import at_lattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.constants import SETTING_ABS, SETTING_REL, SETTING_ADD
from pySC.utils import sc_tools, classdef_tools


def test_set_cm_setpoints_side_effects(sc):
    indices = np.arange(11, 450, 22, dtype=int)
    setpoints = 1e-4 * np.ones(len(indices))
    for _ in range(2):
        sc.set_cm_setpoints(indices, setpoints, True, method=SETTING_ADD)
        assert_equal(indices, np.arange(11, 450, 22, dtype=int))
        assert_equal(setpoints, 1e-4 * np.ones(len(indices)))


def test_set_magnet_setpoints_side_effects(sc):
    lattice_copy = copy.deepcopy(sc.RING)
    for ind, element in enumerate(sc.RING):
        lattice_copy[ind] = element.deepcopy()
    indices = np.arange(11, 450, 22, dtype=int)
    setpoints = 1e-4 * np.ones(len(indices))
    for _ in range(2):
        sc.set_magnet_setpoints(indices, setpoints, True, 1, method=SETTING_ADD)
        assert_equal(indices, np.arange(11, 450, 22, dtype=int))
        assert_equal(setpoints, 1e-4 * np.ones(len(indices)))
        assert sc.RING.__repr__() != lattice_copy.__repr__()


def test_set_cavity_setpoints_side_effects(sc):
    lattice_copy = copy.deepcopy(sc.RING)
    for ind, element in enumerate(sc.RING):
        lattice_copy[ind] = element.deepcopy()
    indices = np.zeros(1, dtype=int)
    setpoints = np.ones(1)
    for _ in range(2):
        sc.set_cavity_setpoints(indices, setpoints, "Frequency", method=SETTING_ADD)
        assert_equal(indices, np.zeros(1, dtype=int))
        assert_equal(setpoints, np.ones(1))
        assert sc.RING.__repr__() != lattice_copy.__repr__()


def test_check_input_and_setpoints():
    wanted_ord, wanted_setpoint = np.array([2], dtype=int),  np.array([3.5])
    for o, s in zip((2, [2], np.array([2])),
                    (3.5,  np.array([3.5]), [3.5])):
        ord1d, setp1d = classdef_tools.check_input_and_setpoints(SETTING_ABS, o, s)
        assert_equal(ord1d, wanted_ord)
        assert_equal(setp1d, wanted_setpoint)

    wanted_ords, wanted_setpoints = np.array([2, 6], dtype=int), np.array([3.5, 3.5])
    for os, ss in zip(((2, 6), [2, 6], np.array([2, 6])),
                      (3.5, np.array([3.5, 3.5]), [3.5])):
        ords1d, setps1d = classdef_tools.check_input_and_setpoints(SETTING_ABS, os, ss)
        assert_equal(ords1d, wanted_ords)
        assert_equal(setps1d, wanted_setpoints)
    # check also empty ord
    wanted_ord, wanted_setpoint = np.array([], dtype=int), np.array([])
    ord1d, setp1d = classdef_tools.check_input_and_setpoints(SETTING_REL, [], 1.0)
    assert_equal(ord1d, wanted_ord)
    assert_equal(setp1d, wanted_setpoint)

def test_set_magnet_setpoints_empty(unit_sc):
    unit_sc.set_magnet_setpoints([], 1.0, False, 1, method=SETTING_REL)


def test_set_magnet_setpoints(unit_sc):
    unit_sc.set_magnet_setpoints([2, 3], 1.1, False, 1, method=SETTING_REL)
    assert_equal(unit_sc.RING[2].PolynomB, np.array([0, 1.32]))
    assert_equal(unit_sc.RING[3].PolynomB, np.array([0, -1.32]))
    unit_sc.set_magnet_setpoints(2, 0.1, False, 1, method=SETTING_ADD)
    assert_allclose(unit_sc.RING[2].PolynomB, np.array([0, 1.42]))
    assert_equal(unit_sc.RING[3].PolynomB, np.array([0, -1.32]))


def test_set_cm_setpoints(unit_sc):
    unit_sc.set_cm_setpoints([2, 3], 1e-4, False, method=SETTING_ABS)
    assert_equal(unit_sc.RING[2].PolynomB, np.array([-2e-4, 1.2]))  # 2 due to 0.5 m long
    assert_equal(unit_sc.RING[3].PolynomB, np.array([-2e-4, -1.2]))
    unit_sc.set_cm_setpoints([2, 3], 1e-1, False, method=SETTING_ABS)  # clipping at 1 mrad
    assert_equal(unit_sc.RING[2].PolynomB, np.array([-2e-3, 1.2]))
    assert_equal(unit_sc.RING[3].PolynomB, np.array([-2e-3, -1.2]))
    unit_sc.set_cm_setpoints(2, 1e-4, True, method=SETTING_ADD)
    assert_equal(unit_sc.RING[2].PolynomB, np.array([-2e-3, 1.2]))  # PolynomB unchanged
    assert_equal(unit_sc.RING[3].PolynomB, np.array([-2e-3, -1.2]))
    assert_equal(unit_sc.RING[2].PolynomA, np.array([2e-4, 0]))
    assert_equal(unit_sc.RING[3].PolynomA, np.array([0, 0]))  # other magnet's PolynomA also unchanged


@pytest.fixture
def unit_sc():
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass')
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass')
    sf = at.Sextupole('SF', 0.1, 6.0487, PassMethod='StrMPoleSymplectic4RadPass')
    sd = at.Sextupole('SD', 0.1, -9.5203, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='BndMPoleSymplectic4RadPass')
    cm = at.Corrector("C", 0.1, [1e-4, 5e-4])  # TODO corrector does not have PolynomA/B in at?

    cell = at.Lattice([bend, bend, qf, qd, sf, sd], name='Unit cell', energy=2.5E9)
    inds = np.arange(6, dtype=int)
    SC = SimulatedCommissioning(cell)
    SC.register_magnets(inds, HCM=1E-3, VCM=1E-3, SkewQuad=1E-3)
    SC.apply_errors()
    return SC


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QF'), HCM=1E-3,
                        CalErrorB=np.array([5E-2, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QD'), VCM=1E-3,
                        CalErrorA=np.array([5E-2, 0]),
                        CalErrorB=np.array([0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_cavities(sc_tools.ords_from_regex(SC.RING, 'RFCav'),
                         FrequencyOffset=5E3,
                         VoltageOffset=5E3,
                         TimeLagOffset=0.5)
    SC.apply_errors()
    return SC
