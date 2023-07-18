import copy

import pytest
import numpy as np
from numpy.testing import assert_equal
from tests.test_at_wrapper import at_lattice
from pySC.utils.sc_tools import SCgetOrds
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.lattice_setting import set_cm_setpoints, set_magnet_setpoints, set_cavity_setpoints, get_cm_setpoints


def test_set_cm_setpoints_side_effects(sc):
    indices = np.arange(11, 450, 22, dtype=int)
    setpoints = 1e-4 * np.ones(len(indices))
    for _ in range(2):
        sc, _ = set_cm_setpoints(sc, indices, setpoints, True, method="add")
        assert_equal(indices, np.arange(11, 450, 22, dtype=int))
        assert_equal(setpoints, 1e-4 * np.ones(len(indices)))


def test_set_magnet_setpoints_side_effects(sc):
    lattice_copy = copy.deepcopy(sc.RING)
    for ind, element in enumerate(sc.RING):
        lattice_copy[ind] = element.deepcopy()
    indices = np.arange(11, 450, 22, dtype=int)
    setpoints = 1e-4 * np.ones(len(indices))
    for _ in range(2):
        sc = set_magnet_setpoints(sc, indices, True, 1, setpoints,  method="add",)
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
        sc = set_cavity_setpoints(sc,indices, "Frequency", setpoints, method="add")
        assert_equal(indices, np.zeros(1, dtype=int))
        assert_equal(setpoints, np.ones(1))
        assert sc.RING.__repr__() != lattice_copy.__repr__()


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_magnets(SCgetOrds(SC.RING, 'QF'), HCM=1E-3,
                        CalErrorB=np.array([5E-2, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_magnets(SCgetOrds(SC.RING, 'QD'), VCM=1E-3,
                        CalErrorA=np.array([5E-2, 0]),
                        CalErrorB=np.array([0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_cavities(SCgetOrds(SC.RING, 'RFCav'),
                         FrequencyOffset=5E3,
                         VoltageOffset=5E3,
                         TimeLagOffset=0.5)
    SC.apply_errors()
    return SC