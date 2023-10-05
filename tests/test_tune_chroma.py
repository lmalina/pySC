import copy
import at
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from tests.test_at_wrapper import at_lattice
from pySC.correction.tune import tune_scan, fit_tune
from pySC.correction.chroma import fit_chroma
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds


def test_fit_tune(at_lattice):
    sc = SimulatedCommissioning(at_lattice)
    sc.register_cavities(SCgetOrds(sc.RING, 'RFCav'),
                         FrequencyOffset=5E3,
                         VoltageOffset=5E3,
                         TimeLagOffset=0.5)
    sc.register_magnets(SCgetOrds(sc.RING, 'QF'), HCM=1E-3,
                        CalErrorB=np.array([5E-2, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc.register_magnets(SCgetOrds(sc.RING, 'QD'), VCM=1E-3,
                        CalErrorA=np.array([5E-2, 0]),
                        CalErrorB=np.array([0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc = fit_tune(sc, q_ords=[SCgetOrds(sc.RING, 'QF'), SCgetOrds(sc.RING, 'QD')], target_tune=np.array([0.16, 0.21]))
    return sc