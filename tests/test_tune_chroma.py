import numpy as np
from numpy.testing import assert_allclose
from tests.test_at_wrapper import at_lattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.correction.tune import fit_tune
from pySC.utils.sc_tools import SCgetOrds


def test_fit_tune(at_lattice):
    SC = sc(at_lattice)
    SC = fit_tune(SC, q_ords=[SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD')], target_tune=SC.RING.get_tune()[:2]+[0.005,-0.005])
    assert_allclose(actual=SC.RING.get_tune()[:2], desired=SC.RING.get_tune()[:2]+[0.1,0.1], rtol=1e-2)
    return sc

def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_magnets(SCgetOrds(SC.RING, 'QF'))
    SC.register_magnets(SCgetOrds(SC.RING, 'QD'))
    return SC