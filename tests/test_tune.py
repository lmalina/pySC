import pytest
import numpy as np
from numpy.testing import assert_allclose
from tests.test_at_wrapper import at_lattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.correction.tune import fit_tune
from pySC.utils import sc_tools


def test_fit_tune(sc):
    sc = fit_tune(sc, q_ords=[sc_tools.ords_from_regex(sc.RING, 'QF'),
                              sc_tools.ords_from_regex(sc.RING, 'QD')],
                  target_tune=sc.RING.get_tune(get_integer=True)[:2] + [0.005, -0.005], fit_integer=True)
    # TODO with this tolerance it is not testing much
    assert_allclose(actual=sc.RING.get_tune()[:2], desired=sc.RING.get_tune()[:2] + [0.005, -0.005], rtol=1e-2)
    return sc


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QF'))
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QD'))
    return SC
