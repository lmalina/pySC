import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from pySC.plotting.plot_lattice import plot_lattice, plot_cm_strengths, _get_s_range
from pySC.plotting.plot_support import plot_support
from pySC.plotting.plot_phase_space import plot_phase_space
from pySC.utils import at_wrapper, sc_tools
from pySC.core.simulated_commissioning import SimulatedCommissioning
from tests.test_at_wrapper import at_lattice


def test_plot_lattice(sc):
    plot_lattice(sc)
    plot_lattice(sc, n_sectors=10, plot_magnet_names=True)


def test_get_s_range(sc):
    s_range = np.array([0, 20])
    indices = np.array([30, 130], dtype=int)
    s_indices = at_wrapper.findspos(sc.RING)[indices]
    n_sectors = 10
    assert_equal(_get_s_range(sc.RING, s_range, indices, n_sectors), s_range)
    assert_equal(_get_s_range(sc.RING, None, indices, n_sectors), s_indices)
    assert_allclose(_get_s_range(sc.RING, None, None, n_sectors), np.array([0., 10.]))
    assert_equal(_get_s_range(sc.RING, None, None, None), sc.RING.s_range)


def test_plot_cm_strengths(sc):
    plot_cm_strengths(sc)


def test_plot_phase_space(sc):
    plot_phase_space(sc, nParticles=3, nTurns=20, plotCO=True)


def test_plot_support(sc):
    plot_support(sc)


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_bpms(sc_tools.ords_from_regex(SC.RING, 'BPM'), CalError=5E-2 * np.ones(2),
                     Offset=500E-6 * np.ones(2),
                     Noise=10E-6 * np.ones(2),
                     NoiseCO=0E-6 * np.ones(2),
                     Roll=1E-3)
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QF'),
                        HCM=1E-3,
                        CalErrorB=np.array([5E-2, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, 'QD'), VCM=1E-3,
                        CalErrorA=np.array([5E-2, 0]),
                        CalErrorB=np.array([0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    SC.register_cavities(sc_tools.ords_from_regex(SC.RING, 'RFCav'), FrequencyOffset=5E3,
                         VoltageOffset=5E3,
                         TimeLagOffset=0.5)
    SC.register_supports(np.vstack((sc_tools.ords_from_regex(SC.RING, 'GirderStart'),
                                    sc_tools.ords_from_regex(SC.RING, 'GirderEnd'))),
                         "Girder",
                         Offset=100E-6 * np.array([1, 1, 0]),
                         Roll=200E-6 * np.array([1, 0, 0]))
    for e_ord in sc_tools.ords_from_regex(SC.RING, 'Drift'):
        SC.RING[e_ord].EApertures = 13E-3 * np.array([1, 1])
    np.random.seed(123)
    SC.apply_errors()
    return SC
