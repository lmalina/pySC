import pytest
import numpy as np
from numpy.testing import assert_equal
from pySC.core.beam import bpm_reading, all_elements_reading, beam_transmission, generate_bunches
from pySC.utils.sc_tools import SCgetOrds
from pySC.core.simulated_commissioning import SimulatedCommissioning
from tests.test_at_wrapper import at_lattice


def test_beam_transmission_basic(sc):
    max_turns, lost_fraction = beam_transmission(sc)
    assert max_turns == 1
    assert lost_fraction == np.zeros(1)
    max_turns, lost_fraction = beam_transmission(sc, nTurns=2)
    assert max_turns == 2
    assert_equal(lost_fraction, np.zeros(2))
    max_turns, lost_fraction = beam_transmission(sc, nTurns=2, nParticles=3, plot=True)
    assert max_turns == 2
    assert_equal(lost_fraction, np.zeros(2))


def test_all_reading_basic(sc):
    sc.INJ.nParticles = 2
    sc.INJ.nShots = 3
    bpm_readings, all_readings = all_elements_reading(sc)
    assert bpm_readings.shape == (2, sc.INJ.nTurns * sc.ORD.BPM.shape[0])
    assert all_readings.shape == (2, sc.INJ.nParticles, len(sc.RING) + 1, sc.INJ.nTurns, sc.INJ.nShots)
    sc.plot = True
    bpm_readings2, all_readings2 = all_elements_reading(sc)
    assert not np.allclose(bpm_readings2, bpm_readings)
    assert_equal(all_readings2, all_readings)


def test_bpm_reading_basic(sc):
    bpm_readings = bpm_reading(sc)
    assert bpm_readings.shape == (2, sc.INJ.nTurns * sc.ORD.BPM.shape[0])
    bpm_readings2 = bpm_reading(sc)
    assert not np.allclose(bpm_readings2, bpm_readings)
    bpm_readings = bpm_reading(sc, np.arange(3))
    assert bpm_readings.shape == (2, 0)
    sc.INJ.trackMode = "PORB"
    sc.INJ.nTurns = 3
    bpm_readings = bpm_reading(sc, np.arange(18))
    assert bpm_readings.shape == (2, 1)
    sc.plot = True
    bpm_readings = bpm_reading(sc, 20 * np.ones(2))
    assert bpm_readings.shape == (2, 1)


def test_generate_bunches_basic(sc):
    bunch_positions = generate_bunches(sc)
    assert_equal(bunch_positions, np.zeros((6, 1)))
    bunch_positions = generate_bunches(sc, nParticles=3)
    assert_equal(bunch_positions, np.zeros((6, 3)))


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_bpms(SCgetOrds(SC.RING, 'BPM'), CalError=5E-2 * np.ones(2),
                     Offset=500E-6 * np.ones(2),
                     Noise=10E-6 * np.ones(2),
                     NoiseCO=0E-6 * np.ones(2),
                     Roll=1E-3)
    SC.apply_errors()
    return SC
