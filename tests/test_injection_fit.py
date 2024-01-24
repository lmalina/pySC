import pytest
import numpy as np
from pySC.utils import sc_tools
from pySC.correction.injection_fit import fit_injection_trajectory, fit_injection_drift
from pySC.core.simulated_commissioning import SimulatedCommissioning
from tests.test_at_wrapper import at_lattice


def test_fit_injection_trajectory(sc):
    fit_injection_trajectory(sc, np.arange(3, dtype=int), plot=False)
    fit_injection_trajectory(sc, np.arange(3, dtype=int), plot=True)


def test_fit_injection_drift(sc):
    sc.INJ.nTurns = 2
    fit_injection_drift(sc, np.arange(2, dtype=int), plot=False)
    fit_injection_drift(sc, np.arange(2, dtype=int), plot=True)


@pytest.fixture
def sc(at_lattice):
    SC = SimulatedCommissioning(at_lattice)
    SC.register_bpms(sc_tools.ords_from_regex(SC.RING, 'BPM'), CalError=5E-2 * np.ones(2),
                     Offset=200E-6 * np.ones(2),
                     Noise=10E-6 * np.ones(2),
                     Roll=1E-3)
    SC.INJ.beamSize = np.diag(np.array([200E-6, 100E-6, 100E-6, 50E-6, 1E-3, 1E-4]) ** 2)
    SC.SIG.randomInjectionZ = np.array([1E-4, 1E-5, 1E-4, 1E-5, 1E-4, 1E-4])
    SC.SIG.staticInjectionZ = np.array([1E-3, 1E-4, 1E-3, 1E-4, 1E-3, 1E-3])

    SC.apply_errors()
    return SC
