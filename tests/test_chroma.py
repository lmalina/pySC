import at
import numpy as np
from numpy.testing import assert_allclose
import pytest
from pathlib import Path
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils import logging_tools, sc_tools
from pySC.correction.chroma import fit_chroma

LOGGER = logging_tools.get_logger(__name__)
INPUTS = Path(__file__).parent / "inputs"


def test_chroma_hmba(at_ring):
    np.random.seed(12345678)
    sc = SimulatedCommissioning(at_ring)
    sc.register_bpms(sc_tools.ords_from_regex(sc.RING, 'BPM'), Roll=0.0)
    sc.register_magnets(sc_tools.ords_from_regex(sc.RING, 'SF|SD'), CalErrorB=np.array([0, 0, 0.05]))  # [1/m]
    sc.register_cavities(sc_tools.ords_from_regex(sc.RING, 'RFC'))
    sc.apply_errors()
    s_ords = [sc_tools.ords_from_regex(sc.RING, '^SF'), sc_tools.ords_from_regex(sc.RING, '^SD')]
    target = np.array([2.0,2.0])
    sc = fit_chroma(sc, s_ords, target_chroma=target)
    assert_allclose(sc.RING.get_chrom()[0:2], target, atol=2e-6, rtol=1e-6)


@pytest.fixture
def at_ring():
    ring = at.load_mat(f'{INPUTS}/hmba.mat')
    ring.enable_6d()
    at.set_cavity_phase(ring)
    at.set_rf_frequency(ring)

    ring.tapering(niter=3, quadrupole=True, sextupole=True)
    ring = at.Lattice(ring)
    return ring
