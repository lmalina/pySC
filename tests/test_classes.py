import copy

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import at
from pySC.classes import DotDict, Injection, Indices, Sigmas, SimulatedComissioning



def test_indices(at_cell):
    sc = SimulatedComissioning(at_cell)
    sc.RING[9].PolynomB[1] = 1
    assert sc.RING[10].PolynomB[1] == 1.2
    assert sc.IDEALRING[9].PolynomB[1] == 1.2
    assert at_cell[9].PolynomB[1] == 1.2
    pb_sf = np.array([0, 0, 4])
    sc.RING[4].PolynomB = pb_sf
    pb_sf[0] = 1
    assert_allclose(sc.RING[4].PolynomB, np.array([0, 0, 4]))
    assert_allclose(sc.RING[5].PolynomB, np.array([0, 0, 6]))
    pb_sf2 = copy.deepcopy(getattr(sc.RING[5], "PolynomB"))
    pb_sf2[0] = 1
    assert_allclose(sc.RING[5].PolynomB, np.array([0, 0, 6]))
    sc.RING[5].PolynomB[0] += 1
    assert_allclose(sc.RING[5].PolynomB, np.array([1, 0, 6]))
    pb_sf3 = np.zeros(3)
    pb_sf3[:] = sc.RING[5].PolynomB
    pb_sf3[0] = 1
    assert_allclose(sc.RING[5].PolynomB, np.array([1, 0, 6]))





@pytest.fixture
def at_cell():
    def _marker(name):
        return at.Marker(name, PassMethod='IdentityPass')
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass')
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass')
    sf = at.Sextupole('SF', 0.1, 6.0, PassMethod='StrMPoleSymplectic4RadPass')
    sd = at.Sextupole('SD', 0.1, -9.5, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='StrMPoleSymplectic4RadPass')
    d2 = at.Drift('Drift', 0.25)
    d3 = at.Drift('Drift', 0.2)
    cell = at.Lattice([d2, _marker('SectionStart'), bend, d3, sf, sf, d3, _marker('BPM'), qf, qf, d2, d2, bend, d3,
                       sd, sd, d3, qd, qd, d2, _marker('BPM'), _marker('SectionEnd')], name='Simple cell', energy=2.5E9)
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=1, harmonic_number=50, length=0)
    cell.insert(0, rfc)
    return cell
