import copy

import pytest
import numpy as np
from numpy.testing import assert_equal
import at
from at import Lattice
from pySC.utils.at_wrapper import findspos, atgetfieldvalues, atpass, findorbit6, findorbit4



def test_findspos(at_lattice):
    lattice_copy = copy.deepcopy(at_lattice)
    s_pos = findspos(at_lattice)
    assert_equal(s_pos, np.concatenate(([0.0], np.cumsum([getattr(el, 'Length', 0.0) for el in at_lattice]))))
    assert at_lattice.__repr__() == lattice_copy.__repr__()


def test_atgetfieldvalues(at_lattice):
    lattice_copy = copy.deepcopy(at_lattice)
    indices = np.arange(11, 450, 22, dtype=int)
    polynoms_b = atgetfieldvalues(at_lattice, indices, "PolynomB")
    assert_equal(polynoms_b, np.ones((20, 2)) * np.array([0.0, 1.2]))
    assert at_lattice.__repr__() == lattice_copy.__repr__()
    assert_equal(indices, np.arange(11, 450, 22, dtype=int))


def test_atpass(at_lattice):
    lattice_copy = copy.deepcopy(at_lattice)
    indices = np.arange(11, 450, 22, dtype=int)
    initial_pos = np.random.randn(6)
    copy_initial_pos = copy.deepcopy(initial_pos)
    tracking = atpass(at_lattice, initial_pos, 3, indices,)
    assert tracking.shape == (6, 1, 20, 3)
    assert_equal(initial_pos, copy_initial_pos)
    assert at_lattice.__repr__() == lattice_copy.__repr__()
    assert_equal(indices, np.arange(11, 450, 22, dtype=int))


def test_findorbit6(at_lattice):
    lattice_copy = copy.deepcopy(at_lattice)
    indices = np.arange(11, 450, 22, dtype=int)
    orbit0, orbit1 = findorbit6(at_lattice, indices)
    assert orbit0.shape == (6,)
    assert orbit1.shape == (20,6,)
    assert at_lattice.__repr__() == lattice_copy.__repr__()
    assert_equal(indices, np.arange(11, 450, 22, dtype=int))


def test_findorbit4(at_lattice):
    new_lattice = copy.deepcopy(at_lattice)
    new_lattice.disable_6d()
    lattice_copy = copy.deepcopy(new_lattice)
    indices = np.arange(11, 450, 22, dtype=int)
    orbit0, orbit1 = findorbit4(new_lattice, 0.0, indices)
    assert orbit0.shape == (6,)
    assert orbit1.shape == (20,6,)
    assert new_lattice.__repr__() == lattice_copy.__repr__()
    assert_equal(indices, np.arange(11, 450, 22, dtype=int))


@pytest.fixture
def at_lattice() -> Lattice:
    def _marker(name):
        return at.Marker(name, PassMethod='IdentityPass')
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass')
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass')
    sf = at.Sextupole('SF', 0.1, 6.0487, PassMethod='StrMPoleSymplectic4RadPass')
    sd = at.Sextupole('SD', 0.1, -9.5203, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='BndMPoleSymplectic4RadPass')
    d2 = at.Drift('Drift', 0.25)
    d3 = at.Drift('Drift', 0.2)
    BPM= at.Monitor('BPM')

    cell = at.Lattice([d2, _marker('SectionStart'), _marker('GirderStart'), bend, d3, sf, d3, _marker('GirderEnd'),
                       _marker('GirderStart'), BPM, qf, d2, d2, bend, d3, sd, d3, qd, d2, _marker('BPM'),
                       _marker('GirderEnd'), _marker('SectionEnd')], name='Simple FODO cell', energy=2.5E9)
    new_ring = at.Lattice(cell * 20)
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=149896228.99999985, harmonic_number=50, length=0)
    new_ring.insert(0, rfc)
    new_ring.enable_6d()
    return new_ring
