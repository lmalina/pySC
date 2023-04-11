import copy

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import at
from pySC.classes import SimulatedComissioning
from pySC.utils.sc_tools import SCgetOrds


def test_simulated_commissioning_data_structure(at_cell):
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


def test_register_bpms(at_cell):
    sc = SimulatedComissioning(at_cell)
    bpm_dict = dict(CalError=5E-2 * np.ones(2), Offset=500E-6 * np.ones(2),
                    Noise=10E-6 * np.ones(2), NoiseCO=1E-6 * np.ones(2), Roll=1E-3)
    sc.register_bpms(np.array([8, 21]), CalError=5E-2 * np.ones(2), Offset=500E-6 * np.ones(2),
                     Noise=10E-6 * np.ones(2), NoiseCO=1E-6 * np.ones(2), Roll=1E-3)
    bpm_update_dict = dict(SumError=5E-2)
    sc.register_bpms(np.array([8, 8]), SumError=5E-2)
    assert_equal(sc.ORD.BPM, np.array([8, 21]))
    assert_equal(sc.SIG.BPM[21], bpm_dict)
    bpm_dict.update(bpm_update_dict)
    assert_equal(sc.SIG.BPM[8], bpm_dict)


def test_register_cavities(at_cell):
    sc = SimulatedComissioning(at_cell)
    rf_dict = dict(FrequencyOffset=5E3, VoltageOffset=5E3, TimeLagOffset=0.5)
    sc.register_cavities(np.array([0, 0, 0]), FrequencyOffset=5E3, VoltageOffset=5E3, TimeLagOffset=0.5)
    assert_equal(sc.ORD.RF, np.array([0]))
    assert_equal(sc.SIG.RF[0], rf_dict)


def test_register_magnets(at_cell):
    sc = SimulatedComissioning(at_cell)
    mag_dict = dict(CalErrorB=np.array([5E-2, 1E-3]), MagnetOffset=200E-6 * np.array([1, 1, 0]),)
    indices = SCgetOrds(sc.RING, "Q")
    sc.register_magnets(np.repeat(indices, 2), CalErrorB=np.array([5E-2, 1E-3]), MagnetOffset=200E-6 * np.array([1, 1, 0]),)
    update_dict = dict(CalErrorA=5E-2)
    sc.register_magnets(indices[2::-1], CalErrorA=5E-2)
    assert_equal(sc.ORD.Magnet, indices)
    assert_equal(sc.SIG.Magnet[indices[-1]], mag_dict)
    mag_dict.update(update_dict)
    assert_equal(sc.SIG.Magnet[indices[0]], mag_dict)


def test_register_supports(at_cell):
    sc = SimulatedComissioning(at_cell)
    mag_dict = dict(SectionOffset=200E-6 * np.array([1, 1, 0]),)
    indices = np.vstack((SCgetOrds(sc.RING, "SF"), SCgetOrds(sc.RING, "SD")))
    sc.register_supports(np.repeat(indices, 2, axis=1), "Section",  Offset=200E-6 * np.array([1, 1, 0]),)
    update_dict1 = dict(SectionRoll=5E-2 * np.array([1, 0, 0]), SectionOffset=100E-6 * np.array([1, 1, 0]))
    update_dict2 = dict(SectionRoll=5E-2 * np.array([1, 0, 0]))
    sc.register_supports(indices[:, 1:], "Section", Offset=100E-6 * np.array([1, 1, 0]), Roll=5E-2 * np.array([[1, 0, 0], [1, 0, 0]]),)
    assert_equal(sc.ORD.Section, indices)
    assert_equal(sc.SIG.Support[indices[0, 0]], mag_dict)
    mag_dict.update(update_dict1)
    assert_equal(sc.SIG.Support[indices[0, 1]], mag_dict)
    assert_equal(sc.SIG.Support[indices[1, 1]], update_dict2)

def test_register_bad_keywords(at_cell):
    sc = SimulatedComissioning(at_cell)
    with pytest.raises(ValueError) as e_info:
        sc.register_bpms(np.array([8, 21]), BadKeyword=5E-2 * np.ones(2),)
    assert "BadKeyword" in str(e_info.value)
    with pytest.raises(ValueError) as e_info:
        sc.register_cavities(np.array([0]), BadKeyword=5E-2 * np.ones(2),)
    assert "BadKeyword" in str(e_info.value)
    with pytest.raises(ValueError) as e_info:
        sc.register_magnets(np.array([0]), BadKeyword=5E-2 * np.ones(2), )
    assert "BadKeyword" in str(e_info.value)
    with pytest.raises(ValueError) as e_info:
        sc.register_supports(np.array([0]), "Girder", BadKeyword=5E-2 * np.ones(2),)
    assert "BadKeyword" in str(e_info.value)
    with pytest.raises(ValueError) as e_info:
        sc.register_supports(np.array([0]), "BadKeyword", Offset=5E-2 * np.ones(2,6),)
    assert "BadKeyword" in str(e_info.value)


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
