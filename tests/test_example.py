import pytest
from tests.test_at_wrapper import at_lattice
import numpy as np
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.correction import orbit_trajectory
from pySC.core.beam import bpm_reading, beam_transmission
from pySC.correction.tune import tune_scan
from pySC.correction.bba import trajectory_bba, fake_bba, _get_bpm_offset_from_mag
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.utils.sc_tools import SCgetOrds
from pySC.core.lattice_setting import switch_cavity_and_radiation
from pySC.correction.rf import correct_rf_phase, correct_rf_frequency


def test_example(at_lattice):
    np.random.seed(12345678)
    sc = SimulatedCommissioning(at_lattice)
    sc.register_bpms(SCgetOrds(sc.RING, 'BPM'),
                     CalError=5E-2 * np.ones(2),
                     Offset=500E-6 * np.ones(2),
                     Noise=10E-6 * np.ones(2),
                     NoiseCO=1E-6 * np.ones(2),
                     Roll=1E-3)
    sc.register_magnets(SCgetOrds(sc.RING, 'QF'),
                        HCM=1E-3,
                        CalErrorB=np.array([5E-2, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc.register_magnets(SCgetOrds(sc.RING, 'QD'), VCM=1E-3,
                        CalErrorA=np.array([5E-2, 0]),
                        CalErrorB=np.array([0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc.register_magnets(SCgetOrds(sc.RING, 'BEND'),
                        BendingAngle=1E-3,  # relative
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc.register_magnets(SCgetOrds(sc.RING, 'SF|SD'),
                        SkewQuad=0.1,
                        CalErrorA=np.array([0, 1E-3, 0]),
                        CalErrorB=np.array([0, 0, 1E-3]),
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))
    sc.register_cavities(SCgetOrds(sc.RING, 'RFCav'), FrequencyOffset=5E3,
                         VoltageOffset=5E3,
                         TimeLagOffset=0.5)
    sc.register_supports(np.vstack((SCgetOrds(sc.RING, 'GirderStart'), SCgetOrds(sc.RING, 'GirderEnd'))),
                         "Girder",
                         Offset=100E-6 * np.array([1, 1, 0]),
                         Roll=200E-6 * np.array([1, 0, 0]))
    sc.register_supports(np.vstack((SCgetOrds(sc.RING, 'SectionStart'), SCgetOrds(sc.RING, 'SectionEnd'))),
                         "Section",
                         Offset=100E-6 * np.array([1, 1, 0]))
    sc.INJ.beamSize = np.diag(np.array([200E-6, 100E-6, 100E-6, 50E-6, 1E-3, 1E-4]) ** 2)
    sc.SIG.randomInjectionZ = np.array([1E-4, 1E-5, 1E-4, 1E-5, 1E-4, 1E-4])
    sc.SIG.staticInjectionZ = np.array([1E-3, 1E-4, 1E-3, 1E-4, 1E-3, 1E-3])
    sc.SIG.Circumference = 2E-4
    sc.INJ.beamLostAt = 0.6
    for ord in SCgetOrds(sc.RING, 'Drift'):
        sc.RING[ord].EApertures = 13E-3 * np.array([1, 1])
    for ord in SCgetOrds(sc.RING, 'QF|QD|BEND|SF|SD'):
        sc.RING[ord].EApertures = 10E-3 * np.array([1, 1])
    sc.RING[sc.ORD.Magnet[50]].EApertures = np.array([6E-3, 3E-3])

    sc.apply_errors()
    sc.verify_structure()
    sc.RING = switch_cavity_and_radiation(sc.RING, 'cavityoff')
    sext_ords = SCgetOrds(sc.RING, 'SF|SD')
    sc.set_magnet_setpoints(sext_ords, 0.0, False, 2, method='abs')
    rm1 = SCgetModelRM(sc, sc.ORD.BPM, sc.ORD.CM, nTurns=1)
    rm2 = SCgetModelRM(sc, sc.ORD.BPM, sc.ORD.CM, nTurns=2)
    sc.INJ.nParticles = 1
    sc.INJ.nTurns = 1
    sc.INJ.nShots = 1
    sc.INJ.trackMode = 'TBT'
    eps = 5E-4  # Noise level
    sc = orbit_trajectory.first_turn(sc, rm1, alpha=50)

    sc.INJ.nTurns = 2
    sc = orbit_trajectory.stitch(sc, rm2, alpha=50, n_bpms=3, maxsteps=20)
    sc = orbit_trajectory.balance(sc, rm2, alpha=50, maxsteps=32, eps=eps)

    # Performing BBA
    sc.INJ.nParticles = 1
    quadOrds = np.tile(SCgetOrds(sc.RING, 'QF|QD'), (2, 1))
    BPMords = np.tile(sc.ORD.BPM, (2, 1))
    init_offsets = _get_bpm_offset_from_mag(sc.RING, BPMords, quadOrds)
    sc, bba_offsets, bba_offset_errors = trajectory_bba(sc, np.tile(sc.ORD.BPM, (2, 1)), np.tile(SCgetOrds(sc.RING, 'QF|QD'), (2, 1)),
                                                        q_ord_phase=SCgetOrds(sc.RING, 'QF|QD')[0],
                                                        q_ord_setpoints=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                                                        plot_results=True,
                                                        magnet_strengths=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                                                        dipole_compensation=False)
    final_offsets = _get_bpm_offset_from_mag(sc.RING, BPMords, quadOrds)
    assert np.std(init_offsets) > 2 * np.std(final_offsets)

    # Turning on the sextupoles
    for rel_setting in np.linspace(0.1, 1, 5):
        sc.set_magnet_setpoints(sext_ords, rel_setting, False, 2, method='rel')
        sc = orbit_trajectory.balance(sc, rm2, alpha=50, maxsteps=32, eps=eps)

    sc.RING = switch_cavity_and_radiation(sc.RING, 'cavityon')

    # RF cavity correction
    sc.INJ.nTurns = 5
    sc = correct_rf_phase(sc, n_steps=15)
    sc.INJ.nTurns = 15
    sc = correct_rf_frequency(sc, n_steps=15, f_range=4E3 * np.array([-1, 1]))

    sc = fake_bba(sc, np.tile(sc.ORD.BPM, (2, 1)), np.tile(SCgetOrds(sc.RING, 'QF|QD'), (2, 1)),
                  fake_offset=np.array([50E-6, 50E-6]))

    # Orbit correction
    sc.INJ.trackMode = 'ORB'
    mco = SCgetModelRM(sc, sc.ORD.BPM, sc.ORD.CM, trackMode='ORB')
    eta = SCgetModelDispersion(sc, sc.ORD.BPM, sc.ORD.RF)
    resp_with_disp = np.column_stack((mco, 1E8 * eta))
    for alpha in range(10, 0, -1):
        try:
            cur = orbit_trajectory.correct(sc, resp_with_disp, alpha=50, target=0, maxsteps=50, scaleDisp=1E8)
        except RuntimeError:
            break
        B0rms = np.sqrt(np.mean(np.square(bpm_reading(sc)[0]), axis=1))
        Brms = np.sqrt(np.mean(np.square(bpm_reading(cur)[0]), axis=1))
        if np.mean(B0rms) < np.mean(Brms):
            break
        sc = cur
    sc.RING = switch_cavity_and_radiation(sc.RING, 'cavityon')
    max_turns, fraction_survived = beam_transmission(sc, nParticles=100, nTurns=200, plot=True)
    sc, _, _, _ = tune_scan(sc, np.vstack((SCgetOrds(sc.RING, 'QF'), SCgetOrds(sc.RING, 'QD'))),
                            np.outer(np.ones(2), 1 + np.linspace(-0.01, 0.01, 51)), do_plot=False, nParticles=50,
                            nTurns=100, target=0.95)
    max_turns, fraction_survived = beam_transmission(sc, nParticles=100, nTurns=200, plot=True)
    assert max_turns == 200
    assert fraction_survived[-1] >= 0.95
