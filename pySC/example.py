import at
import numpy as np
from pySC.correction.bba import trajectory_bba, fake_bba

from at import Lattice
from pySC.utils.at_wrapper import atloco
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.correction import orbit_trajectory
from pySC.core.beam import bpm_reading, beam_transmission
from pySC.correction.tune import tune_scan
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.utils.sc_tools import SCgetOrds, SCgetPinv
from pySC.correction.loco_wrapper import (loco_model, loco_fit_parameters, apply_lattice_correction, loco_measurement,
                                          loco_bpm_structure, loco_cm_structure)
from pySC.plotting.plot_phase_space import plot_phase_space
from pySC.plotting.plot_support import plot_support
from pySC.plotting.plot_lattice import plot_lattice
from pySC.core.lattice_setting import switch_cavity_and_radiation
from pySC.correction.rf import correct_rf_phase, correct_rf_frequency, phase_and_energy_error
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def create_at_lattice() -> Lattice:
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
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=500653404.8599995, harmonic_number=167, length=0)
    new_ring.insert(0, rfc)
    new_ring.enable_6d()
    at.set_cavity_phase(new_ring)
    at.set_rf_frequency(new_ring)
    return new_ring


if __name__ == "__main__":
    ring = at.Lattice(create_at_lattice())
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)
    # at.summary(ring)
    ords = SCgetOrds(SC.RING, 'BPM')
    SC.register_bpms(ords, CalError=5E-2 * np.ones(2),  # x and y, relative
                     Offset=500E-6 * np.ones(2),  # x and y, [m]
                     Noise=10E-6 * np.ones(2),  # x and y, [m]
                     NoiseCO=1E-6 * np.ones(2),  # x and y, [m]
                     Roll=1E-3)  # az, [rad]
    ords = SCgetOrds(SC.RING, 'QF')
    SC.register_magnets(ords, HCM=1E-3,  # [rad]
                        CalErrorB=np.array([5E-2, 1E-3]),  # relative
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = SCgetOrds(SC.RING, 'QD')
    SC.register_magnets(ords, VCM=1E-3,  # [rad]
                        CalErrorA=np.array([5E-2, 0]),  # relative
                        CalErrorB=np.array([0, 1E-3]),  # relative
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = SCgetOrds(SC.RING, 'BEND')
    SC.register_magnets(ords,
                        BendingAngle=1E-3,  # relative
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = SCgetOrds(SC.RING, 'SF|SD')
    SC.register_magnets(ords,
                        SkewQuad=0.1,  # [1/m]
                        CalErrorA=np.array([0, 1E-3, 0]),  # relative
                        CalErrorB=np.array([0, 0, 1E-3]),  # relative
                        MagnetOffset=200E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                        MagnetRoll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = SCgetOrds(SC.RING, 'RFCav')
    SC.register_cavities(ords, FrequencyOffset=5E3,  # [Hz]
                         VoltageOffset=5E3,  # [V]
                         TimeLagOffset=0.5)  # [m]
    ords = np.vstack((SCgetOrds(SC.RING, 'GirderStart'), SCgetOrds(SC.RING, 'GirderEnd')))
    SC.register_supports(ords, "Girder",
                         Offset=100E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                         Roll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = np.vstack((SCgetOrds(SC.RING, 'SectionStart'), SCgetOrds(SC.RING, 'SectionEnd')))
    SC.register_supports(ords, "Section",
                           Offset=100E-6 * np.array([1, 1, 0]))  # x, y and z, [m]
    SC.INJ.beamSize = np.diag(np.array([200E-6, 100E-6, 100E-6, 50E-6, 1E-3, 1E-4]) ** 2)
    SC.SIG.randomInjectionZ = np.array([1E-4, 1E-5, 1E-4, 1E-5, 1E-4, 1E-4])  # [m; rad; m; rad; rel.; m]
    SC.SIG.staticInjectionZ = np.array([1E-3, 1E-4, 1E-3, 1E-4, 1E-3, 1E-3])  # [m; rad; m; rad; rel.; m]
    SC.SIG.Circumference = 2E-4  # relative
    SC.INJ.beamLostAt = 0.6  # relative
    for ord in SCgetOrds(SC.RING, 'Drift'):
        SC.RING[ord].EApertures = 13E-3 * np.array([1, 1])  # [m]
    for ord in SCgetOrds(SC.RING, 'QF|QD|BEND|SF|SD'):
        SC.RING[ord].EApertures = 10E-3 * np.array([1, 1])  # [m]
    SC.RING[SC.ORD.Magnet[50]].EApertures = np.array([6E-3, 3E-3])  # [m]

    plot_lattice(SC, s_range=np.array([0, 20]))
    SC.apply_errors()
    SC.verify_structure()
    plot_support(SC)

    SC.RING = switch_cavity_and_radiation(SC.RING, 'cavityoff')
    sextOrds = SCgetOrds(SC.RING, 'SF|SD')
    SC.set_magnet_setpoints(sextOrds, 0.0, False, 2, method='abs')
    RM1 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=1)
    RM2 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=2)
    SC.INJ.nParticles = 1
    SC.INJ.nTurns = 1
    SC.INJ.nShots = 1
    SC.INJ.trackMode = 'TBT'
    eps = 5E-4  # Noise level
    bpm_reading(SC)
    SC = orbit_trajectory.first_turn(SC, RM1, alpha=50)

    SC.INJ.nTurns = 2
    SC = orbit_trajectory.stitch(SC, RM2, n_bpms=3, maxsteps=20, alpha=50)
    # SC = orbit_trajectory.correct(SC, RM2, target=300E-6, maxsteps=30, eps=eps, alpha=50)
    SC = orbit_trajectory.balance(SC, RM2, maxsteps=32, eps=eps, alpha=50)

    # plot_cm_strengths(SC)
    # Performing trajectory BBA
    SC.INJ.nParticles = 1
    quadOrds = np.tile(SCgetOrds(SC.RING, 'QF|QD'), (2, 1))
    BPMords = np.tile(SC.ORD.BPM, (2, 1))
    SC, bba_offsets, bba_offset_errors = trajectory_bba(SC, BPMords, quadOrds, q_ord_phase=SCgetOrds(SC.RING, 'QF|QD')[0],
                                                        q_ord_setpoints=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                                                        magnet_strengths=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                                                        dipole_compensation=True, plot_results=True)

    # Turning on the sextupoles
    for rel_setting in np.linspace(0.1, 1, 5):
        SC.set_magnet_setpoints(sextOrds, rel_setting, False, 2, method='rel')
        try:
            SC = orbit_trajectory.balance(SC, RM2, maxsteps=32, eps=eps, alpha=50)
        except RuntimeError:
            pass

    SC.RING = switch_cavity_and_radiation(SC.RING, 'cavityon')

    # Plot initial phasespace
    plot_phase_space(SC, nParticles=10, nTurns=100)

    # RF cavity correction
    for nIter in range(2):
        SC.INJ.nTurns = 5
        SC = correct_rf_phase(SC, n_steps=25, plot_results=False, plot_progress=False)
        SC.INJ.nTurns = 15
        SC = correct_rf_frequency(SC, n_steps=15, f_range=4E3 * np.array([-1, 1]), plot_results=False,
                                  plot_progress=False)

    # Plot phasespace after RF correction
    plot_phase_space(SC, nParticles=10, nTurns=100)
    [maxTurns, lostCount] = beam_transmission(SC, nParticles=100, nTurns=10)

    # Faking-BBA
    quadOrds = np.tile(SCgetOrds(SC.RING, 'QF|QD'), (2, 1))
    BPMords = np.tile(SC.ORD.BPM, (2, 1))
    SC = fake_bba(SC, BPMords, quadOrds, fake_offset=np.array([50E-6, 50E-6]))

    # Orbit correction
    SC.INJ.trackMode = 'ORB'
    MCO = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, trackMode='ORB')
    eta = SCgetModelDispersion(SC, SC.ORD.BPM, SC.ORD.RF)
    resp_with_disp = np.column_stack((MCO, 1E8 * eta))
    for alpha in range(10, 0, -1):
        try:
            CUR = orbit_trajectory.correct(SC, resp_with_disp, target=0, maxsteps=50, scaleDisp=1E8, alpha=alpha)
        except RuntimeError:
            break
        B0rms = np.sqrt(np.mean(np.square(bpm_reading(SC)[0]), axis=1))
        Brms = np.sqrt(np.mean(np.square(bpm_reading(CUR)[0]), axis=1))
        if np.mean(B0rms) < np.mean(Brms):
            break
        SC = CUR
    SC.RING = switch_cavity_and_radiation(SC.RING, 'cavityon')
    plot_phase_space(SC, nParticles=10, nTurns=1000)
    maxTurns, lostCount = beam_transmission(SC, nParticles=100, nTurns=200, plot=True)
    SC, _, _, _ = tune_scan(SC, np.vstack((SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD'))),
                            np.outer(np.ones(2), 1 + np.linspace(-0.01, 0.01, 51)), do_plot=False, nParticles=100,
                            nTurns=200)
    CMstep = 1E-4  # [rad] # TODO later in the structure it is in mrad, ???
    RFstep = 1E3  # [Hz]
    ring_data, loco_flags, init = loco_model(SC, Dispersion=True, HorizontalDispersionWeight=.1E2,
                                             VerticalDispersionWeight= .1E2)
    bpm_data = loco_bpm_structure(SC, FitGains=True)
    cm_data = loco_cm_structure(SC, CMstep, FitKicks=True)
    loco_meas_data = loco_measurement(SC, CMstep, RFstep, SC.ORD.BPM, SC.ORD.CM)
    fit_parameters = loco_fit_parameters(SC, init.SC.RING, ring_data, RFstep,
                                         [SCgetOrds(SC.RING, 'QF'), False, 'individual', 1E-3],
                                         # {Ords, normal/skew, ind/fam, deltaK}
                                         [SCgetOrds(SC.RING, 'QD'), False, 'individual', 1E-4])

    for n in range(6):
        _, bpm_data, cm_data, fit_parameters, loco_flags, ring_data = atloco(loco_meas_data, bpm_data, cm_data,
                                                                             fit_parameters, loco_flags, ring_data)
        SC = apply_lattice_correction(SC, fit_parameters)
        SC = orbit_trajectory.correct(SC, MCO, alpha=50, target=0, maxsteps=30)
        if n == 3:
            loco_flags.Coupling = True
            fit_parameters = loco_fit_parameters(SC, init.SC.RING, ring_data, RFstep,
                                                 [SCgetOrds(SC.RING, 'QF'), False, 'individual', 1E-3],
                                                 [SCgetOrds(SC.RING, 'QD'), False, 'individual', 1E-4],
                                                 [SC.ORD.SkewQuad, True, 'individual', 1E-3])
