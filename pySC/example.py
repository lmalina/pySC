import sys

import at
import numpy as np
from at import Lattice
from pySC.at_wrapper import atloco
from pySC.classes import SimulatedComissioning
from pySC.core.SCcronoff import SCcronoff
from pySC.core.SCfeedback import SCfeedbackFirstTurn, SCfeedbackStitch, SCfeedbackRun, SCfeedbackBalance
from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCgetBeamTransmission import SCgetBeamTransmission
from pySC.core.SCgetModelDispersion import SCgetModelDispersion
from pySC.core.SCgetModelRM import SCgetModelRM
from pySC.utils.sc_tools import SCgetOrds, SCgetPinv
from pySC.core.SClocoLib import SClocoLib
from pySC.core.SCplotLattice import SCplotLattice
from pySC.core.SCplotPhaseSpace import SCplotPhaseSpace
#from pySC.core.SCplotSupport import SCplotSupport
from pySC.core.SCpseudoBBA import SCpseudoBBA
from pySC.core.SCmemberFunctions import SCregisterBPMs, SCregisterCAVs, SCregisterMagnets, SCregisterSupport, SCinit, SCapplyErrors
from pySC.core.SCsanityCheck import SCsanityCheck
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetMags2SetPoints
from pySC.core.SCsynchEnergyCorrection import SCsynchEnergyCorrection
from pySC.core.SCsynchPhaseCorrection import SCsynchPhaseCorrection
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def create_at_lattice() -> Lattice:
    def _marker(name):
        return at.Marker(name, PassMethod='IdentityPass')
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=1)
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=1)
    sf = at.Sextupole('SF', 0.1, 6.0487, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=2)
    sd = at.Sextupole('SD', 0.1, -9.5203, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='StrMPoleSymplectic4RadPass')
    d2 = at.Drift('Drift', 0.25)
    d3 = at.Drift('Drift', 0.2)

    cell = at.Lattice([d2, _marker('SectionStart'), _marker('GirderStart'), bend, d3, sf, d3, _marker('GirderEnd'),
                       _marker('GirderStart'), _marker('BPM'), qf, d2, d2, bend, d3, sd, d3, qd, d2, _marker('BPM'),
                       _marker('GirderEnd'), _marker('SectionEnd')], name='Simple FODO cell', energy=2.5E9)
    new_ring = at.Lattice([el.deepcopy() for _ in range(20) for el in cell], name='Simple Ring', energy=2.5E9)
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=1, harmonic_number=50, length=0)
    new_ring.insert(0, rfc)
    return new_ring


if __name__ == "__main__":
    ring = create_at_lattice()
    print(len(ring))
    SC = SimulatedComissioning(ring)
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
    #SCsanityCheck(SC)
    #SCplotLattice(SC, nSectors=10)  # TODO
    SC.apply_errors()
    #SCplotSupport(SC)  # TODO
    SC.RING = SCcronoff(SC.RING, 'cavityoff')
    sextOrds = SCgetOrds(SC.RING, 'SF|SD')
    SC = SCsetMags2SetPoints(SC, sextOrds, False, 2, np.array([0.0]), method='abs')
    RM1 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=1)
    RM2 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=2)
    Minv1 = SCgetPinv(RM1, alpha=50,plot=False)
    Minv2 = SCgetPinv(RM2, alpha=50)
    SC.INJ.nParticles = 1
    SC.INJ.nTurns = 1
    SC.INJ.nShots = 1
    SC.INJ.trackMode = 'TBT'
    eps = 1E-4  # Noise level
    SCgetBPMreading(SC, do_plot=False)
    SC = SCfeedbackFirstTurn(SC, Minv1)
    #Minv1test = SCgetPinv(RM1, alpha=5,plot=False) # added 1turn feedback with low regularization to test feedback
    #SC = SCfeedbackRun(SC, Minv1test, target=50E-6, maxsteps=30, eps=eps)

    SC.INJ.nTurns = 2
    SC = SCfeedbackStitch(SC, Minv2, nBPMs=3, maxsteps=20)
    # SC = SCfeedbackRun(SC, Minv2, target=500E-6, maxsteps=30, eps=eps)
    SC = SCfeedbackBalance(SC, Minv2, maxsteps=32, eps=eps)

    for S in np.linspace(0.1, 1, 5):
        SC = SCsetMags2SetPoints(SC, sextOrds, False, 2, np.array([S]), method='rel')
        try:
            SC = SCfeedbackBalance(SC, Minv2, maxsteps=32, eps=eps)
        except RuntimeError:
            pass

    SC.RING = SCcronoff(SC.RING, 'cavityon')
    SCplotPhaseSpace(SC, nParticles=10, nTurns=100)
    for nIter in range(2):
        [deltaPhi, ERROR] = SCsynchPhaseCorrection(SC, nTurns=5, nSteps=25, plotResults=True, verbose=True)
        if ERROR:
            sys.exit('Phase correction crashed')
        SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, 'TimeLag', deltaPhi, method='add')
        [deltaF, ERROR] = SCsynchEnergyCorrection(SC, range=40E3 * np.array([-1, 1]),  # Frequency range [kHz]
                                                  nTurns=20, nSteps=15,  # Number of frequency steps
                                                  plotResults=True, verbose=True)
        if not ERROR:
            SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, 'Frequency', deltaF, method='add')
        else:
            sys.exit()
    SCplotPhaseSpace(SC, nParticles=10, nTurns=1000)
    [maxTurns, lostCount, ERROR] = SCgetBeamTransmission(SC, nParticles=100, nTurns=10, verbose=True)
    if ERROR:
        sys.exit()
    SC.INJ.trackMode = 'ORB'
    MCO = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, trackMode='ORB')
    eta = SCgetModelDispersion(SC, SC.ORD.BPM, SC.ORD.RF)
    quadOrds = np.tile(SCgetOrds(SC.RING, 'QF|QD'), 2)
    BPMords = np.tile(SC.ORD.BPM, 2)
    SC = SCpseudoBBA(SC, BPMords, quadOrds, 50E-6)
    for alpha in range(10, 0, -1):
        MinvCO = SCgetPinv(np.concatenate((MCO, 1E8 * eta)), alpha=alpha)
        try:
            CUR = SCfeedbackRun(SC, MinvCO, target=0, maxsteps=50, scaleDisp=1E8)
        except RuntimeError:
            break
        B0rms = np.sqrt(np.mean(np.square(SCgetBPMreading(SC)), 1))
        Brms = np.sqrt(np.mean(np.square(SCgetBPMreading(CUR)), 1))
        if np.mean(B0rms) < np.mean(Brms):
            break
        SC = CUR
    SC.RING = SCcronoff(SC.RING, 'cavityon')
    SCplotPhaseSpace(SC, nParticles=10, nTurns=1000)
    [maxTurns, lostCount, ERROR] = SCgetBeamTransmission(SC, nParticles=100, nTurns=10, verbose=True)
    if ERROR:
        sys.exit()
    CMstep = 1E-4  # [rad]
    RFstep = 1E3  # [Hz]
    [RINGdata, LOCOflags, Init] = SClocoLib('setupLOCOmodel', SC,
                                            'Dispersion', 'Yes',
                                            'HorizontalDispersionWeight', .1E2,
                                            'VerticalDispersionWeight', .1E2)
    [BPMData, CMData] = SClocoLib('getBPMCMstructure', SC, CMstep,
                                  {'BPM', 'FitGains', 'Yes'},
                                  {'CM', 'FitKicks', 'Yes'})
    LOCOmeasData = SClocoLib('getMeasurement', SC, CMstep, RFstep, SC.ORD.BPM, SC.ORD.CM)
    FitParameters = SClocoLib('setupFitparameters', SC, Init.SC.RING, RINGdata, RFstep,
                              {SCgetOrds(SC.RING, 'QF'), 'normal', 'individual', 1E-3},
                              # {Ords, normal/skew, ind/fam, deltaK}
                              {SCgetOrds(SC.RING, 'QD'), 'normal', 'individual',
                               1E-4})  # {Ords, normal/skew, ind/fam, deltaK}
    for n in range(6):
        _, BPMData, CMData, FitParameters, LOCOflags, RINGdata = atloco(LOCOmeasData,  BPMData,  CMData,  FitParameters,  LOCOflags,  RINGdata)
        SC = SClocoLib('applyLatticeCorrection', SC, FitParameters)
        SC = SClocoLib('applyOrbitCorrection', SC)
        SClocoLib('plotStatus', SC, Init, BPMData, CMData)
        if n == 3:
            LOCOflags.Coupling = 'Yes'
            FitParameters = SClocoLib('setupFitparameters', SC, Init.SC.RING, RINGdata, RFstep,
                                      {SCgetOrds(SC.RING, 'QF'), 'normal', 'individual', 1E-3},
                                      {SCgetOrds(SC.RING, 'QD'), 'normal', 'individual', 1E-4},
                                      {SC.ORD.SkewQuad, 'skew', 'individual', 1E-3})
