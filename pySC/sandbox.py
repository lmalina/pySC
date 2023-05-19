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
from pySC.core.SCsynchCorrection import SCsynchPhaseCorrection, SCsynchEnergyCorrection
#from pySC.core.SCplotSupport import SCplotSupport
from pySC.core.SCpseudoBBA import SCpseudoBBA
from pySC.core.SCmemberFunctions import SCregisterBPMs, SCregisterCAVs, SCregisterMagnets, SCregisterSupport, SCinit, SCapplyErrors
from pySC.core.SCsanityCheck import SCsanityCheck
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetMags2SetPoints
from pySC.utils import logging_tools

#LOGGER = logging_tools.get_logger(__name__)
LOGGER = logging_tools.get_logger(__name__, level_console=logging_tools.DEBUG)


def create_at_lattice() -> Lattice:
    def _marker(name):
        return at.Marker(name, PassMethod='IdentityPass')
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=1)
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=1)
    sf = at.Sextupole('SF', 0.1, 6.0487, PassMethod='StrMPoleSymplectic4RadPass', MaxOrder=2)
    sd = at.Sextupole('SD', 0.1, -9.5203, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='BndMPoleSymplectic4RadPass')
    d2 = at.Drift('Drift', 0.25)
    d3 = at.Drift('Drift', 0.2)

    cell = at.Lattice([d2, _marker('SectionStart'), _marker('GirderStart'), bend, d3, sf, d3, _marker('GirderEnd'),
                       _marker('GirderStart'), _marker('BPM'), qf, d2, d2, bend, d3, sd, d3, qd, d2, _marker('BPM'),
                       _marker('GirderEnd'), _marker('SectionEnd')], name='Simple FODO cell', energy=2.5E9)
    new_ring = at.Lattice([el.deepcopy() for _ in range(20) for el in cell], name='Simple Ring', energy=2.5E9)
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=149896228.99999985, harmonic_number=50, length=0)
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
                        MagnetOffset=2*200E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
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
                         VoltageOffset=0*5E3,  # [V]
                         TimeLagOffset=0*0.5)  # [m]
    ords = np.vstack((SCgetOrds(SC.RING, 'GirderStart'), SCgetOrds(SC.RING, 'GirderEnd')))
    SC.register_supports(ords, "Girder",
                         Offset=100E-6 * np.array([1, 1, 0]),  # x, y and z, [m]
                         Roll=200E-6 * np.array([1, 0, 0]))  # az, ax and ay, [rad]
    ords = np.vstack((SCgetOrds(SC.RING, 'SectionStart'), SCgetOrds(SC.RING, 'SectionEnd')))
    SC.register_supports(ords, "Section",
                           Offset=100E-6 * np.array([1, 1, 0]))  # x, y and z, [m]
    SC.INJ.beamSize = np.diag(np.array([200E-6, 100E-6, 100E-6, 50E-6, 1E-3, 1E-4]) ** 2)
    SC.SIG.randomInjectionZ = np.array([1E-5, 1E-5, 1E-5, 1E-5, 1E-4, 1E-4])  # [m; rad; m; rad; rel.; m]
    SC.SIG.staticInjectionZ = np.array([1E-3, 1E-4, 1E-3, 1E-4, 1E-3, 1E-3])  # [m; rad; m; rad; rel.; m]
    SC.SIG.Circumference = 2E-4  # relative
    SC.INJ.beamLostAt = 0.6  # relative
    for ord in SCgetOrds(SC.RING, 'Drift'):
        SC.RING[ord].EApertures = 15E-3 * np.array([1, 1])  # [m]
    for ord in SCgetOrds(SC.RING, 'QF|QD|BEND|SF|SD'):
        SC.RING[ord].EApertures = 15E-3 * np.array([1, 1])  # [m]
    SC.RING[SC.ORD.Magnet[50]].EApertures = np.array([6E-3, 3E-3])  # [m]

    SC.plot = False  # TODO: replace with proper global-like variable






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
    eps = 5E-4  # Noise level
    SCgetBPMreading(SC)

    #SC._plot=True
    # SC.plot=True
    # SCgetBPMreading(SC)


    SC = SCfeedbackFirstTurn(SC, Minv1, wiggle_after=5, wiggle_range=np.array([500E-6, 1000E-6]))
    #Minv1test = SCgetPinv(RM1, alpha=5,plot=False) # added 1turn feedback with low regularization to test feedback
    # SC = SCfeedbackRun(SC, Minv1, target=50E-6, maxsteps=30, eps=eps)

    


    SC.INJ.nTurns = 2
    # SC.plot=True

    SC = SCfeedbackStitch(SC, Minv2, nBPMs=5, maxsteps=30, wiggle_range=np.array([500E-6, 1000E-6]))
    # SC = SCfeedbackRun(SC, Minv2, target=500E-6, maxsteps=30, eps=eps)
    SC = SCfeedbackBalance(SC, Minv2, maxsteps=32, eps=eps)

    # for S in np.linspace(0.1, 1, 5):
    #     SC = SCsetMags2SetPoints(SC, sextOrds, False, 2, np.array([S]), method='rel')
    #     try:
    #         SC = SCfeedbackBalance(SC, Minv2, maxsteps=32, eps=eps)
    #     except RuntimeError:
    #         pass

    

    # SC.plot=True
    # CUR = SCfeedbackRun(SC, Minv2, target=0, maxsteps=50, scaleDisp=0E8)

    SC.RING = SCcronoff(SC.RING, 'cavityon')

    # Plot initial phasespace
    #SCplotPhaseSpace(SC, nParticles=10, nTurns=100)

    # RF cavity correction
    for nIter in range(2):
        deltaPhi = SCsynchPhaseCorrection(SC, nTurns=15, nSteps=25, plotResults=True, plotProgress=False)
        SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, 'TimeLag', deltaPhi, method='add')

        deltaF = SCsynchEnergyCorrection(SC, f_range=40E3 * np.array([-1, 1]),  # Frequency range [kHz]
                                         nTurns=20, nSteps=15,  # Number of frequency steps
                                         plotResults=True, plotProgress=False)
        SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, 'Frequency', np.array([deltaF]), method='add')








    # SC.INJ.trackMode = 'ORB'
    # SC.RING = SCcronoff(SC.RING, 'cavityon')

    # MCO = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, trackMode='ORB')
    # print(np.shape(MCO))
    # eta = SCgetModelDispersion(SC, SC.ORD.BPM, SC.ORD.RF)
    # quadOrds = np.tile(SCgetOrds(SC.RING, 'QF|QD'), (2,1))
    # BPMords = np.tile(SC.ORD.BPM, (2,1))
    # SC = SCpseudoBBA(SC, BPMords, quadOrds, np.array([50E-6]))

    # SC.plot=True
    # MinvCO = SCgetPinv(np.column_stack((MCO, 1E8 * eta)), alpha=5)
    # CUR = SCfeedbackRun(SC, MinvCO, target=0, maxsteps=50, scaleDisp=1E8)


    # for alpha in range(10, 0, -1):
    #     MinvCO = SCgetPinv(np.column_stack((MCO, 1E8 * eta)), alpha=alpha)
    #     try:
    #         CUR = SCfeedbackRun(SC, MinvCO, target=0, maxsteps=50, scaleDisp=1E8)
    #     except RuntimeError:
    #         break
    #     B0rms = np.sqrt(np.mean(np.square(SCgetBPMreading(SC)), 1))
    #     Brms = np.sqrt(np.mean(np.square(SCgetBPMreading(CUR)), 1))
    #     if np.mean(B0rms) < np.mean(Brms):
    #         break
    #     SC = CUR