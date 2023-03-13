import numpy as np
from pySC.classes import SimulatedComissioning
from pySC.utils.sc_tools import SCrandnc
from numpy import ndarray


def SCgenBunches(SC: SimulatedComissioning, nParticles=None) -> ndarray:
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    Z = np.tile(np.transpose(SC.INJ.randomInjectionZ * SCrandnc(2, (1, 6)) + SC.INJ.Z0), nParticles)  # TODO every particle should be random
    if nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        Z += np.diag(np.sqrt(V)) @ L @ SCrandnc(3, (6, nParticles))
    return SC.INJ.postFun(Z)
