import numpy as np
from pySC.classes import SimulatedComissioning
from pySC.utils.sc_utils import SCrandnc
from numpy import ndarray


def SCgenBunches(SC: SimulatedComissioning) -> ndarray:
    Z = np.tile(np.transpose(SC.INJ.randomInjectionZ * SCrandnc(2, (1, 6)) + SC.INJ.Z0), SC.INJ.nParticles)  # TODO
    if SC.INJ.nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        Z += np.diag(np.sqrt(V)) @ L @ SCrandnc(3, (6, SC.INJ.nParticles))
    return SC.INJ.postFun(Z)
