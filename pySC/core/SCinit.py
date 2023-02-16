from pySC.classes import SimulatedComissioning
from at import Lattice
global plotFunctionFlag, SCinjections
def SCinit(RING: Lattice) -> SimulatedComissioning:
    global plotFunctionFlag, SCinjections
    SC = SimulatedComissioning(RING)
    SCinjections = 0 # TODO only used in SCgenBunches
    plotFunctionFlag = []
    return SC
