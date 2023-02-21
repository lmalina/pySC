from pySC.classes import SimulatedComissioning
from at import Lattice


def SCinit(RING: Lattice) -> SimulatedComissioning:
    SC = SimulatedComissioning(RING)
    return SC
