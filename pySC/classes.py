import numpy as np
from at import Lattice


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key in self:
            if isinstance(self[key], dict):
                self[key] = DotDict(self[key])

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return super(DotDict, self).__getitem__(key)
        except KeyError as e:
            raise AttributeError(e).with_traceback(e.__traceback__) from e


class SimulatedComissioning(DotDict):
    def __init__(self, RING: Lattice):
        super(SimulatedComissioning, self).__init__()
        global plotFunctionFlag
        global SCinjections
        self.RING = RING
        self.IDEALRING = RING
        self.INJ = Injection()
        self.SIG = DotDict()
        self.ORD = DotDict()
        SCinjections = 0
        plotFunctionFlag = []


class Injection(DotDict):
    def __init__(self):
        super(Injection, self).__init__()
        self.beamLostAt = 1
        self.Z0ideal = np.zeros(6)
        self.Z0 = np.zeros(6)
        self.beamSize = np.zeros((6, 6))
        self.randomInjectionZ = np.zeros(6)
        self.nParticles = 1
        self.nTurns = 1
        self.nShots = 1
        self.trackMode = 'TBT'


if __name__ == "__main__":
    SC = SimulatedComissioning(Lattice([], energy=6e9))
    print("Atribute call:")
    print(SC.INJ)


    print(plotFunctionFlag)

    print("Key call:")
    print(SC["INJ"])
