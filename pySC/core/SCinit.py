from pySC.classes import SimulatedComissioning

def SCinit(RING):
    global plotFunctionFlag, SCinjections
    SC = SimulatedComissioning(RING)
    SCinjections = 0 # TODO only used in SCgenBunches
    plotFunctionFlag = []
    return SC
