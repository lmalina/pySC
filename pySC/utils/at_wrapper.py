import at
from copy import deepcopy


def atpass(*args, **kwargs):
    return at.lattice_pass(*deepcopy(args), **deepcopy(kwargs))


def atgetfieldvalues(*args, **kwargs):
    return at.get_value_refpts(*deepcopy(args), **deepcopy(kwargs))


# TODO straight in pyAT there are switches between orbit4 and orbit6 (maybe usefull)
def findorbit6(*args, **kwargs):
    return at.find_orbit(*deepcopy(args), **deepcopy(kwargs))


def findorbit4(*args, **kwargs):
    return at.find_orbit4(*deepcopy(args), **deepcopy(kwargs))


def findspos(*args, **kwargs):
    return at.get_s_pos(*deepcopy(args), **deepcopy(kwargs))


def atlinopt(*args, **kwargs):
    return at.get_optics(*deepcopy(args), **deepcopy(kwargs))


def atmatchchromdelta(*args, **kwargs):  # TODO maybe match
    raise NotImplementedError


def twissline(*args, **kwargs):  # TODO find single pass linopt
    raise NotImplementedError


def atloco(*args, **kwargs):
    raise NotImplementedError
