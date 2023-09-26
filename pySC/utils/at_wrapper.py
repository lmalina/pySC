"""
AT wrapper
-------------

This module contains wrappers to all the ``pyAT`` functions, used in ``pySC``,
which are not member functions of used ``pyAT`` objects.
This is due to observed side effects, such as modification of input parameters.

Tracking fuctions ``latice_pass``, ``find_orbit4``, ``find_orbit6``
index the result the same way as ``get_s_pos``,
i.e. 0 means entrance of the first element, len(elements) means end of the last element.
Function ``get_value_refpts`` indexes elements as usual.


"""

import at
from copy import deepcopy

from numpy import ndarray
from at import Lattice


def atpass(ring: Lattice, init_pos: ndarray, nturns: int, refpts: ndarray, keep_lattice: bool = False):
    return at.lattice_pass(lattice=ring.copy(), r_in=init_pos.copy(), nturns=nturns, refpts=refpts,
                           keep_lattice=keep_lattice)


def patpass(ring: Lattice, init_pos: ndarray, nturns: int, refpts: ndarray, keep_lattice: bool = False):
    return at.patpass(lattice=ring.copy(), r_in=init_pos.copy(), nturns=nturns, refpts=refpts,
                      keep_lattice=keep_lattice)


def atgetfieldvalues(ring: Lattice, refpts: ndarray, attrname: str, index: int = None):
    return at.get_value_refpts(ring, refpts, attrname, index)


def findorbit6(ring: Lattice, refpts: ndarray = None, keep_lattice: bool = False, **kwargs):
    return at.find_orbit6(ring=ring.copy(), refpts=refpts, keep_lattice=keep_lattice, **kwargs)


def findorbit4(ring: Lattice, dp: float = 0.0, refpts: ndarray = None,  keep_lattice: bool = False, **kwargs):
    return at.find_orbit4(ring=ring.copy(), dp=dp, refpts=refpts, keep_lattice=keep_lattice, **kwargs)


def findspos(ring: Lattice):
    return at.get_s_pos(ring=ring)


def atlinopt(*args, **kwargs):
    return at.get_optics(*deepcopy(args), **deepcopy(kwargs))


def twissline(*args, **kwargs):  # TODO find single pass linopt
    raise NotImplementedError


def atloco(*args, **kwargs):
    raise NotImplementedError
