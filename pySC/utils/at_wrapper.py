"""
AT wrapper
-------------

This module contains wrappers to all the ``pyAT`` functions, used in ``pySC``,
which are not member functions of used ``pyAT`` objects.
This is due to observed side effects, such as modification of input parameters.

Tracking fuctions ``latice_pass``, ``find_orbit``,  ``find_orbit4``, ``find_orbit6``
index the result the same way as ``get_s_pos``,
i.e. 0 means entrance of the first element, len(elements) means end of the last element.
Function ``get_value_refpts`` indexes elements as usual.


"""

import at
from copy import deepcopy

import numpy as np
from numpy import ndarray
from at import Lattice


def atpass(ring, initial_pos, nturns, refpts, keep_lattice=False):
    return at.lattice_pass(ring.copy(), initial_pos.copy(), nturns, refpts, keep_lattice=keep_lattice)
#lattice, r_in, nturns=1, refpts=None, keep_lattice=False,
#refpts
# 0 means entrance of the first element, len(line) means end of the last element ,
# i.e. 6D positions in the start of elements

def atgetfieldvalues(ring: Lattice, refpts: ndarray, attrname: str, index: int = None):
    return at.get_value_refpts(ring, refpts, attrname, index)  # refpts goes by element index, i.e. starts with 0


# TODO straight in pyAT there are switches between orbit4 and orbit6 (maybe usefull)
def findorbit6(*args, **kwargs):
    return at.find_orbit6(*deepcopy(args), **deepcopy(kwargs))
# i.e. 6D positions in the start of elements

def findorbit4(*args, **kwargs):
    return at.find_orbit4(*deepcopy(args), **deepcopy(kwargs))
# i.e. 6D positions in the start of elements


def findspos(ring):
    return at.get_s_pos(ring=ring)


def atlinopt(*args, **kwargs):
    return at.get_optics(*deepcopy(args), **deepcopy(kwargs))


def twissline(*args, **kwargs):  # TODO find single pass linopt
    raise NotImplementedError


def atloco(*args, **kwargs):
    raise NotImplementedError
