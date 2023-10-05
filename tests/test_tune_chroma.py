import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import at
import pySC
from pySC.correction.tune import tune_scan, fit_tune
from pySC.correction.chroma import fit_chroma
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds


def test_fit_tune(sc):
    sc = fit_tune(sc, q_ords=[SCgetOrds(sc.RING, 'QF1'), SCgetOrds(sc.RING, 'QD')], target_tune=numpy.array([0.16, 0.21]))
    return sc

def test_fit_chroma(sc):
    sc = fit_chroma()
    return sc