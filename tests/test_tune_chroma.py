import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import at
import pySC
from pySC.correction.tune import tune_scan
from pySC.correction.chroma import fit_chroma
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds

path = '/machfs/hoummi/pySCDOC/test_files/'
filename = 'S28Dmerged.mat'
key = 'LOW_EMIT_RING_INJ'
latticef = path + filename
ring = at.load_lattice(latticef,key=key)
optics = ring.get_optics() #dictionary

sc = SimulatedCommissioning(ring)

#%quad_ord = SCgetOrds(sc, {'QF1, 'QD2'})
#rel_quad_changes =
#tune_scan(sc, quad_ords,rel_quad_changes)

SF = SCgetOrds(sc.RING, 'SF2')
SD = SCgetOrds(sc.RING, 'SD1A');
sc.register_cavities(SCgetOrds(sc.RING, 'RFCav'),
                     FrequencyOffset=5E3,
                     VoltageOffset=5E3,
                     TimeLagOffset=0.5)
sc.register_magnets(SCgetOrds(sc.RING, 'QF'), HCM=1E-3,
                    CalErrorB=np.array([5E-2, 1E-3]),
                    MagnetOffset=200E-6 * np.array([1, 1, 0]),
                    MagnetRoll=200E-6 * np.array([1, 0, 0]))
sc.register_magnets(SCgetOrds(sc.RING, 'QD'), VCM=1E-3,
                    CalErrorA=np.array([5E-2, 0]),
                    CalErrorB=np.array([0, 1E-3]),
                    MagnetOffset=200E-6 * np.array([1, 1, 0]),
                    MagnetRoll=200E-6 * np.array([1, 0, 0]))

sc.register_magnets(SCgetOrds(sc.RING, 'SF'))
sc.register_magnets(SCgetOrds(sc.RING, 'SD'))
sc = fit_chroma(sc, s_ords=[SCgetOrds(sc.RING, 'SF2'), SCgetOrds(sc.RING, 'SD1A')], target_chroma = np.array([1,1]))
