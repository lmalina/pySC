"""
pySC package
~~~~~~~~~~~~~~~~

pySC

"""
import at
__title__ = "pySC"
__description__ = "Python version of Simulated Commissioning toolkit for synchrotrons (https://github.com/ThorstenHellert/SC) "
__url__ = "https://github.com/lmalina/pySC"
__version__ = "0.1.0"
__author__ = "lmalina"
__author_email__ = "lukas.malina@desy.de"

atgetfieldvalues = at.get_value_refpts  # TODO are these correct methods?
atpass = at.atpass
findorbit6 = at.find_orbit6
findorbit4 = at.find_orbit4
findspos = at.get_s_pos
atlinopt = at.get_optics
# atmatchchromdelta
# twissline # TODO find single pass linopt
#pol2cart
#cart2pol
#rem
#spiral

#fminsearch
#optimset
#circshift
#getRingAperture
__all__ = [__version__]
