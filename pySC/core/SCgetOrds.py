import re
import numpy as np
from at import Lattice


def SCgetOrds(RING: Lattice, rx, verbose=False):  # TODO consider forbiding list of regexes
    if isinstance(rx, str):
        if verbose:
            return [_print_elem_get_index(ind, el) for ind, el in enumerate(RING) if re.search(rx, el.FamName)]
        return np.array([ind for ind, el in enumerate(RING) if re.search(rx, el.FamName)])
    return [SCgetOrds(RING, r, verbose=verbose) for r in rx]


def _print_elem_get_index(ind, el):
    print(f'Matched: {el.FamName}')
    return ind
