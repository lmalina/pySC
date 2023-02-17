import re
import numpy as np
from at import Lattice
from numpy import ndarray


def SCgetOrds(ring: Lattice, regex: str, verbose: bool = False) -> ndarray:
    """
    Returns the indices of the elements in the ring whose names match the regex.

    Parameters
    ----------
    ring : Lattice
        The ring to search.
    regex : str
        The regular expression to match.
    verbose : bool, optional
        If True, prints the names of matched elements.

    Returns
    -------
    ndarray
        The indices of the matched elements.
    """
    if verbose:
        return np.array([_print_elem_get_index(ind, el) for ind, el in enumerate(ring) if re.search(regex, el.FamName)],
                        dtype=int)
    return np.array([ind for ind, el in enumerate(ring) if re.search(regex, el.FamName)], dtype=int)


def _print_elem_get_index(ind, el):
    print(f'Matched: {el.FamName}')
    return ind
