import re


def SCgetOrds(RING, rx, verbose=0):
    if isinstance(rx, str):
        if verbose:
            return [_print_elem_get_index(ind, el) for ind, el in enumerate(RING) if re.search(rx, el.FamName)]
        return [ind for ind, el in enumerate(RING) if re.search(rx, el.FamName)]
    return [SCgetOrds(RING, r, verbose=0) for r in rx]


def _print_elem_get_index(ind, el):
    print(f'Matched: {el.FamName}')
    return ind