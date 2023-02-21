from at import Lattice


def SCcronoff(ring: Lattice, *args: str) -> Lattice:  # TODO some at methods do that?
    valid_args = ('radiationoff', 'radiationon', 'cavityoff', 'cavityon')
    if invalid_args := [arg for arg in args if arg not in valid_args]:
        raise ValueError(f"Unknown arguments found: {invalid_args}"
                         f"Available options are: {valid_args}")
    for mode in args:
        if mode == 'radiationoff':
            for ind in range(len(ring)):
                if ring[ind].PassMethod == 'BndMPoleSymplectic4RadPass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4Pass'
                elif ring[ind].PassMethod == 'BndMPoleSymplectic4E2RadPass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4E2Pass'
                elif ring[ind].PassMethod == 'StrMPoleSymplectic4RadPass':
                    ring[ind].PassMethod = 'StrMPoleSymplectic4Pass'
        elif mode == 'radiationon':
            for ind in range(len(ring)):
                if ring[ind].PassMethod == 'BndMPoleSymplectic4Pass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4RadPass'
                elif ring[ind].PassMethod == 'BndMPoleSymplectic4E2Pass':
                    ring[ind].PassMethod = 'BndMPoleSymplectic4E2RadPass'
                elif ring[ind].PassMethod == 'StrMPoleSymplectic4Pass':
                    ring[ind].PassMethod = 'StrMPoleSymplectic4RadPass'
        elif mode == 'cavityoff':
            for ind in range(len(ring)):
                if hasattr(ring[ind], 'Frequency'):
                    ring[ind].PassMethod = 'IdentityPass'
        elif mode == 'cavityon':
            for ind in range(len(ring)):
                if hasattr(ring[ind], 'Frequency'):
                    ring[ind].PassMethod = 'RFCavityPass'
    return ring
