from at import Lattice


def SCcronoff(RING: Lattice, *args: str) -> Lattice:
    valid_args = ('radiationoff', 'radiationon', 'cavityoff', 'cavityon')
    if invalid_args := [arg for arg in args if arg not in valid_args]:
        raise ValueError(f"Unknown arguments found: {invalid_args}")
    for mode in args:
        if mode == 'radiationoff':
            for ind in range(len(RING)):
                if RING[ind].PassMethod == 'BndMPoleSymplectic4RadPass':
                    RING[ind].PassMethod = 'BndMPoleSymplectic4Pass'
                elif RING[ind].PassMethod == 'BndMPoleSymplectic4E2RadPass':
                    RING[ind].PassMethod = 'BndMPoleSymplectic4E2Pass'
                elif RING[ind].PassMethod == 'StrMPoleSymplectic4RadPass':
                    RING[ind].PassMethod = 'StrMPoleSymplectic4Pass'
        elif mode == 'radiationon':
            for ind in range(len(RING)):
                if RING[ind].PassMethod == 'BndMPoleSymplectic4Pass':
                    RING[ind].PassMethod = 'BndMPoleSymplectic4RadPass'
                elif RING[ind].PassMethod == 'BndMPoleSymplectic4E2Pass':
                    RING[ind].PassMethod = 'BndMPoleSymplectic4E2RadPass'
                elif RING[ind].PassMethod == 'StrMPoleSymplectic4Pass':
                    RING[ind].PassMethod = 'StrMPoleSymplectic4RadPass'
        elif mode == 'cavityoff':
            for ind in range(len(RING)):
                if 'Frequency' in RING[ind]:
                    RING[ind].PassMethod = 'IdentityPass'
        elif mode == 'cavityon':
            for ind in range(len(RING)):
                if 'Frequency' in RING[ind]:
                    RING[ind].PassMethod = 'RFCavityPass'
    return RING
