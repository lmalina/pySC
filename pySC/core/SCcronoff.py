def SCcronoff(RING,*varargin):
    for i in range(len(varargin)):
        mode = varargin[i]
        if mode == 'radiationoff':
            for ord in range(len(RING)):
                if RING[ord].PassMethod == 'BndMPoleSymplectic4RadPass':
                    RING[ord].PassMethod = 'BndMPoleSymplectic4Pass'
                elif RING[ord].PassMethod == 'BndMPoleSymplectic4E2RadPass':
                    RING[ord].PassMethod = 'BndMPoleSymplectic4E2Pass'
                elif RING[ord].PassMethod == 'StrMPoleSymplectic4RadPass':
                    RING[ord].PassMethod = 'StrMPoleSymplectic4Pass'
        elif mode == 'radiationon':
            for ord in range(len(RING)):
                if RING[ord].PassMethod == 'BndMPoleSymplectic4Pass':
                    RING[ord].PassMethod = 'BndMPoleSymplectic4RadPass'
                elif RING[ord].PassMethod == 'BndMPoleSymplectic4E2Pass':
                    RING[ord].PassMethod = 'BndMPoleSymplectic4E2RadPass'
                elif RING[ord].PassMethod == 'StrMPoleSymplectic4Pass':
                    RING[ord].PassMethod = 'StrMPoleSymplectic4RadPass'
        elif mode == 'cavityoff':
            for ord in range(len(RING)):
                if 'Frequency' in RING[ord]:
                    RING[ord].PassMethod = 'IdentityPass'
        elif mode == 'cavityon':
            for ord in range(len(RING)):
                if 'Frequency' in RING[ord]:
                    RING[ord].PassMethod = 'RFCavityPass'
        else:
            print('SCcronoff: mode %s not recognized. RING unchanged.' % mode)