import numpy as np


def SCsanityCheck(SC):  # TODO this translated mostly for reference, will change a lot
    if 'ORD' not in SC:
        raise ValueError('Nothing is registered.')
    else:
        if 'BPM' not in SC['ORD']:
            print('No BPMs registered. Use ''SCregisterBPMs''.')
        else:
            if len(SC['ORD']['BPM']) == 0:
                print('No BPMs registered. Use ''SCregisterBPMs''.')
            else:
                print('%d BPMs registered.' % len(SC['ORD']['BPM']))
            if len(np.unique(SC['ORD']['BPM'])) != len(SC['ORD']['BPM']):
                print('BPMs not uniquely defined.')
        if 'Girder' not in SC['ORD'] and ('Plinth' in SC['ORD'] or 'Section' in SC['ORD']):
            print('Girders must be registered for other support structure misalingments to work.')
        if 'CM' not in SC['ORD']:
            print('No CMs registered. Use ''SCregisterCMs''.')
        else:
            if len(SC['ORD']['CM'][0]) == 0:
                print('No horizontal CMs registered. Use ''SCregisterCMs''.')
            else:
                print('%d HCMs registered.' % len(SC['ORD']['CM'][0]))
            if len(SC['ORD']['CM']) != 2 or len(SC['ORD']['CM'][1]) == 0:
                print('No vertical CMs registered. Use ''SCregisterCMs''.')
            else:
                print('%d VCMs registered.' % len(SC['ORD']['CM'][1]))
            if len(np.unique(SC['ORD']['CM'][0])) != len(SC['ORD']['CM'][0]):
                print('Horizontal CMs not uniquely defined.')
            if len(np.unique(SC['ORD']['CM'][1])) != len(SC['ORD']['CM'][1]):
                print('Vertical CMs not uniquely defined.')
            for ord in SC['ORD']['CM'][0]:
                if SC['RING'][ord]['CMlimit'][0] == 0:
                    print('HCM limit is zero (Magnet ord: %d). Sure about that?' % ord)
            for ord in SC['ORD']['CM'][1]:
                if SC['RING'][ord]['CMlimit'][1] == 0:
                    print('VCM limit is zero (Magnet ord: %d). Sure about that?' % ord)
        if 'Magnet' not in SC['ORD']:
            print('No magnets are registered. Use ''SCregisterMagnets''.')
        else:
            for ord in SC['ORD']['Magnet']:
                if len(SC['RING'][ord]['PolynomB']) != len(SC['RING'][ord]['PolynomA']):
                    raise ValueError('Length of PolynomB and PolynomA are not equal (Magnet ord: %d)' % ord)
                elif len(SC['RING'][ord]['SetPointB']) != len(SC['RING'][ord]['CalErrorB']):
                    print('Length of SetPointB and CalErrorB are not equal (Magnet ord: %d)' % ord)
                elif len(SC['RING'][ord]['SetPointA']) != len(SC['RING'][ord]['CalErrorA']):
                    print('Length of SetPointA and CalErrorA are not equal (Magnet ord: %d)' % ord)
                if 'PolynomBOffset' in SC['RING'][ord]:
                    if len(SC['RING'][ord]['PolynomBOffset']) != len(SC['RING'][ord]['PolynomAOffset']):
                        raise ValueError(
                            'Length of PolynomBOffset and PolynomAOffset are not equal (Magnet ord: %d)' % ord)
                if 'CombinedFunction' in SC['RING'][ord] and SC['RING'][ord]['CombinedFunction'] == 1:
                    if 'BendingAngle' not in SC['RING'][ord]:
                        raise ValueError('Combined function magnet (ord: %d) requires field ''BendingAngle''.' % ord)
                    if SC['RING'][ord]['NomPolynomB'][1] == 0 or SC['RING'][ord]['BendingAngle'] == 0:
                        print(
                            'Combined function magnet (ord: %d) has zero bending angle or design quadrupole component.' % ord)
                if 'Mag' in SC['SIG'] and len(SC['SIG']['Mag'][ord]) != 0:
                    for field in SC['SIG']['Mag'][ord]:
                        if field not in SC['RING'][ord]:
                            print('Field ''%s'' in SC.SIG.Mag doesn''t match lattice element (Magnet ord: %d)' % (
                            field, ord))
                        if field == 'MagnetOffset':
                            if isinstance(SC['SIG']['Mag'][ord][field], list):
                                off = SC['SIG']['Mag'][ord][field][0]
                            else:
                                off = SC['SIG']['Mag'][ord][field]
                            if len(off) != 3:
                                print('SC.SIG.Mag{%d}.MagnetOffset should be a [1x3] (dx,dy,dz) array.' % ord)
                if 'MasterOf' in SC['RING'][ord]:
                    masterFields = SC['RING'][ord].keys()
                    for cOrd in SC['RING'][ord]['MasterOf']:
                        for field in SC['RING'][cOrd]:
                            if field not in masterFields:
                                raise ValueError(
                                    'Child magnet (ord: %d) has different field ''%s'' than master magnet (ord: %d).' % (
                                    cOrd, field, ord))
        if 'Cavity' not in SC['ORD']:
            print('No cavity registered. Use ''SCregisterCAVs''.')
        else:
            if len(SC['ORD']['Cavity']) == 0:
                print('No cavity registered. Use ''SCregisterBPMs''.')
            else:
                print('%d cavity/cavities registered.' % len(SC['ORD']['Cavity']))
            if len(np.unique(SC['ORD']['Cavity'])) != len(SC['ORD']['Cavity']):
                print('Cavities not uniquely defined.')
            if 'RF' in SC['SIG']:
                for ord in SC['ORD']['Cavity']:
                    for field in SC['SIG']['RF'][ord]:
                        if field not in SC['RING'][ord]:
                            print('Field in SC.SIG.RF doesn''t match lattice element (Cavity ord: %d)' % ord)
        if SC['INJ']['beamSize'].shape != (6, 6):
            raise ValueError('6x6 sigma matrix has to be used!')
        apEl = []
        for ord in range(len(SC['RING'])):
            if 'EApertures' in SC['RING'][ord] and 'RApertures' in SC['RING'][ord]:
                print('Lattice element #%d has both EAperture and RAperture' % ord)
            if 'EApertures' in SC['RING'][ord] or 'RApertures' in SC['RING'][ord]:
                apEl.append(ord)
        if len(apEl) == 0:
            print('No apertures found.')
        else:
            print('Apertures defined in %d out of %d elements.' % (len(apEl), len(SC['RING'])))
# End
