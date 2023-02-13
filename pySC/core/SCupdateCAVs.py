def SCupdateCAVs(SC,ords=None):
    if ords is None:
        ords = SC.ORD.Cavity
    fields = ['Voltage', 'Frequency', 'TimeLag']
    for ord in ords:
        for field in fields:
            SC.RING[ord][field] = SC.RING[ord][field + 'SetPoint'] * (1 + SC.RING[ord][field + 'CalError']) + SC.RING[ord][field + 'Offset']
    return SC
