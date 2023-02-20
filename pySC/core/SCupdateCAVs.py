from pySC.constants import RF_PROPERTIES


def SCupdateCAVs(SC, ords=None):
    for ord in (SC.ORD.Cavity if ords is None else ords):
        for field in RF_PROPERTIES:
            setattr(SC.RING[ord], field,
                    getattr(SC.RING[ord], f"{field}SetPoint")
                    * (1 + getattr(SC.RING[ord], f"{field}CalError"))
                    + getattr(SC.RING[ord], f"{field}Offset"))
    return SC
