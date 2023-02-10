# Copyright (c) EEEM071, University of Surrey

from .vehicleid import VehicleID
from .veri import VeRi

__imgreid_factory = {
    "veri": VeRi,
    "vehicleID": VehicleID,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError(
            'Invalid dataset, got "{}", but expected to be one of {}'.format(
                name, list(__imgreid_factory.keys())
            )
        )
    return __imgreid_factory[name](**kwargs)
