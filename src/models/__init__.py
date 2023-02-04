# Copyright (c) EEEM071, University of Surrey

from .resnet import resnet50, resnet50_fc512


__model_factory = {
    # image classification models
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
