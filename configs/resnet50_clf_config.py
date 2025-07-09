import ml_collections

from configs import default_config


def get_config():
    config = default_config.get_default_configs()

    model: ml_collections.ConfigDict = config.model
    model.name = "object_clf"
    model.backbone = "resnet50"
    model.num_classes = config.dataset.num_classes

    training: ml_collections.ConfigDict = config.training

    return config
