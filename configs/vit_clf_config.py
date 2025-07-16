import ml_collections

from configs import default_config


def get_config():
    config = default_config.get_default_configs()

    model: ml_collections.ConfigDict = config.model
    model.name = "object_clf"
    model.backbone = "vit"
    model.num_classes = config.dataset.num_classes

    training: ml_collections.ConfigDict = config.training
    training.batch_size = 32

    return config
