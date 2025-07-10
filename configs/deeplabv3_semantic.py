import ml_collections

from configs import default_config
from configs.datasets import clutter_segmentation_config


def get_config():
    config = default_config.get_default_configs()
    config.dataset = clutter_segmentation_config.get_config()

    model: ml_collections.ConfigDict = config.model
    model.name = "segmentation"
    model.num_classes = config.dataset.num_classes

    training: ml_collections.ConfigDict = config.training
    training.batch_size = 16

    return config
