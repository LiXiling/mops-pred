import ml_collections

from configs import default_config
from configs.datasets import kitchen_affordance_config


def get_config():
    config = default_config.get_default_configs()
    config.dataset = kitchen_affordance_config.get_config()

    model: ml_collections.ConfigDict = config.model
    model.name = "segmentation"
    model.num_classes = config.dataset.num_classes
    model.task = config.dataset.labels[0]
    model.multilabel = True

    training: ml_collections.ConfigDict = config.training
    training.batch_size = 16
    training.num_epochs = 200

    return config
