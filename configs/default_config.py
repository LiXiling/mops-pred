import ml_collections

from configs import wandb_config
from configs.datasets import obj_centr_config


def get_default_configs():
    config = ml_collections.ConfigDict()

    config.wandb = wandb_config.get_config()
    config.dataset = obj_centr_config.get_config()
    config.training = training = ml_collections.ConfigDict()

    # training.learnrate = 0.001
    training.batch_size = 64
    training.num_epochs = 40
    # training.val_epochs = val_epochs = 5
    # training.checkpoint_epochs = val_epochs

    # Fill in SubConfig
    config.model = ml_collections.ConfigDict()

    config.seed = 42
    return config


def get_config():
    return get_default_configs()
