import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "clutter"
    dataset.data_dir = "data/mops_data/mops_clutter_dataset_v2.h5"
    dataset.num_classes = 139

    return dataset
