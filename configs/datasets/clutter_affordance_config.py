import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "clutter"
    # dataset.data_dir = "../hyperpg/data/birds/CUB_200_2011"
    dataset.data_dir = "data/mops_data/mops_single_dataset_big_v2.h5"
    dataset.num_classes = 56
    dataset.labels = ["affordance"]  # Specify the task type

    return dataset
