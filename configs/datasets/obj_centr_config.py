import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "object_centric"
    # dataset.data_dir = "../hyperpg/data/birds/CUB_200_2011"
    dataset.data_dir = "data/mops_data/mops_single_dataset_v2.h5"
    dataset.num_classes = 46

    return dataset
