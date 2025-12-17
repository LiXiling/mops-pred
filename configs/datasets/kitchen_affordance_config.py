import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "clutter"
    dataset.alias = "kitchen_affordance"  # Alias for the dataset
    dataset.data_dir = "data/mops_data/mops_kitchen_dataset_100k_v2.h5"
    dataset.test_dir = (
        "data/mops_data/mops_kitchen_dataset_100k_v2_test.h5"  # Test data directory
    )
    dataset.num_classes = 56
    dataset.labels = ["affordance"]  # Specify the task type

    return dataset
