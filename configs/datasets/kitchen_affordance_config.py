import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "clutter"
    dataset.alias = "kitchen_affordance"  # Alias for the dataset
    dataset.data_dir = "data/mops_data/mops_kitchen_dataset_v3_train.h5"
    dataset.test_dir = (
        "data/mops_data/mops_kitchen_dataset_v3_test_new.h5"  # Test data directory
    )
    dataset.num_classes = 56
    dataset.labels = ["affordance"]  # Specify the task type

    return dataset
