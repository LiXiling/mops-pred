from __future__ import annotations

from torch.utils.data import DataLoader, IterableDataset

from mops_pred.config import DatasetConfig

_DATA_REPOSITORY = {}


def register_dataset(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _DATA_REPOSITORY:
            return cls
        _DATA_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_dataloader(
    dataset_cfg: DatasetConfig,
    batch_size: int = 64,
    shuffle_train: bool = True,
    augment: bool = True,
):
    """Create train and test DataLoaders from a DatasetConfig.

    Args:
        dataset_cfg: Dataset configuration specifying name, paths, and labels.
        batch_size: Batch size for both loaders.
        shuffle_train: Whether to shuffle the training DataLoader.
        augment: Whether to apply data augmentation to the training split.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    cls = _DATA_REPOSITORY[dataset_cfg.name]
    data_dir = dataset_cfg.data_dir
    test_dir = dataset_cfg.test_dir or data_dir

    train_ds = cls(
        data_dir,
        train=True,
        augment=augment,
        labels=dataset_cfg.labels,
    )
    test_ds = cls(test_dir, train=False, labels=dataset_cfg.labels)

    # IterableDataset (e.g. WebDataset) handles shuffling internally.
    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train and not is_iterable,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return train_loader, test_loader
