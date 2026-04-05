from torch.utils.data import DataLoader

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
    cfg,
    shuffle_train: bool = True,
    batch_size=None,
    augment: bool = True,
):
    """Create train and test DataLoaders from a config dict.

    Args:
        cfg: Config dict with ``dataset`` and (optionally) ``training`` keys.
            ``cfg["dataset"]["name"]`` must match a registered dataset name.
            ``cfg["dataset"]["data_dir"]`` is the path to the training HDF5 file.
            ``cfg["dataset"]["test_dir"]`` is the test HDF5 path (defaults to ``data_dir``).
            ``cfg["dataset"]["labels"]`` optionally specifies which label types to load.
        shuffle_train: Whether to shuffle the training DataLoader.
        batch_size: Overrides ``cfg["training"]["batch_size"]`` when provided.
        augment: Whether to apply data augmentation to the training split.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    cls = _DATA_REPOSITORY[cfg["dataset"]["name"]]
    data_dir = cfg["dataset"]["data_dir"]
    test_dir = cfg["dataset"].get("test_dir", None)
    if test_dir is None:
        test_dir = data_dir
    batch_size = batch_size if batch_size is not None else cfg["training"]["batch_size"]

    train_loader = DataLoader(
        cls(
            data_dir,
            train=True,
            augment=augment,
            labels=cfg["dataset"].get("labels", None),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=8,
    )
    test_loader = DataLoader(
        cls(test_dir, train=False, labels=cfg["dataset"].get("labels", None)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return train_loader, test_loader
