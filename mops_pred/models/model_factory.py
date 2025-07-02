from .backbones import backbone_factory

_MODEL_REPOSITORY = {}


def register_model(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _MODEL_REPOSITORY:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODEL_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_model(model_cfg: dict):
    cls = _MODEL_REPOSITORY[model_cfg.pop("name")]
    backbone = backbone_factory.create_backbone(model_cfg.pop("backbone"))

    return cls(backbone, **model_cfg)
