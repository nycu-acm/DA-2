from openpoints.utils import registry
DADAMODELS = registry.Registry('dadamodels')

def build_dadamodel_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return DADAMODELS.build(cfg, **kwargs)