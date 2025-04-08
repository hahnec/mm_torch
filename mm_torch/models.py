from mm.model_luchipman import LuChipmanModel
from mm.model_pyramid import LuChipmanPyramid
from mm.model_mueller import MuellerMatrixSelector

from mm.functions.mm_filter import charpoly


def init_mm_model(cfg=None, train_opt=True, filter_opt=False, *args, **kwargs):

    # default values
    if cfg is None: cfg = {}
    ochs = cfg.ochs if hasattr(cfg, 'ochs') else 16 # output channels
    norm_opt = cfg.norm_opt if hasattr(cfg, 'norm_opt') else False
    norm_mueller = cfg.norm_mueller if hasattr(cfg, 'norm_mueller') else True
    feature_keys = cfg.feature_keys if hasattr(cfg, 'feature_keys') else []
    wlens = cfg.wlens if hasattr(cfg, 'wlens') else [550]
    levels = cfg.levels if hasattr(cfg, 'levels') else 0
    kernel_size = cfg.kernel_size if hasattr(cfg, 'kernel_size') else 0
    method = cfg.method if hasattr(cfg, 'method') else 'pooling'
    activation = cfg.activation if hasattr(cfg, 'activation') else None
    device = cfg.device if hasattr(cfg, 'device') else 'cpu'

    if levels == 0:
        MuellerMatrixModel = MuellerMatrixSelector
    elif levels == 1:
        MuellerMatrixModel = LuChipmanModel
    elif levels > 1 or kernel_size > 0:
        MuellerMatrixModel = LuChipmanPyramid
    
    mm_model = MuellerMatrixModel(
        feature_keys=feature_keys, 
        mask_fun=charpoly if 'mask' in feature_keys else None,
        method=method,
        levels=levels, 
        kernel_size=kernel_size,
        perc=.99,
        activation=activation,
        wnum=len(wlens),
        filter_opt=filter_opt,
        norm_opt=norm_opt,
        norm_mueller=norm_mueller,
        ochs=ochs,
        *args, 
        **kwargs,
        )

    mm_model.to(device=device)
    mm_model.train() if train_opt and kernel_size > 0 else mm_model.eval()

    return mm_model
