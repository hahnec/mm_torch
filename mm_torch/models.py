from mm.model_luchipman import LuChipmanModel
from mm.model_pyramid import LuChipmanPyramid
from mm.model_mueller import MuellerMatrixSelector

from mm.functions.mm_filter import charpoly


def init_mm_model(cfg, train_opt=True, filter_opt=False, *args, **kwargs):

    if cfg.levels > 1 or cfg.kernel_size > 0:
        MuellerMatrixModel = LuChipmanPyramid
    elif cfg.levels == 0:
        MuellerMatrixModel = MuellerMatrixSelector
    else:
        MuellerMatrixModel = LuChipmanModel

    # default values
    ochs = cfg.ochs if hasattr(cfg, 'ochs') else 10
    norm_opt = cfg.norm_opt if hasattr(cfg, 'norm_opt') else False
    norm_mueller = cfg.norm_mueller if hasattr(cfg, 'norm_mueller') else True
    
    mm_model = MuellerMatrixModel(
        feature_keys=cfg.feature_keys, 
        mask_fun=charpoly if 'mask' in cfg.feature_keys else None,
        method=cfg.method,
        levels=cfg.levels, 
        kernel_size=cfg.kernel_size,
        perc=.99,
        activation=cfg.activation,
        wnum=len(cfg.wlens),
        filter_opt=filter_opt,
        norm_opt=norm_opt,
        norm_mueller=norm_mueller,
        ochs=ochs,
        *args, 
        **kwargs,
        )

    mm_model.to(device=cfg.device)
    mm_model.train() if train_opt and cfg.kernel_size > 0 else mm_model.eval()

    return mm_model
