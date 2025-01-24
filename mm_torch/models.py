from mm.model_luchipman import LuChipmanModel
from mm.model_pyramid import LuChipmanPyramid
from mm.model_mueller import MuellerMatrixSelector


def init_mm_model(cfg, train_opt=True, filter_opt=False, *args, **kwargs):

    if cfg.levels > 1 or cfg.kernel_size > 0:
        MMM = LuChipmanPyramid
    elif cfg.levels == 0:
        MMM = MuellerMatrixSelector
    else:
        MMM = LuChipmanModel

    # default values
    ochs = cfg.ochs if hasattr(cfg, 'ochs') else 10
    norm_opt = cfg.norm_opt if hasattr(cfg, 'norm_opt') else False
    
    mm_model = MMM(
        feature_keys=cfg.feature_keys, 
        method=cfg.method,
        levels=cfg.levels, 
        kernel_size=cfg.kernel_size,
        perc=.99,
        activation=cfg.activation,
        wnum=len(cfg.wlens),
        filter_opt=filter_opt,
        norm_opt=norm_opt,
        ochs=ochs,
        *args, 
        **kwargs,
        )

    mm_model.to(device=cfg.device)
    mm_model.train() if train_opt and cfg.kernel_size > 0 else mm_model.eval()

    return mm_model
