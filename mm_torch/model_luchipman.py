import torch
import torch.nn as nn

from mm.functions.mm import compute_mm
from mm.functions.mm_filter import charpoly
from mm.functions.lu_chipman import lu_chipman
from mm.functions.polarimetry import batched_polarimetry
from mm.utils.roll_win import batched_rolling_window_metric, circstd


class LuChipmanModel(nn.Module):
    def __init__(
            self, 
            feature_keys=[], 
            mask_fun=charpoly, 
            norm_opt=0, 
            norm_mueller=True, 
            wnum=1, 
            patch_size=8, 
            perc=1, 
            filter_opt=False, 
            in_channels=None, 
            bA=None, 
            bW=None, 
            *args, 
            **kwargs,
            ):
        super(LuChipmanModel, self).__init__()

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc
        self.bA = bA
        self.bW = bW
        self.wnum = wnum
        self.filter_opt = filter_opt
        self.feature_chs = [('intensity', 1), ('mueller', 16), ('decompose', 14), ('azimuth', 1), ('std', 1), ('linr', 1), ('totp', 1), ('datt', 1), ('mask', 0)]
        self.ochs = sum([el[-1] for el in self.feature_chs if el[0] in self.feature_keys]) * wnum
        self.ichs = 48 * wnum if in_channels is None else in_channels
        self.rolling_fun = lambda x: circstd(x/180*torch.pi, high=torch.pi, low=0, dim=-1)/torch.pi*180
        self.mask_fun = mask_fun
        self.norm_opt = norm_opt
        self.norm_mueller = norm_mueller

    def forward(self, x):
        b, f, h, w = x.shape
        # unravel wavelength dimension and pack Mueller features to last dimension
        x = x.view(b, self.wnum, f//self.wnum, h, w).moveaxis(2, -1)
        # split calibration data
        x, bA, bW = (x[..., :16], self.bA, self.bW) if f == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        # compute Mueller matrix
        m = compute_mm(bA, bW, x, norm=self.norm_mueller)
        # compute polarimetry feature maps
        y = torch.zeros((b, 0, h, w), dtype=x.dtype, device=x.device)
        if 'intensity' in self.feature_keys:
            intensity = x[..., 0]
            intensity_min = intensity.amin(dim=(1, 2, 3), keepdim=True)
            intensity_max = intensity.amax(dim=(1, 2, 3), keepdim=True)
            intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
            y = torch.cat((y, intensity_norm), dim=1)
        if 'mueller' in self.feature_keys:
            y = torch.cat((y, m), dim=1)
        if any(key in self.feature_keys for key in ('azimuth', 'std', 'totp', 'linr', 'datt', 'mask')):
            v = self.mask_fun(m) if self.mask_fun is not None else torch.ones_like(m[..., 0], dtype=bool)
            l = lu_chipman(m, mask=v, filter_opt=self.filter_opt)
            p = batched_polarimetry(l)
            if 'linr' in self.feature_keys:
                linr = p[:, 6] / p[:, 6].max() if self.norm_opt else p[:, 6]
                y = torch.cat([y, linr], dim=1)
            if 'totp' in self.feature_keys:
                totp = p[:, -1] / p[:, -1].max() if self.norm_opt else p[:, -1]
                y = torch.cat([y, totp], dim=1)
            if 'datt' in self.feature_keys:
                diatt = p[:, 0] / p[:, 0].max() if self.norm_opt else p[:, 0]
                y = torch.cat([y, diatt], dim=1)
            if any(key in self.feature_keys for key in ('azimuth', 'std', 'mask')):
                feat_azi = p[:, 7] / 180 if self.norm_opt else p[:, 7]
                if 'azimuth' in self.feature_keys:
                    y = torch.cat([y, feat_azi], dim=1)
                if 'std' in self.feature_keys:
                    feat_std = batched_rolling_window_metric(feat_azi, patch_size=self.patch_size, perc=self.perc, function=self.rolling_fun)
                    if self.norm_opt: feat_std = feat_std/feat_std.max()
                    y = torch.cat((y, feat_std), dim=1)
            if 'mask' in self.feature_keys:
                y = torch.cat([y, v], dim=1)

        return y
