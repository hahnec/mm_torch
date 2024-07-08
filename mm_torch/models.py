import torch
import torch.nn as nn

from mm.functions.mm import batched_mm
from mm.functions.lu_chipman import batched_lc
from mm.functions.polarimetry import batched_polarimetry
from mm.functions.azimuth import compute_azimuth, batched_rolling_window_metric, circstd


class MuellerMatrixModel(nn.Module):
    def __init__(self, feature_keys=[], patch_size=4, perc=1):
        super(MuellerMatrixModel, self).__init__()

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc

    def forward(self, x):
        bc, fc, hc, wc = x.shape
        x, bA, bW = (x[:, :16], x[:, 16:32], x[:, 32:48]) if x.shape[1] == 48 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        m = batched_mm(bA, bW, x)
        x = torch.zeros((bc, 0, hc, wc), dtype=x.dtype, device=x.device)
        if 'decompose' in self.feature_keys:
            l = batched_lc(m)
            p = batched_polarimetry(l)
            x = torch.cat([x, p], dim=1)
        if 'azimuth' in self.feature_keys or 'std' in self.feature_keys:
            feat_azi = compute_azimuth(m, dim=1)
            x = torch.cat([x, feat_azi], dim=1)
            if 'std' in self.feature_keys:
                feat_std = batched_rolling_window_metric(feat_azi.squeeze(1), patch_size=self.patch_size, function=circstd, perc=self.perc)[:, None]
                x = torch.cat((x, feat_std), dim=1)

        return x
