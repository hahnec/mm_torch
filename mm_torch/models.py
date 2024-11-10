import torch
import torch.nn as nn

from mm.functions.mm import compute_mm
from mm.functions.mm_filter import charpoly
from mm.functions.lu_chipman import lu_chipman
from mm.functions.polarimetry import batched_polarimetry
from mm.utils.roll_win import batched_rolling_window_metric, circstd


class MuellerMatrixModel(nn.Module):
    def __init__(
            self, 
            feature_keys=[], 
            mask_fun=charpoly, 
            norm_opt=0, 
            patch_size=8, 
            perc=1, 
            wnum=1, 
            filter_opt=False, 
            in_channels=None, 
            bA=None, 
            bW=None, 
            *args, 
            **kwargs,
            ):
        super(MuellerMatrixModel, self).__init__()

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc
        self.bA = bA
        self.bW = bW
        self.wnum = wnum
        self.filter_opt = filter_opt
        self.feature_chs = [('intensity', 1), ('mueller', 16), ('decompose', 14), ('azimuth', 1), ('std', 1), ('linr', 1), ('totp', 1), ('mask', 0)]
        self.ochs = sum([el[-1] for el in self.feature_chs if el[0] in self.feature_keys]) * wnum
        self.ichs = 48 * wnum if in_channels is None else in_channels
        self.rolling_fun = lambda x: circstd(x/180*torch.pi, high=torch.pi, low=0, dim=-1)/torch.pi*180
        self.mask_fun = mask_fun
        self.norm_opt = norm_opt

    def forward(self, x):
        b, f, h, w = x.shape
        # unravel wavelength dimension and pack Mueller features to last dimension
        x = x.view(b, self.wnum, f//self.wnum, h, w).moveaxis(2, -1)
        # split calibration data
        x, bA, bW = (x[..., :16], self.bA, self.bW) if f == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        # compute Mueller matrix
        m = compute_mm(bA, bW, x, norm=self.norm_opt)
        # compute polarimetry feature maps
        y = torch.zeros((b, 0, h, w), dtype=x.dtype, device=x.device)
        if 'intensity' in self.feature_keys:
            y = torch.cat((y, x.mean(-1)), dim=1)
        if 'mueller' in self.feature_keys:
            y = torch.cat((y, m), dim=1)
        if any(key in self.feature_keys for key in ('azimuth', 'std', 'totp', 'linr', 'mask')):
            v = self.mask_fun(m) if self.mask_fun is not None else torch.ones_like(m[..., 0], dtype=bool)
            l = lu_chipman(m, mask=v, filter_opt=self.filter_opt)
            p = batched_polarimetry(l)
            if 'linr' in self.feature_keys:
                linr = p[:, 6] / p[:, 6].max() if self.norm_opt else p[:, 6]
                y = torch.cat([y, linr], dim=1)
            if 'totp' in self.feature_keys:
                totp = p[:, -1] / p[:, -1].max() if self.norm_opt else p[:, -1]
                y = torch.cat([y, totp], dim=1)
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


class MuellerMatrixPyramid(MuellerMatrixModel):
    def __init__(self, *args, **kwargs):

        self.levels = kwargs.pop('levels', 2)
        assert self.levels > 0 and isinstance(self.levels, int), 'Levels must be a greater integer than zero.'
        self.method = kwargs.pop('method', 'pooling')
        self.mode = kwargs.pop('mode', 'bicubic')
        self.kernel_size = kwargs.pop('kernel_size', 0)
        self.activation = kwargs.pop('activation', 'leaky')
        self.downsample_factor = kwargs.pop('downsample_factor', 4)
        super().__init__(*args, **kwargs)
        self.ochs *= self.levels

        # use differentiable window function for back-propagation
        if self.kernel_size > 0: self.rolling_win_fun = torch.std

        # spatial scalers
        if self.method == 'pooling':
            self.downsampler = nn.MaxPool2d(self.downsample_factor, stride=None, padding=0, dilation=1)
        elif self.method == 'averaging':
            self.ds = nn.AvgPool2d(self.downsample_factor, stride=None, padding=0)
            self.downsampler = lambda x: self.ds(x) * self.downsample_factor**2
        elif self.method == 'window':
            fun = lambda x: torch.std(x, dim=-1)
            self.downsampler = lambda x: batched_rolling_window_metric(x, patch_size=self.downsample_factor, perc=1, function=fun, step_size=self.downsample_factor)
        self.act_fun = None
        if self.activation:
            self.act_fun = nn.LeakyReLU(inplace=True) if self.activation.lower().__contains__('leaky') else nn.ReLU(inplace=True)

        # weight layers
        self.i_layers, self.o_layers = [], []
        p = self.kernel_size // 2
        for i in range(self.levels):
            if self.kernel_size > 0:
                i_conv = nn.Conv2d(kernel_size=self.kernel_size, stride=1, padding=p, in_channels=self.ichs, out_channels=self.ichs)
                o_conv = nn.Conv2d(kernel_size=self.kernel_size, stride=1, padding=p, in_channels=self.ochs//self.levels, out_channels=self.ochs//self.levels)
                if self.act_fun: i_conv, o_conv = nn.Sequential(i_conv, self.act_fun), nn.Sequential(o_conv, self.act_fun)
                self.i_layers.append(i_conv)
                self.o_layers.append(o_conv)
                if self.activation:
                    nn.init.kaiming_normal_(self.i_layers[i][0].weight, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(self.i_layers[i][0].weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(self.i_layers[i].weight)
                    nn.init.xavier_uniform_(self.o_layers[i].weight)
            else:
                # initialize empty layer functions to make model jit scriptable
                self.i_layers.append(Identity())
                self.o_layers.append(Identity())

        # use module list to enable device transfer
        self.i_layers = nn.ModuleList(self.i_layers)
        self.o_layers = nn.ModuleList(self.o_layers)

    def forward(self, x):
        b, _, h, w = x.shape
        y = torch.zeros((b, 0, h, w), device=x.device, dtype=x.dtype)
        for i in range(self.levels):
            x = self.i_layers[i](x)
            m = super().forward(x)
            m = nn.functional.interpolate(m, size=(h, w), mode=self.mode, align_corners=True)
            m = self.o_layers[i](m)
            y = torch.cat([y, m], dim=1)
            x = self.downsampler(x)

        return y


class Identity(nn.Module): 
    def forward(self, x): return x

class MuellerMatrixSelector(nn.Module):
    def __init__(self, ochs=10, norm_opt=1, wnum=1, bA=None, bW=None, *args, **kwargs):
        super(MuellerMatrixSelector, self).__init__()
        self.bA = bA
        self.bW = bW
        self.norm_opt = norm_opt
        self.wnum = wnum
        self.ochs = ochs

    def forward(self, x):
        b, f, h, w = x.shape
        # unravel wavelength dimension and pack Mueller features to last dimension
        x = x.view(b, self.wnum, 48, h, w).moveaxis(2, -1)
        # split calibration data
        x, bA, bW = (x[..., :16], self.bA, self.bW) if f == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        # compute Mueller matrix
        m = compute_mm(bA, bW, x, norm=self.norm_opt)

        if self.ochs in (9, 10):
            # extract matrix skipping first column and first row (3x3 matrix)
            r = m.view(*m.shape[:-1], 4, 4)[..., 1:, 1:].flatten(-2, -1)
            if self.ochs == 10:
                if self.norm_opt:
                    # concatenate normalized images
                    norm_img = x.sum(-1, keepdim=True) / x.flatten(1, -1).max()
                    r = torch.cat((norm_img, r), dim=-1)
                else:
                    # merge 1,1 entry with 3x3 matrix
                    r = torch.cat((m[..., 0][..., None], r), dim=-1)

        return r.squeeze(1).moveaxis(-1, 1)


def init_mm_model(cfg, train_opt=True, filter_opt=False, *args, **kwargs):

    if cfg.levels > 1 or cfg.kernel_size > 0:
        MMM = MuellerMatrixPyramid
    elif cfg.levels == 0:
        MMM = MuellerMatrixSelector
    else:
        MMM = MuellerMatrixModel

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
        *args, 
        **kwargs,
        )

    mm_model.to(device=cfg.device)
    mm_model.train() if train_opt and cfg.kernel_size > 0 else mm_model.eval()

    return mm_model
