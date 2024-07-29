import torch
import torch.nn as nn

from mm.functions.mm import compute_mm
from mm.functions.mm_filter import EIG, charpoly
from mm.functions.lu_chipman import lu_chipman, batched_lc
from mm.functions.polarimetry import batched_polarimetry
from mm.utils.roll_win import batched_rolling_window_metric, circstd


class MuellerMatrixModel(nn.Module):
    def __init__(self, feature_keys=[], mask_fun=charpoly, patch_size=4, perc=1, wnum=1, in_channels=None, bA=None, bW=None, *args, **kwargs):
        super(MuellerMatrixModel, self).__init__()

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc
        self.bA = bA
        self.bW = bW
        self.wnum = wnum
        self.feature_chs = [('intensity', 1), ('mueller', 16), ('decompose', 14), ('azimuth', 1), ('std', 1), ('mask', 1)]
        self.ochs = sum([el[-1] for el in self.feature_chs if el[0] in self.feature_keys]) * wnum
        self.ichs = 48 * wnum if in_channels is None else in_channels
        self.rolling_fun = lambda x: circstd(x/180*torch.pi, high=torch.pi, low=0, dim=-1)/torch.pi*180
        self.mask_fun = mask_fun

    def forward(self, x):
        b, f, h, w = x.shape
        # unravel wavelength dimension and pack Mueller features to last dimension
        x = x.view(b, self.wnum, 48, h, w).moveaxis(2, -1)
        # split calibration data
        x, bA, bW = (x[..., :16], self.bA, self.bW) if f == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        # compute Mueller matrix
        m = compute_mm(bA, bW, x, norm=True)
        # compute polarimetry feature maps
        y = torch.zeros((b, 0, h, w), dtype=x.dtype, device=x.device)
        if 'intensity' in self.feature_keys:
            y = torch.cat((y, x.mean(-1)), dim=1)
        if 'mueller' in self.feature_keys:
            y = torch.cat((y, m), dim=1)
        if any(key in self.feature_keys for key in ('azimuth', 'std', 'mask')):
            v = self.mask_fun(m)
            l = lu_chipman(m, mask=v)
            p = batched_polarimetry(l)
            if any(key in self.feature_keys for key in ('azimuth', 'std', 'mask')):
                feat_azi = p[:, 7]
                if 'azimuth' in self.feature_keys:
                    y = torch.cat([y, feat_azi], dim=1)
                if 'std' in self.feature_keys:
                    feat_std = batched_rolling_window_metric(feat_azi, patch_size=self.patch_size, perc=self.perc, function=self.rolling_fun)
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
        self.activation = kwargs.pop('activation', None)
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


def init_mm_model(cfg, train_opt=True):

    MMM = MuellerMatrixPyramid if cfg.levels > 1 or cfg.kernel_size > 0 else MuellerMatrixModel
        
    mm_model = MMM(
        feature_keys=cfg.feature_keys, 
        method=cfg.method,
        levels=cfg.levels, 
        kernel_size=cfg.kernel_size,
        perc=.95,
        activation=cfg.activation,
        wnum=len(cfg.wlens),
        )
    mm_model.to(device=cfg.device)
    mm_model.train() if train_opt and cfg.kernel_size > 0 else mm_model.eval()

    return mm_model
