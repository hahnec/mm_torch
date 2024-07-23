import torch
import torch.nn as nn

from mm.functions.mm import compute_mm
from mm.functions.mm_filter import EIG, PSD, SC
from mm.functions.lu_chipman import batched_lc
from mm.functions.polarimetry import batched_polarimetry
from mm.utils.roll_win import batched_rolling_window_metric, circstd


class MuellerMatrixModel(nn.Module):
    def __init__(self, feature_keys=[], mask_fun=EIG, patch_size=4, perc=1, bA=None, bW=None, wnum=None, *args, **kwargs):
        super(MuellerMatrixModel, self).__init__(*args, **kwargs)

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc
        self.bA = bA
        self.bW = bW
        self.wnum = wnum
        self.feature_chs = [('intensity', 16), ('mueller', 16), ('decompose', 14), ('azimuth', 1), ('std', 1)]
        self.ochs = sum([el[-1] for el in self.feature_chs if el[0] in self.feature_keys]) * wnum
        self.rolling_fun = lambda x: circstd(x/180*torch.pi, high=torch.pi, low=0, dim=-1)/torch.pi*180
        self.mask_fun = mask_fun

    def forward(self, x):
        bc, fc, hc, wc = x.shape
        x = x.view(bc, self.wnum, 48, hc, wc).permute(0,1,3,4,2) if self.wnum > 1 else x.view(bc, 48, hc, wc).permute(0,2,3,1) # pack mueller matrix to last dimension
        x, bA, bW = (x[..., :16], self.bA, self.bW) if fc == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        m = compute_mm(bA, bW, x, norm=True)
        y = torch.zeros((bc, 0, hc, wc), dtype=x.dtype, device=x.device)
        if 'intensity' in self.feature_keys:
            y = torch.cat((y, x.moveaxis(-1, 1).flatten(1, 2) if self.wnum > 1 else x.moveaxis(-1, 1)), dim=1)
        if 'mueller' in self.feature_keys:
            y = torch.cat((y, m), dim=1)
        if any(key in self.feature_keys for key in ('azimuth', 'std')):
            v = self.mask_fun(m)
            v = v if self.wnum > 1 else v.unsqueeze(1)
            l = batched_lc(m, mask=v)
            p = batched_polarimetry(l)
            if any(key in self.feature_keys for key in ('azimuth', 'std')):
                feat_azi = p[:, 7] if self.wnum > 1 else p[:, 7][:, None]
                if 'azimuth' in self.feature_keys:
                    y = torch.cat([y, feat_azi], dim=1)
                if 'std' in self.feature_keys:
                    feat_std = batched_rolling_window_metric(feat_azi, patch_size=self.patch_size, perc=self.perc, function=self.rolling_fun)
                    y = torch.cat((y, feat_std), dim=1)
            y = torch.cat([y, v], dim=1)

        return y


class MuellerMatrixPyramid(MuellerMatrixModel):
    def __init__(self, *args, **kwargs):

        self.levels = kwargs.pop('levels', 2)
        assert self.levels > 0 and isinstance(self.levels, int), 'Levels must be a greater integer than zero.'
        self.method = kwargs.pop('method', 'pooling')
        self.mode = kwargs.pop('mode', 'bilinear')
        self.kernel_size = kwargs.pop('kernel_size', 0)
        self.activation = kwargs.pop('activation', None)
        self.ichs = kwargs.pop('in_channels', 48)
        super().__init__(*args, **kwargs)
        self.ochs *= self.levels

        # use differentiable window function for back-propagation
        if self.kernel_size > 0: self.rolling_win_fun = torch.std

        # spatial scalers
        if self.method == 'pooling':
            self.downsampler = nn.MaxPool2d(2, stride=None, padding=0, dilation=1)
        elif self.method == 'averaging':
            self.downsampler = nn.AvgPool2d(2, stride=None, padding=0, dilation=1)
        self.upsampler = nn.Upsample(scale_factor=2, mode=self.mode, align_corners=True)
        self.act_fun = None
        if self.activation:
            self.act_fun = nn.LeakyReLU(inplace=True) if self.activation.lower().__contains__('leaky') else nn.ReLU(inplace=True)

        # weight layers
        self.i_layers, self.o_layers = [], []
        if self.kernel_size > 0:
            p = self.kernel_size // 2
            for i in range(self.levels):
                i_conv = nn.Conv2d(kernel_size=self.kernel_size, stride=1, padding=p, in_channels=self.ichs, out_channels=self.ichs)
                o_conv = nn.Conv2d(kernel_size=self.kernel_size, stride=1, padding=p, in_channels=self.ochs, out_channels=self.ochs)
                if self.act_fun: i_conv, o_conv = self.act_fun(i_conv), self.act_fun(o_conv)
                self.i_layers.append(i_conv)
                self.o_layers.append(o_conv)
                if self.activation:
                    nn.init.kaiming_normal_(self.i_layers[i].weight, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(self.i_layers[i].weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(self.i_layers[i].weight)
                    nn.init.xavier_uniform_(self.o_layers[i].weight)
            # use module list to enable device transfer
            self.i_layers = nn.ModuleList(self.i_layers)
            self.o_layers = nn.ModuleList(self.o_layers)

    def pad(self, m, h, w):

        dh = h - m.size()[-2]
        dw = w - m.size()[-1]
        return nn.functional.pad(m, [dw//2, dw-dw//2, dh//2, dh-dh//2])

    def forward(self, x):
        b, _, h, w = x.shape
        y = torch.zeros((b, 0, h, w), device=x.device, dtype=x.dtype)
        for i in range(self.levels):
            if self.kernel_size > 0: x = self.i_layers[i](x)
            m = super().forward(x)
            m = self.upsampler(m) if i > 0 else m
            m = self.pad(m, h, w)
            if self.kernel_size > 0: m = self.o_layers[i](m)
            y = torch.cat([y, m], dim=1)
            x = self.downsampler(x) if i < self.levels-1 else x

        return y
