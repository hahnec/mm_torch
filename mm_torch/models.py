import torch
import torch.nn as nn

from mm.functions.mm import batched_mm
from mm.functions.lu_chipman import batched_lc
from mm.functions.polarimetry import batched_polarimetry
from mm.functions.azimuth import compute_azimuth, batched_rolling_window_metric, circstd


class MuellerMatrixModel(nn.Module):
    def __init__(self, feature_keys=[], patch_size=4, perc=1, bA=None, bW=None, *args, **kwargs):
        super(MuellerMatrixModel, self).__init__(*args, **kwargs)

        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.perc = perc
        self.bA = bA
        self.bW = bW
        self.feature_chs = [('intensity', 16), ('mueller', 16), ('decompose', 14), ('azimuth', 1), ('std', 1)]
        self.ochs = sum([el[-1] for el in self.feature_chs if el[0] in self.feature_keys])

    def forward(self, x):
        bc, fc, hc, wc = x.shape
        x, bA, bW = (x[:, :16], x[:, 16:32], x[:, 32:48]) if fc == 48 else (x[..., :16], self.bA, self.bW)
        y = torch.zeros((bc, 0, hc, wc), dtype=x.dtype, device=x.device)
        m = batched_mm(bA, bW, x, filter=False)
        if 'intensity' in self.feature_keys:
            y = torch.cat((y, x), dim=1)
        if 'mueller' in self.feature_keys:
            y = torch.cat((y, m), dim=1)
        if 'decompose' in self.feature_keys:
            l = batched_lc(m, filter=True)
            p = batched_polarimetry(l)
            y = torch.cat([y, p], dim=1)
        if 'azimuth' in self.feature_keys or 'std' in self.feature_keys:
            feat_azi = compute_azimuth(m, dim=1)
            if 'azimuth' in self.feature_keys:
                y = torch.cat([y, feat_azi], dim=1)
            if 'std' in self.feature_keys:
                feat_std = batched_rolling_window_metric(feat_azi.squeeze(1), patch_size=self.patch_size, perc=self.perc)[:, None]
                y = torch.cat((y, feat_std), dim=1)

        return y


class MuellerMatrixPyramid(MuellerMatrixModel):
    def __init__(self, *args, **kwargs):

        self.levels = kwargs.pop('levels', 2)
        self.method = kwargs.pop('method', 'pooling')
        self.mode = kwargs.pop('mode', 'bilinear')
        self.kernel_size = kwargs.pop('kernel_size', 0)
        self.activation = kwargs.pop('activation', None)
        self.ichs = kwargs.pop('in_channels', 48)
        super().__init__(*args, **kwargs)

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
