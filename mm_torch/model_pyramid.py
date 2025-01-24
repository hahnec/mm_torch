import torch
import torch.nn as nn

from mm.model_luchipman import LuChipmanModel
from mm.utils.roll_win import batched_rolling_window_metric


class LuChipmanPyramid(LuChipmanModel):
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
