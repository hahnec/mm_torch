import torch
import torch.nn as nn

from mm.functions.mm import compute_mm
from mm.functions.mm_filter import charpoly


class MuellerMatrixSelector(nn.Module):
    def __init__(
            self, 
            ochs=10, 
            norm_opt=1, 
            wnum=1,
            bA=None, 
            bW=None, 
            mask_fun=charpoly,
            *args, 
            **kwargs
        ):
        super(MuellerMatrixSelector, self).__init__()
        self.bA = bA                # calibration matrix A
        self.bW = bW                # calibration matrix W
        self.norm_opt = norm_opt    # normalization option
        self.wnum = wnum            # wavelength number
        self.ochs = ochs            # output channel number
        self.mask_fun = mask_fun    # realizability

    def forward(self, x):
        b, f, h, w = x.shape
        # unravel wavelength dimension and pack Mueller features to last dimension
        x = x.view(b, self.wnum, 48, h, w).moveaxis(2, -1)
        # split calibration data
        x, bA, bW = (x[..., :16], self.bA, self.bW) if f == 16 else (x[..., :16], x[..., 16:32], x[..., 32:48])
        # compute Mueller matrix
        m = compute_mm(bA, bW, x, norm=self.norm_opt)

        r = m
        if self.ochs in (9, 10):
            # extract matrix skipping first column and first row (3x3 matrix)
            r = m.view(*m.shape[:-1], 4, 4)[..., 1:, 1:].flatten(-2, -1)
            if self.ochs == 10:
                if self.norm_opt:
                    # concatenate normalized images
                    intensity = x[..., 0]
                    intensity_min = intensity.amin(dim=(1, 2, 3), keepdim=True)
                    intensity_max = intensity.amax(dim=(1, 2, 3), keepdim=True)
                    intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
                    r = torch.cat((intensity_norm[..., None], r), dim=-1)
                else:
                    # merge 1,1 entry with 3x3 matrix
                    r = torch.cat((m[..., 0][..., None], r), dim=-1)

        # append realizability mask
        if self.mask_fun is not None: r = torch.cat((self.mask_fun(m)[..., None], r), dim=-1)

        return r.squeeze(1).moveaxis(-1, 1)
