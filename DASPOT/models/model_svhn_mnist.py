from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import binom

from torch.autograd import Variable
from torch.nn import init

# Discriminator Model

class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.PReLU, downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.act1 = self.activation()
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad))
        self.act2 = self.activation()
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.act1(h)
        h = self.c1(h)
        h = self.act2(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h,2)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return F.avg_pool2d(x,2)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.PReLU):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, 1, pad)
        self.act1 = self.activation()
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, 1, pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.act1(h)
        h = self.c2(h)
        h = F.avg_pool2d(h,2)
        return h

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x,2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class CoDis32x32(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t):
        super(CoDis32x32, self).__init__()
        self.width = 128

        self.block1_s = OptimizedBlock(ch_s, self.width)
        self.block1_t = OptimizedBlock(ch_t, self.width)
        self.block2 = DisBlock(self.width, self.width, downsample=True)
        self.block3 = DisBlock(self.width, self.width, downsample=True)
        #self.block4 = DisBlock(self.width, self.width, downsample=False)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(self.width, 500, kernel_size=4, stride=1, padding=0))
        self.prelu = nn.PReLU()
        self.conv2_s = nn.utils.spectral_norm(nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0))
        self.conv2_t = nn.utils.spectral_norm(nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0))
        self.conv_cl = nn.utils.spectral_norm(nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.conv1.weight.data)
        init.xavier_uniform_(self.conv2_s.weight.data)
        init.xavier_uniform_(self.conv2_t.weight.data)
        init.xavier_uniform_(self.conv_cl.weight.data)

    def forward(self, x_s, x_t):
        h1_s = self.block1_s(x_s)
        h1_t = self.block1_t(x_t)
        h2_s = self.block2(h1_s)
        h2_t = self.block2(h1_t)
        h3_s = self.block3(h2_s)
        h3_t = self.block3(h2_t)
        h4_s = h3_s#self.block4(h3_s)
        h4_t = h3_t#self.block4(h3_t)
        h5_s = self.prelu(self.conv1(h4_s))
        h5_t = self.prelu(self.conv1(h4_t))
        h6_s = self.conv2_s(h5_s)
        h6_t = self.conv2_t(h5_t)
        return h6_s, h1_s, h5_s, h6_t, h1_t, h5_t

    def pred_s(self, x_s, target=None):
        h1_s = self.block1_s(x_s)
        h2_s = self.block2(h1_s)
        h3_s = self.block3(h2_s)
        h4_s = h3_s#self.block4(h3_s)
        h5_s = self.prelu(self.conv1(h4_s))
        h6_s = self.conv_cl(h5_s)
        return h6_s.squeeze(), h5_s.squeeze()

    def pred_t(self, x_t, target=None):
        h1_t = self.block1_t(x_t)
        h2_t = self.block2(h1_t)
        h3_t = self.block3(h2_t)
        h4_t = h3_t#self.block4(h3_t)
        h5_t = self.prelu(self.conv1(h4_t))
        h6_t = self.conv_cl(h5_t)
        return h6_t.squeeze(), h5_t.squeeze()

    def pred_fromrep(self, h, target=None):
        return self.conv_cl(h).squeeze()


# Generator Model
def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.PReLU, upsample=False):
        super(GenBlock, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.b1 = nn.BatchNorm2d(in_channels, affine=False)
        self.act1 = activation()
        self.b2 = nn.BatchNorm2d(hidden_channels, affine=False)
        self.act2 = activation()
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.act1(h)
        h = _upsample(self.c1(h)) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = _upsample(self.c_sc(x)) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class CoGen32x32(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t, zsize):
        super(CoGen32x32, self).__init__()
        self.width = 128
        self.l1 = nn.Linear(zsize, (4 ** 2) * self.width)
        self.block2 = GenBlock(self.width, self.width, upsample=True)
        self.block3 = GenBlock(self.width, self.width, upsample=True)
        self.block4 = GenBlock(self.width, self.width, upsample=True)
        self.b5 = nn.BatchNorm2d(self.width, affine=False)
        self.c5_s = nn.Conv2d(self.width, ch_s, 3, 1, 1)
        self.c5_t = nn.Conv2d(self.width, ch_t, 3, 1, 1)
        self.activation = nn.PReLU()
        self.final_s = nn.Sigmoid()
        self.final_t = nn.Sigmoid()
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.c5_s.weight.data)
        init.xavier_uniform_(self.c5_t.weight.data)

    def forward(self, z):
        h = z
        h = self.l1(h).view(-1,self.width,4,4)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        out_s = self.final_s(self.c5_s(h))
        out_t = self.final_t(self.c5_t(h))
        return out_s, out_t
