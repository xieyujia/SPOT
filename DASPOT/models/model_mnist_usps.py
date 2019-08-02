from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F


# Discriminator Model
class CoDis28x28(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t):
        super(CoDis28x28, self).__init__()
        self.conv0_s = nn.Conv2d(ch_s, 20, kernel_size=5, stride=1, padding=0)
        self.conv0_t = nn.Conv2d(ch_t, 20, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        # self.bn_s = nn.BatchNorm2d(500, affine=False)
        # self.bn_t = nn.BatchNorm2d(500, affine=False)
        self.prelu2 = nn.PReLU()
        self.conv3_s = nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0)
        self.conv3_t = nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0)
        self.conv_cl = nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x_s, x_t):
        h0_s = self.pool0(self.conv0_s(x_s))
        h0_t = self.pool0(self.conv0_t(x_t))
        h1_s = self.pool1(self.conv1(h0_s))
        h1_t = self.pool1(self.conv1(h0_t))
        h2_s = self.prelu2(self.conv2(h1_s))
        h2_t = self.prelu2(self.conv2(h1_t))
        h3_s = self.conv3_s(h2_s)
        h3_t = self.conv3_t(h2_t)
        return h3_s, h2_s, h3_t, h2_t

    def pred_s(self, x_s):
        h0_s = self.pool0(self.conv0_s(x_s))
        h1_s = self.pool1(self.conv1(h0_s))
        h2_s = self.prelu2(self.conv2(h1_s))
        h3_s = self.conv_cl(h2_s)
        return h3_s.squeeze(), h2_s.squeeze()

    def pred_t(self, x_t):
        h0_t = self.pool0(self.conv0_t(x_t))
        h1_t = self.pool1(self.conv1(h0_t))
        h2_t = self.prelu2(self.conv2(h1_t))
        h3_t = self.conv_cl(h2_t)
        return h3_t.squeeze(), h2_t.squeeze()

    def pred_fromrep(self, h2):
        return self.conv_cl(h2).squeeze()


# Generator Model
class CoGen28x28(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t, zsize):
        super(CoGen28x28, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(zsize, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_s = nn.ConvTranspose2d(128, ch_s, kernel_size=6, stride=1, padding=1)
        self.dconv4_t = nn.ConvTranspose2d(128, ch_t, kernel_size=6, stride=1, padding=1)
        self.sig4_s = nn.Sigmoid()
        self.sig4_t = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_s = self.sig4_s(self.dconv4_s(h3))
        out_t = self.sig4_t(self.dconv4_t(h3))
        return out_s, out_t
