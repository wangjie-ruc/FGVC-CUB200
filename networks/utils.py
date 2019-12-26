import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    return conv

def conv2d(ni, no, ks=3, stride=1, padding=None, dilation=1, bias=False):
    if padding is None: padding = ks // 2
    conv = nn.Conv2d(ni, no, ks, stride=stride, padding=padding, dilation=dilation, bias=bias)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv

def deconv(ni, no, ks=2, stride=2):
    return nn.ConvTranspose2d(ni, no, ks, stride)

def relu(leaky=None):
    return nn.LeakyReLU(leaky, inplace=True) if leaky else nn.ReLU(inplace=True)

def bn(ni, affine=True):
    return nn.BatchNorm2d(ni, affine=affine)

def conv3x3(ni, no, stride=1, bias=False):
    return conv2d(ni, no, ks=3, stride=stride, bias=bias)

def conv1x1(ni, no, stride=1, bias=False):
    return conv2d(ni, no, ks=1, stride=stride, bias=bias)

def conv_layer(ni, no, ks, stride=1, padding=None, bias=False, use_norm=True, use_activ=True, leaky=None):
    conv = conv2d(ni, no, ks, stride, padding, bias)
    layers = nn.Sequential([conv])
    if use_norm:   layers.add_module('bn', bn(no))
    if use_activ:  layers.add_module('relu', relu(leaky))
    return layers if len(layers) > 1 else conv

def upsample(ni, no, sc=2, mode='bilinear', align_corners=True):
    up = nn.Upsample(scale_factor=sc, mode=mode, align_corners=align_corners)
    conv = conv3x3(ni, no)
    return nn.Sequential(conv, up)

class SelfAttention(nn.Module):

    def __init__(self, ni, dim=None):
        super().__init__()
        if dim is None: dim = ni // 2
        self.theta = conv1d(ni, dim)
        self.phi   = conv1d(ni, dim)
        self.g = conv1d(ni, ni)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        query, key, value = self.theta(x), self.phi(x), self.g(x)
        p = F.softmax(torch.bmm(query.permute(0, 2, 1).contiguous(), key), dim=1)
        o = self.gamma * torch.bmm(value, p) + x
        return o.view(*size).contiguous()


'''
    Woo et al., 
    "CBAM: Convolutional Block Attention Module", 
    ECCV 2018,
    arXiv:1807.06521
'''
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


'''
    He et al.,
    "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition",
    TPAMI 2015,
    arXiv:1406.4729
'''
class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp
