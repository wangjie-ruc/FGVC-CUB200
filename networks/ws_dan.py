import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Attention(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Attention, self).__init__()
        self.conv1 = BasicConv2d(in_channels, 512, kernel_size=1)
        self.conv2 = BasicConv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_attentions=32, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        
#         self.attention = BasicConv2d(512 * block.expansion, num_attentions, kernel_size=1)
        self.attention = Attention(512 * block.expansion, num_attentions)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * num_attentions, num_classes)

        self.register_buffer('center', torch.zeros(num_classes, num_attentions * 512 * block.expansion))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _bilinear_attention_pooling(self, x, a):
        x = x.view(x.size(0), x.size(1), -1)
        a = a.view(a.size(0), a.size(1), -1)
        x = torch.bmm(x, torch.transpose(a, 1, 2)) / (12**2)
        x = x.view(x.size(0), -1)
        x = torch.sqrt(x+1e-12)
        x = F.normalize(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        att = self.attention(x)
        
        x = self._bilinear_attention_pooling(x, att)
        f = x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x*100)
        
        return x, f, att


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def attention_crop(attention_maps):
    batch_size, num_parts, height, width = attention_maps.shape
    thetas = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weights = torch.mean(torch.mean(attention_map, dim=1), dim=1)
        part_weights = torch.sqrt(part_weights)
        part_weights = part_weights / torch.sum(part_weights)
        selected_index = torch.multinomial(part_weights, 1, replacement=False, out=None)[0]

        mask = attention_map[selected_index, :, :]

        threshold = random.uniform(0.4, 0.6)
        itemindex = torch.nonzero(mask >= mask.max() * threshold)
        ymin = itemindex[:, 0].min().item() / height - 0.1
        ymax = itemindex[:, 0].max().item() / height + 0.1
        xmin = itemindex[:, 1].min().item() / width - 0.1
        xmax = itemindex[:, 1].max().item() / width + 0.1
        a = xmax - xmin
        e = ymax - ymin
        # crop weight=height
        pad = abs(a-e)/2.
        if a <= e:
            a = e
            xmin -= pad
        else:
            e = a
            ymin -= pad
        
        c = 2*xmin - 1 + a
        f = 2*ymin - 1 + e
        theta = np.asarray([[a, 0, c], [0, e, f]], dtype=np.float32)
        thetas.append(theta)
    thetas = np.asarray(thetas, np.float32)
    return thetas


def attention_drop(attention_maps):
    batch_size, num_parts, height, width = attention_maps.shape
    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weights = torch.mean(torch.mean(attention_map, dim=1), dim=1)
        part_weights = torch.sqrt(part_weights)
        part_weights = part_weights / torch.sum(part_weights)
        selected_index = torch.multinomial(part_weights, 1, replacement=False, out=None)[0]

        mask = attention_map[selected_index, :, :]
        # soft mask
        threshold = random.uniform(0.2, 0.5)
        mask = (mask < threshold * mask.max())
        masks.append(mask)
#     masks = np.asarray(masks, dtype=np.float32)
    masks = torch.stack(masks)
    masks = masks.type(torch.float32)
    masks = torch.unsqueeze(masks, dim=1)
    return masks

def mask2bbox(attention_maps):
    height = attention_maps.shape[2]
    width = attention_maps.shape[3]
    thetas = []
    for i in range(attention_maps.shape[0]):
        mask = attention_maps[i][0]
        max_activate = mask.max()
        min_activate = 0.1 * max_activate
#         mask = (mask >= min_activate)
        itemindex = torch.nonzero(mask >= min_activate)
        ymin = itemindex[:, 0].min().item() / height - 0.05
        ymax = itemindex[:, 0].max().item() / height + 0.05
        xmin = itemindex[:, 1].min().item() / width - 0.05
        xmax = itemindex[:, 1].max().item() / width + 0.05
        a = xmax - xmin
        e = ymax - ymin
        # crop weight=height
        pad = abs(a-e)/2.
        if a <= e:
            a = e
            xmin -= pad
        else:
            e = a
            ymin -= pad
        c = 2*xmin - 1 + a
        f = 2*ymin - 1 + e
        theta = np.asarray([[a, 0, c], [0, e, f]], dtype=np.float32)
        thetas.append(theta)
    thetas = np.asarray(thetas, np.float32)
    return thetas
