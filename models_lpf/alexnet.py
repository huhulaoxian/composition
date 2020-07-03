# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from models_lpf import *
from IPython import embed

__all__ = ['AlexNet', 'alexnet']


# model_urls = {
    # 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


class AlexNet(nn.Module):

    def __init__(self, num_classes=9, filter_size=1, pool_only=False, relu_first=True, use_stn=False):
        super(AlexNet, self).__init__()

        if(pool_only): # only apply LPF to pooling layers, so run conv1 at stride 4 as before
            first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),]
        else:
            if(relu_first): # this is the right order
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    Downsample(filt_size=filter_size, stride=2, channels=64),]
            else: # this is the wrong order, since it's equivalent to downsampling the image first
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    Downsample(filt_size=filter_size, stride=2, channels=64),
                    nn.ReLU(inplace=True),]

        first_ds += [nn.MaxPool2d(kernel_size=3, stride=1), 
            Downsample(filt_size=filter_size, stride=2, channels=64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            Downsample(filt_size=filter_size, stride=2, channels=192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            Downsample(filt_size=filter_size, stride=2, channels=256)]
        self.features = nn.Sequential(*first_ds)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # Spatial transformer localization-network
        self.use_stn = use_stn

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=11, stride=4),
            nn.AvgPool2d(5, stride=5),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.AvgPool2d(4, stride=4),
            nn.ReLU(True)
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((2, 2))
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 20),
            nn.ReLU(True),
            nn.Linear(20, 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = self.avgpool1(xs)
        xs = xs.view(-1, 10 * 2 * 2)
        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 2)
        pad = torch.zeros(x.size(0), 2, 1).cuda()
        theta = torch.cat((theta, pad), 2)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


# replacing MaxPool with BlurPool layers
class AlexNetNMP(nn.Module):

    def __init__(self, num_classes=9, filter_size=1, use_stn=False):
        super(AlexNetNMP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=64, pad_off=-1, hidden=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=192, pad_off=-1, hidden=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=256, pad_off=-1, hidden=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # Spatial transformer localization-network
        self.use_stn = use_stn
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((6, 6))
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 6 * 6, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = self.avgpool1(xs)
        xs = xs.view(-1, 10 * 6 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # embed()
        if self.use_stn:
            x = self.stn(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnetnmp(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetNMP(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model





