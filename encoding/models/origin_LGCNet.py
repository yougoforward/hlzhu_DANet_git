###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample, normalize
from ..nn import PAM_Module
from ..nn import CAM_Module
from ..models import BaseNet
from ..nn import ASPP_Module

__all__ = ['LNLNet', 'get_lnlnet']


class LNLNet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(LNLNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = LNLNetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)
        x[3] = upsample(x[3], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        outputs.append(x[3])
        return tuple(outputs)


class LNLNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(LNLNetHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels),
        #                             nn.ReLU())
        #
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels),
        #                             nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.aspp = ASPP_Module(in_channels, inner_features=256, out_features=512, dilations=(12, 24, 36))


    def forward(self, x):
        feat1 = self.aspp(x)
        aspp_output = self.conv5(feat1)

        # feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # output = [sa_output]

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        output.append(aspp_output)
        return tuple(output)


def get_lnlnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='./pretrain_models', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = LNLNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

