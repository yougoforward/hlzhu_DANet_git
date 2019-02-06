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
from ..nn import reduce_CAM_Module
from ..nn import SE_ASPP_Module


from ..models import BaseNet

__all__ = ['GLCNet_fast', 'get_glcnet_fast']


class GLCNet_fast(BaseNet):
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
        super(GLCNet_fast, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = GLCNet_fastHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)


class GLCNet_fastHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(GLCNet_fastHead, self).__init__()
        inter_channels = in_channels//4

        self.sc = reduce_CAM_Module(in_channels,inter_channels)
        # self.sa = reduce_PAM_Module(inter_channels, stride=2)
        self.sa = PAM_Module(inter_channels)
        self.seaspp = SE_ASPP_Module(inter_channels, inner_features=256, out_features=256, dilations=(6, 12, 18))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, out_channels, 1))

    def forward(self, x):
        # n,c,h,w = x.size()

        sc_feat = self.sc(x)
        sc_output = self.conv6(sc_feat)

        sa_feat = self.sa(sc_feat)
        sa_output = self.conv7(sa_feat)

        seaspp_feat = self.seaspp(sa_feat)
        seaspp_output = self.conv8(seaspp_feat)

        output = [seaspp_output]
        output.append(sc_output)
        output.append(sa_output)
        return tuple(output)


def get_glcnet_fast(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = GLCNet_fast(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

