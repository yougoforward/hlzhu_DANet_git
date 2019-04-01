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
from ..nn import PAM_Module, topk_PAM_Module, guided_channel_aggregation
from ..nn import CAM_Module , SE_CAM_Module, selective_aggregation_ASPP_Module, reduce_PAM_Module,SE_CAM_Module2, selective_aggregation_ASPP_Module2
from ..models import BaseNet
from ..nn import ASPP_Module

__all__ = ['GLCNet6', 'get_glcnet6']


class GLCNet6(BaseNet):
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
        super(GLCNet6, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = GLCNet5Head(2048, nclass, norm_layer)

        # self.dsn = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512), nn.ReLU(),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        # )
    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        # x_dsn = self.dsn(c3)
        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)
        x[3] = upsample(x[3], imsize, **self._up_kwargs)
        # x_dsn = upsample(x_dsn, imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        outputs.append(x[3])
        # outputs.append(x_dsn)

        return tuple(outputs)


class GLCNet6Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(GLCNet6Head, self).__init__()
        inter_channels = in_channels // 4


        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv5as = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.sa = topk_PAM_Module(inter_channels, 256, inter_channels, 10)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())



        self.conv5s = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.aspp = selective_aggregation_ASPP_Module2(inter_channels, inner_features=256, out_features=256,
                                                      dilations=(12, 24, 36))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())


        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.sc = SE_CAM_Module2(inter_channels)
        self.conv53 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))


        self.gca = guided_channel_aggregation(512*3, 512)

    def forward(self, x):
        #ssa
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        aspp_output = self.conv5(sa_conv)
        #aaspp
        # feat2 = self.conv5s(x)
        # feat_fuse = sa_conv + feat2

        feat_as = self.conv5as(x)
        aspp_feat,bottle,bottle_se,ca_feat = self.aspp(feat_as)
        aspp_conv = self.conv52(aspp_feat)

        sa_output = self.conv6(aspp_conv)

        #sec
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv53(sc_feat)

        sc_output = self.conv7(sc_conv)

        #fuse
        # feat_sum = aspp_conv + sc_conv + sa_conv
        # sasc_output = self.conv8(feat_sum)

        feat_cat = torch.cat([aspp_conv , sc_conv , sa_conv],1)
        # feat_cat = self.bottleneck(feat_cat)
        feat_cat = self.gca(feat_cat)
        sasc_output = self.conv8(feat_cat)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        output.append(aspp_output)

        return tuple(output)


def get_glcnet6(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
        'cocostuff': 'cocostuff',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = GLCNet6(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model


