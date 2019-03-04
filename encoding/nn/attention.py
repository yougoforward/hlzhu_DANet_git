###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv1d, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, BatchNorm2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax,Dropout2d, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from .mask_softmax import Mask_Softmax
from .mask_softmax import gauss_Mask_Softmax
torch_ver = torch.__version__[:3]

__all__ = ['nonlocal_sampling_Module','selective_aggregation_ASPP_Module','selective_channel_aggregation_Module','Propagation_Pooling_Module','cascaded_mvPAM_Module_mask','PCAM_Module','pyramid_Reason_Module','PRI_CAM_Module','PAM_Module_gaussmask','pooling_PAM_Module','SE_ASPP_Module','reduce_CAM_Module','reduce_PAM_Module','SE_module','pool_CAM_Module','ASPP_Module','mvPAM_Module_unfold','mvPAM_Module_mask','mvPAM_Module_mask_cascade','msPAM_Module','PAM_Module', 'CAM_Module']


class mvPAM_Module_unfold(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, view=None):
        super(mvPAM_Module_unfold, self).__init__()
        self.chanel_in = in_dim
        self.view=view
        self.pad=int((view-1)/2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.view,  self.view), dilation=1, padding=self.pad, stride=1)
        # self.fold = torch.nn.Fold(output_size=None,kernel_size=(self.view,  self.view), dilation=1, padding=self.pad, stride=1)
        # input = torch.randn(2, 5, 3, 4)
        # output = unfold(input)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, 1, width * height).permute(0, 3, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_key = self.unfold(self.key_conv(x)).view(m_batchsize, -1, self.view * self.view, width * height).permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2)
        energy = torch.matmul(proj_query, proj_key)
        attention = self.softmax(energy)

        # attention = attention.view(m_batchsize,width*height,self.view*self.view).permute(0,2,1)
        # attention = torch.diag_embed(attention,offset=0,dim1=-2,dim2=-1)
        # attention = attention.permute(0,3,1,2).contiguous().view((m_batchsize,width*height*self.view*self.view,width*height))
        # attention = torch.nn.functional.fold(attention, output_size=(height,width),kernel_size=(self.view,  self.view), padding=self.pad).view((m_batchsize,width*height,width*height))

        # attention = torch.nn.functional.fold(torch.diag_embed(attention.view(m_batchsize,width*height,self.view*self.view).permute(0,2,1),offset=0,dim1=-2,dim2=-1).permute(0,3,1,2).contiguous().view((m_batchsize,width*height*self.view*self.view,width*height)), output_size=(height,width),kernel_size=(self.view,  self.view), padding=self.pad).view((m_batchsize,width*height,width*height))

        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width)

        proj_value = self.unfold(self.value_conv(x)).view(m_batchsize, -1, self.view * self.view,
                                                          width * height).permute(0, 3, 2, 1)
        out = torch.matmul(attention, proj_value).view(m_batchsize, width * height, -1).permute(0, 2, 1)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class mvPAM_Module_mask(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(mvPAM_Module_mask, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.mask0 = Parameter(mask[0], requires_grad=False)
        self.mask1 = Parameter(mask[1], requires_grad=False)
        self.mask_softmax = Mask_Softmax(mask=[self.mask0, self.mask1], dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.mask_softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class sig_attention(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(sig_attention, self).__init__()

        self.chanel_in = in_dim

        self.conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.sig = Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        out = torch.mul(self.sig(self.conv(x)),x)

        return out

class chl_attention(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(chl_attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Sequential(Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in//4, kernel_size=1), BatchNorm2d(self.inter_chanel), ReLU(inplace=True))
        # self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.exp_conv = Conv1d(in_channels=in_dim//4, out_channels=in_dim, kernel_size=1)

        # self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, self.chanel_in//4, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        expand_energy = self.exp_conv(energy)

        energy_new = torch.max(expand_energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        return out
class mvPAM_Module_mask_cascade_chl(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(mvPAM_Module_mask_cascade_chl, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)


        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    BatchNorm2d(in_dim),
                                    ReLU(inplace=True))
        self.value_conv2 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                 BatchNorm2d(in_dim),
                                 ReLU(inplace=True))
        # self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        # self.mask0 = Parameter(mask[0], requires_grad=False)
        self.mask1 = Parameter(mask[1], requires_grad=False)
        self.mask2 = Parameter(mask[2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)

        self.chla1 = chl_attention(in_dim)
        self.chla2 = chl_attention(in_dim)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)


        proj_value = self.value_conv0(x).view(m_batchsize, -1, width*height)

        out0 = torch.bmm(proj_value, attention0.permute(0, 2, 1))
        out0 = out0.view(m_batchsize, C, height, width)
        out0 = self.gamma * out0 + x
        out0 = self.value_conv1(out0)

        out1 = torch.bmm(self.chla1(out0).view(m_batchsize, -1, width * height), attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = self.gamma1 * out1 + out0
        out1 = self.value_conv2(out1)

        out2 = torch.bmm(self.chla2(out1).view(m_batchsize, -1, width * height), attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out = self.gamma2 * out2 + out1
        # out = self.value_conv2(out1).view(m_batchsize, -1, width * height)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        # out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        return out

class mvPAM_Module_mask_cascade_sig(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(mvPAM_Module_mask_cascade_sig, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)


        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    BatchNorm2d(in_dim),
                                    ReLU(inplace=True))
        self.value_conv2 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                 BatchNorm2d(in_dim),
                                 ReLU(inplace=True))
        # self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        # self.mask0 = Parameter(mask[0], requires_grad=False)
        self.mask1 = Parameter(mask[1], requires_grad=False)
        self.mask2 = Parameter(mask[2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)

        self.siga1 = sig_attention(in_dim)
        self.siga2 = sig_attention(in_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)


        proj_value = self.value_conv0(x).view(m_batchsize, -1, width*height)

        out0 = torch.bmm(proj_value, attention0.permute(0, 2, 1))
        out0 = out0.view(m_batchsize, C, height, width)
        out0 = self.gamma * out0 + x
        out0 = self.value_conv1(out0)

        out1 = torch.bmm(self.siga1(out0).view(m_batchsize, -1, width * height), attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = self.gamma1 * out1 + out0
        out1 = self.value_conv2(out1)

        out2 = torch.bmm(self.siga2(out1).view(m_batchsize, -1, width * height), attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out = self.gamma2 * out2 + out1
        # out = self.value_conv2(out1).view(m_batchsize, -1, width * height)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        # out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        return out


class mvPAM_Module_mask_cascade(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(mvPAM_Module_mask_cascade, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)


        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    BatchNorm2d(in_dim),
                                    ReLU(inplace=True))
        self.value_conv2 = Sequential(Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                 BatchNorm2d(in_dim),
                                 ReLU(inplace=True))
        # self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        # self.mask0 = Parameter(mask[0], requires_grad=False)
        self.mask1 = Parameter(mask[1], requires_grad=False)
        self.mask2 = Parameter(mask[2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)


        proj_value = self.value_conv0(x).view(m_batchsize, -1, width*height)

        out0 = torch.bmm(proj_value, attention0.permute(0, 2, 1))
        out0 = out0.view(m_batchsize, C, height, width)
        out0 = self.gamma * out0 + x
        out0 = self.value_conv1(out0)

        out1 = torch.bmm(out0.view(m_batchsize, -1, width * height), attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = self.gamma1 * out1 + out0
        out1 = self.value_conv2(out1)

        out2 = torch.bmm(out1.view(m_batchsize, -1, width * height), attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out = self.gamma2 * out2 + out1
        # out = self.value_conv2(out1).view(m_batchsize, -1, width * height)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        # out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        return out

class msPAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate):
        super(msPAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        out = self.gamma * out
        return out


class ASPP_Module(Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=256, dilations=(12, 24, 36)):
        super(ASPP_Module, self).__init__()

        self.conv1 = Sequential(AdaptiveAvgPool2d((1, 1)),
                                   Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                BatchNorm2d(inner_features),ReLU())
        self.conv2 = Sequential(
            Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv3 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv4 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv5 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            BatchNorm2d(inner_features),ReLU())

        self.bottleneck = Sequential(
            Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_features),ReLU(),
            Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class pool_CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(pool_CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        self.avgpool = AvgPool2d(2, 2)

        self.se = Sequential(AdaptiveAvgPool2d((1, 1)),
                             Conv2d(in_dim, in_dim // 8, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             BatchNorm2d(in_dim // 8), ReLU(),
                             Conv2d(in_dim // 8, in_dim, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             Sigmoid()
                             )

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        pool_x = self.avgpool(x)

        proj_query = pool_x.view(m_batchsize, C, -1)
        proj_key = pool_x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma * out + x

        se_x = self.se(x)
        se_out = se_x * x

        out = se_out + self.gamma * out + x
        return out

class SE_module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(SE_module, self).__init__()
        self.chanel_in = in_dim

        self.se = Sequential(AdaptiveAvgPool2d((1, 1)),
                                Conv2d(in_dim, in_dim//8, kernel_size=1, padding=0, dilation=1,
                                       bias=False),
                                BatchNorm2d(in_dim//8), ReLU(),
                                Conv2d(in_dim//8, in_dim, kernel_size=1, padding=0, dilation=1,
                                       bias=False),
                                Sigmoid()
                                )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X 1 X 1
        """
        se_x = self.se(x)
        out = se_x*x
        out = out + x

        return out

class concat_mvPAM_Module_mask_cascade(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(concat_mvPAM_Module_mask_cascade, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        # self.mask0 = Parameter(mask[0][0], requires_grad=False)
        self.mask1 = Parameter(mask[0][1], requires_grad=False)
        self.mask2 = Parameter(mask[0][2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)

        self.bottleneck = Sequential(
            Conv2d(in_dim * 4, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(in_dim), ReLU(),
            Dropout2d(0.1)
        )


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)


        proj_value0 = self.value_conv0(x).view(m_batchsize, -1, width*height)
        proj_value1 = self.value_conv0(x).view(m_batchsize, -1, width * height)
        proj_value2 = self.value_conv0(x).view(m_batchsize, -1, width * height)

        out0 = torch.bmm(proj_value0, attention0.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out0 = self.value_conv1(out0.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out1 = torch.bmm(proj_value1, attention1.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out1 = self.value_conv2(out1.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        out = torch.cat((out0, out1, out2, x), 1)
        out = self.bottleneck(out)

        # out = self.gamma*out + x
        return out


class add_mvPAM_Module_mask_cascade(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(add_mvPAM_Module_mask_cascade, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // inter_rate, kernel_size=1)
        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma0 = Parameter(torch.zeros(1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))

        # self.mask0 = Parameter(mask[0][0], requires_grad=False)
        self.mask1 = Parameter(mask[0][1], requires_grad=False)
        self.mask2 = Parameter(mask[0][2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)

        # self.bottleneck = Sequential(
        #     Conv2d(in_dim * 4, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
        #     BatchNorm2d(in_dim), ReLU(),
        #     Dropout2d(0.1)
        # )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)

        proj_value0 = self.value_conv0(x).view(m_batchsize, -1, width * height)
        proj_value1 = self.value_conv0(x).view(m_batchsize, -1, width * height)
        proj_value2 = self.value_conv0(x).view(m_batchsize, -1, width * height)

        out0 = torch.bmm(proj_value0, attention0.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out0 = self.value_conv1(out0.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out1 = torch.bmm(proj_value1, attention1.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out1 = self.value_conv2(out1.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        out = self.gamma0*out0+self.gamma1*out1+self.gamma2*out2 + x
        return out

class cascaded_mvPAM_Module_mask(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(cascaded_mvPAM_Module_mask, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // inter_rate, kernel_size=1)
        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.attention_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.attention_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.attention_sigmoid1 = Sigmoid()
        self.attention_sigmoid2 = Sigmoid()

        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma0 = Parameter(torch.zeros(1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))

        # self.mask0 = Parameter(mask[0], requires_grad=False)
        self.mask1 = Parameter(mask[1], requires_grad=False)
        self.mask2 = Parameter(mask[2], requires_grad=False)

        # range from large to small 1, 1/2, 1/4
        self.mask_softmax0 = Mask_Softmax(mask=None, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)

        self.bottleneck = Sequential(
            Conv2d(in_dim * 4, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(in_dim), ReLU(inplace=True),
            Dropout2d(0.1)
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)

        attention0 = self.mask_softmax0(energy)
        attention1 = self.mask_softmax1(energy)
        attention2 = self.mask_softmax2(energy)
        # attention3 = self.mask_softmax3(energy)

        proj_value0 = self.value_conv0(x).view(m_batchsize, -1, width * height)
        proj_value1 = self.value_conv0(x)
        proj_value2 = self.value_conv0(x)

        out0 = torch.bmm(proj_value0, attention0.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        cascaded_attention1 = self.attention_conv1(out0)
        cascaded_attention1 = self.attention_sigmoid1(cascaded_attention1)

        # out0 = self.value_conv1(out0.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        proj_value1 = torch.mul(proj_value1, cascaded_attention1).view(m_batchsize, -1, width * height)
        out1 = torch.bmm(proj_value1, attention1.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        cascaded_attention2 = self.attention_conv1(out1)
        cascaded_attention2 = self.attention_sigmoid1(cascaded_attention2)
        # out1 = self.value_conv2(out1.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        proj_value2 = torch.mul(proj_value2, cascaded_attention2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1)).view(m_batchsize, C, height, width)
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        out = torch.cat((out0, out1, out2, x), 1)
        out = self.bottleneck(out)
        # out = self.gamma0*out0+self.gamma1*out1+self.gamma2*out2 + x
        return out


class reduce_CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim,out_dim):
        super(reduce_CAM_Module, self).__init__()
        self.channel_in = in_dim
        self.channel_out = out_dim
        self.key_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
                                   BatchNorm2d(out_dim), ReLU(inplace=True))

        self.res_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
                                   BatchNorm2d(out_dim), ReLU(inplace=True))
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.avgpool = AvgPool2d(2, 2)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        pool_x = self.avgpool(x)

        proj_query = pool_x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = self.key_conv(pool_x).view(m_batchsize, self.channel_out, -1)

        energy = torch.bmm(proj_key,proj_query)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, self.channel_out, height, width)

        out = self.gamma*out + self.res_conv(x)
        return out


class reduce_PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, stride):
        super(reduce_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.stride = stride

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=3,stride=stride,padding=1)
        self.value_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                                     BatchNorm2d(in_dim), ReLU(inplace=True))
        self.gamma = Parameter(torch.zeros(1))

        self.res_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=stride, padding=1),
                                   BatchNorm2d(in_dim), ReLU(inplace=True))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height)
        proj_key = self.key_conv(x).view(m_batchsize, -1, (width//self.stride)*(height//self.stride)).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, (height//self.stride), (width//self.stride))

        out = self.gamma*out + self.res_conv(x)
        return out

class ASPP_Module(Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=256, dilations=(12, 24, 36)):
        super(ASPP_Module, self).__init__()

        self.conv1 = Sequential(AdaptiveAvgPool2d((1, 1)),
                                   Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv2 = Sequential(
            Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv3 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv4 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv5 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))

        self.bottleneck = Sequential(
            Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_features),ReLU(inplace=True),
            Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class SE_ASPP_Module(Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=256, dilations=(12, 24, 36)):
        super(SE_ASPP_Module, self).__init__()

        self.conv1 = Sequential(AdaptiveAvgPool2d((1, 1)),
                                   Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv2 = Sequential(
            Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv3 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv4 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))
        self.conv5 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            BatchNorm2d(inner_features),ReLU(inplace=True))

        self.bottleneck = Sequential(
            Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_features),ReLU(inplace=True),
            Dropout2d(0.1)
        )

        self.se = Sequential(AdaptiveAvgPool2d((1, 1)),
                             Conv2d(inner_features*5, inner_features*5 // 8, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             BatchNorm2d(inner_features*5 // 8), ReLU(inplace=True),
                             Conv2d(inner_features*5 // 8, inner_features, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             Sigmoid()
                             )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        se_attention=self.se(out)
        bottle = self.bottleneck(out)

        bottle = bottle+bottle*se_attention

        return bottle


class PAM_Module_gaussmask(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(PAM_Module_gaussmask, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.mask0 = Parameter(mask, requires_grad=False)

        self.mask_softmax = gauss_Mask_Softmax(mask=self.mask0, dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.mask_softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out







class Cls_gloRe_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, num_cls):
        super(Cls_gloRe_Module, self).__init__()
        self.chanel_in = in_dim
        self.inter_chanel = in_dim//2
        self.chanel_out = num_cls
        self.class_aware_pred= Sequential(Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_out, kernel_size=1), Sigmoid())
        self.chanel_reduce = Sequential(Conv2d(in_channels=self.chanel_in, out_channels=self.inter_chanel, kernel_size=1), BatchNorm2d(self.inter_chanel), ReLU(inplace=True))

        self.node_conv = Conv1d(in_channels=self.chanel_out, out_channels=self.chanel_out, kernel_size=1)
        self.chanel_conv = Conv1d(in_channels=self.inter_chanel, out_channels=self.inter_chanel, kernel_size=1)
        self.relu = ReLU()

        self.chanel_expand = Sequential(Conv2d(in_channels=self.inter_chanel, out_channels=self.chanel_in, kernel_size=1), BatchNorm2d(self.chanel_in), ReLU(inplace=True))

        # self.adj = Parameter(torch.randn(self.chanel_out,self.chanel_out))
        # self.diag = torch.diag(torch.ones(self.chanel_out))


        self.gamma = Parameter(torch.zeros(1))

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        proj_query = self.class_aware_pred(x).view(m_batchsize, self.chanel_out, -1)
        proj_key = self.chanel_reduce(x).view(m_batchsize, self.inter_chanel, -1).permute(0, 2, 1)

        Nodes = torch.bmm(proj_query, proj_key)

        graph_conv = self.chanel_conv(self.relu(self.node_conv(Nodes)).permute(0,2,1))

        out = self.chanel_expand(self.relu(torch.matmul(graph_conv, proj_query).view(m_batchsize, -1, height, width)))
        # out = self.gamma*out + x
        out = self.gamma * out
        return out


class PRI_CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(PRI_CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        self.avgpool = AvgPool2d(2, 2)

        self.se = Sequential(AdaptiveAvgPool2d((1, 1)),
                             Conv2d(in_dim, in_dim // 8, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             BatchNorm2d(in_dim // 8), ReLU(inplace=True),
                             Conv2d(in_dim // 8, in_dim, kernel_size=1, padding=0, dilation=1,
                                    bias=False),
                             Sigmoid()
                             )

        self.ClsgloRe = Cls_gloRe_Module(self.chanel_in, num_cls=128)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """

        #pixel level
        m_batchsize, C, height, width = x.size()

        # pool_x = self.avgpool(x)
        pool_x = x
        proj_query = pool_x.view(m_batchsize, C, -1)
        proj_key = pool_x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)



        # out = self.gamma * out + x

        #region level
        ClsgloRe = self.ClsgloRe(x)

        #image level
        se_x = self.se(x)
        se_out = se_x * x


        out = se_out + self.gamma * out + ClsgloRe + x
        return out



class PCAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=2048):
        super(PCAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv_p = Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.pc_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1), BatchNorm2d(in_dim//4), ReLU(inplace=True))

        self.key_conv = Conv2d(in_channels=in_dim//4, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.res_conv_p = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(in_dim//4), ReLU(inplace=True))
        self.pfusion_conv = Sequential(
            Conv2d(in_channels=in_dim//4, out_channels=in_dim // 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.gamma_p = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.query_conv_c = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.res_conv_c = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.cfusion_conv = Sequential(
            Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.gamma_c = Parameter(torch.zeros(1))
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        # pam part
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv_p(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        pc_feat= self.pc_conv(x)
        proj_key = self.key_conv(pc_feat).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out_p = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out_p = out_p.view(m_batchsize, -1, height, width)

        out_p = self.gamma_p*out_p + self.res_conv_p(x)
        out_p = self.pfusion_conv(out_p)

        #cam part

        # pixel level


        proj_query = self.query_conv_c(x).view(m_batchsize, self.chanel_in//4, -1)
        proj_key = pc_feat.view(m_batchsize, self.chanel_in//4, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = out_p.view(m_batchsize, self.chanel_in//4, -1)

        out_c = torch.bmm(attention, proj_value)
        out_c = out_c.view(m_batchsize, -1, height, width)
        out_c = self.gamma_c * out_c + self.res_conv_c(x)
        out_c = self.cfusion_conv(out_c)

        return out_c, out_p



class pyramid_Reason_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=512):
        super(pyramid_Reason_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv_p1 = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1),
            BatchNorm2d(in_dim//2), ReLU(inplace=True))
        self.query_conv_p2 = Sequential(
            Conv2d(in_channels=in_dim//2, out_channels=in_dim//4, kernel_size=1),
            BatchNorm2d(in_dim//4), ReLU(inplace=True))

        self.py1_prop = Conv1d(in_channels=in_dim//2, out_channels=in_dim//2, kernel_size=1)
        self.py2_prop = Conv1d(in_channels=in_dim//2, out_channels=in_dim//2, kernel_size=1)


        self.fusion_conv = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_dim), ReLU(inplace=True))
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        # pam part
        m_batchsize, C, height, width = x.size()
        py1 = self.query_conv_p1(x)
        py2 = self.query_conv_p2(py1)

        proj_query_py1 = py1.view(m_batchsize, self.chanel_in//2, -1)
        proj_key_py1 = x.view(m_batchsize, self.chanel_in, -1).permute(0, 2, 1)
        energy_py1_prop = self.py1_prop(torch.bmm(proj_query_py1, proj_key_py1)).permute(0,2,1)

        proj_query_py2 = py2.view(m_batchsize, self.chanel_in // 4, -1)
        proj_key_py2 = py1.view(m_batchsize, self.chanel_in//2, -1).permute(0, 2, 1)
        energy_py2_prop = self.py1_prop(torch.bmm(proj_query_py2, proj_key_py2).permute(0,2,1))

        energy = torch.bmm(energy_py1_prop, energy_py2_prop)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = py2.view(m_batchsize, self.chanel_in // 4, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        out = self.fusion_conv(out)

        return out



class ori_Propagation_Pooling_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=512):
        super(Propagation_Pooling_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv_p = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_p = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.pc_conv = Sequential(Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1), BatchNorm2d(in_dim//4), ReLU(inplace=True))

        self.key_conv = Conv2d(in_channels=in_dim//4, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.res_conv_p = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(in_dim//4), ReLU(inplace=True))
        self.pfusion_conv = Sequential(
            Conv2d(in_channels=in_dim//4, out_channels=in_dim // 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.gamma_p = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.query_conv_c = Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=3,stride=2, padding=1)
        self.key_conv_c = Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=3,stride =2, padding=1)

        self.res_conv_c = Sequential(
            Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.cfusion_conv = Sequential(
            Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_dim // 4), ReLU(inplace=True))
        self.gamma_c = Parameter(torch.zeros(1))
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : output feature maps( B X 2C X H/2 X W/2)
            exploit cam and pam in global-noloss-downsampling
        """
        m_batchsize, C, height, width = x.size()

        # cam part
        # expand channels C to 2*C

        proj_c_query = self.query_conv_c(x).view(m_batchsize, 2*C, -1)
        proj_c_key = self.key_conv_c(x).view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_c_query, proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out_c = torch.bmm(attention, x.view(m_batchsize, C, -1))
        # out_c = out_c.view(m_batchsize, 2*C, height, width)


        # pam part
        # reduce res HxW to H/2*W/2

        proj_p_query = self.query_conv_p(x).view(m_batchsize, C, -1).permute(0, 2, 1)

        proj_key = self.key_conv_p(x).view(m_batchsize, C, -1)
        energy = torch.bmm(proj_p_query, proj_key)
        attention = self.softmax(energy)

        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out_p = torch.bmm(out_c, attention.permute(0, 2, 1))


        out_p = out_p.view(m_batchsize, -1, height, width)

        # out_p = self.gamma_p*out_p + self.res_conv_p(x)
        # out_p = self.pfusion_conv(out_p)

        # out_c = self.gamma_c * out_c + self.res_conv_c(x)
        # out_c = self.cfusion_conv(out_c)

        return out_c, out_p

class selective_channel_aggregation_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, out_dim):
        super(selective_channel_aggregation_Module, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out = out_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

        self.query_conv_c = Conv2d(in_channels=in_dim, out_channels=out_dim , kernel_size=1)
        # self.res_conv_c = Sequential(
        #     Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
        #     BatchNorm2d(out_dim), ReLU(inplace=True))



    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : output feature maps( B X 2C X H/2 X W/2)
            exploit cam and pam in global-noloss-downsampling
        """
        m_batchsize, C, height, width = x.size()
        # cam part
        # expand channels C to self.chanel_out=2*C

        proj_c_query = self.query_conv_c(x).view(m_batchsize, self.chanel_out, -1)
        # proj_c_key = self.key_conv_c(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_c_query, proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        out_c = torch.bmm(attention, x.view(m_batchsize, C, -1))

        out_c = self.gamma * out_c
        # out_c = self.gamma * out_c + self.res_conv_c(x).view(m_batchsize,self.chanel_out,-1)

        return out_c


class Propagation_Pooling_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, out_dim, stride):
        super(Propagation_Pooling_Module, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out = out_dim
        self.stride=stride

        self.query_conv_p = Conv2d(in_channels=in_dim, out_channels=in_dim // 8,  kernel_size=1)
        self.key_conv_p = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

        self.query_conv_c = Conv2d(in_channels=in_dim, out_channels=out_dim , kernel_size=1)
        self.res_conv_c = Sequential(
            Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            BatchNorm2d(out_dim), ReLU(inplace=True))



    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : output feature maps( B X 2C X H/2 X W/2)
            exploit cam and pam in global-noloss-downsampling
        """
        m_batchsize, C, height, width = x.size()
        # cam part
        # expand channels C to self.chanel_out=2*C

        proj_c_query = self.query_conv_c(x).view(m_batchsize, self.chanel_out, -1)
        # proj_c_key = self.key_conv_c(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_c_query, proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        out_c = torch.bmm(attention, x.view(m_batchsize, C, -1))

        out_c = self.gamma * out_c + self.res_conv_c(x).view(m_batchsize,self.chanel_out,-1)
        # pam part
        # reduce res HxW to H/2*W/2

        proj_p_query = F.interpolate(self.query_conv_p(x),size=(height//self.stride,width//self.stride),mode='bilinear',align_corners=True).view(m_batchsize, -1, height//self.stride*width//self.stride).permute(0, 2, 1)

        proj_key = self.key_conv_p(x).view(m_batchsize, C, height*width)
        energy = torch.bmm(proj_p_query, proj_key)
        attention = self.softmax(energy)

        out_p = torch.bmm(out_c, attention.permute(0, 2, 1))
        out_p = out_p.view(m_batchsize, -1, height//self.stride, width//self.stride)

        return out_p


class pooling_PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, stride):
        super(pooling_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.stride = stride

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        # self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3, stride=stride, padding=1)
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height)
        proj_key = self.key_conv(x).view(m_batchsize, -1, (width//self.stride)*(height//self.stride)).permute(0, 2, 1)

        # proj_key = F.interpolate(self.key_conv(x),size=(height//self.stride, width//self.stride),mode='bilinear',align_corners=True).view(m_batchsize, -1, (width//self.stride)*(height//self.stride)).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height//self.stride, width//self.stride)
        return out


class nonlocal_sampling_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, scale_factor):
        super(nonlocal_sampling_Module, self).__init__()
        self.chanel_in = in_dim
        self.scale = scale_factor

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height)
        proj_key = F.interpolate(self.key_conv(x),scale_factor=(self.scale,self.scale),mode='bilinear',align_corners=True).view(m_batchsize, -1, int(height*self.scale)*int(width*self.scale)).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, int(height*self.scale), int(width*self.scale))
        return out

class selective_aggregation_ASPP_Module(Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=256, dilations=(12, 24, 36)):
        super(selective_aggregation_ASPP_Module, self).__init__()

        self.conv1 = Sequential(AdaptiveAvgPool2d((1, 1)),
                                   Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                BatchNorm2d(inner_features),ReLU())
        self.conv2 = Sequential(
            Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv3 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv4 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            BatchNorm2d(inner_features),ReLU())
        self.conv5 = Sequential(
            Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            BatchNorm2d(inner_features),ReLU())

        self.bottleneck = Sequential(
            Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_features),ReLU(),
            Dropout2d(0.1)
        )
        self.selective_channel_aggregation = selective_channel_aggregation_Module(inner_features * 5, out_features)

    def forward(self, x):
        m_batchsize, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        selective_channel_aggregation = self.selective_channel_aggregation(out).view(m_batchsize,-1,h,w)
        bottle = self.bottleneck(out)
        bottle = selective_channel_aggregation + bottle
        return bottle