###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from .mask_softmax import Mask_Softmax
torch_ver = torch.__version__[:3]

__all__ = ['mvPAM_Module_unfold','mvPAM_Module_mask','mvPAM_Module_mask_cascade','msPAM_Module','PAM_Module', 'CAM_Module']


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

class mvPAM_Module_mask_cascade(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, inter_rate, mask=None):
        super(mvPAM_Module_mask_cascade, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//inter_rate, kernel_size=1)
        self.value_conv0 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv3 = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.mask0 = Parameter(mask[0][0], requires_grad=False)
        self.mask1 = Parameter(mask[0][1], requires_grad=False)
        # self.mask2 = Parameter(mask[0][2], requires_grad=False)

        self.mask_softmax0 = Mask_Softmax(mask=self.mask0, dim=-1)
        self.mask_softmax1 = Mask_Softmax(mask=self.mask1, dim=-1)
        # self.mask_softmax2 = Mask_Softmax(mask=self.mask2, dim=-1)
        self.mask_softmax2 = Mask_Softmax(mask=None, dim=-1)


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
        out0 = self.value_conv1(out0.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out1 = torch.bmm(out0, attention1.permute(0, 2, 1))
        out1 = self.value_conv2(out1.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        out = torch.bmm(out1, attention2.permute(0, 2, 1))
        # out2 = self.value_conv3(out2.view(m_batchsize, C, height, width)).view(m_batchsize, -1, width * height)
        # out = torch.bmm(out2, attention3.permute(0, 2, 1))

        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
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

