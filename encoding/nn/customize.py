##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling','SegmentationMultiLosses', 'SegmentationMultiLosses2'
           'nll_SegmentationMultiLosses','nll4_SegmentationMultiLosses',
           'nll5_SegmentationMultiLosses','nll1_SegmentationMultiLosses',
           'nll41_SegmentationMultiLosses','nll44_SegmentationMultiLosses']

class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect



class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3= tuple(preds)


        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3
        return loss

class SegmentationMultiLosses2(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses2, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 = tuple(preds)


        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        loss = loss1 + 0.2*loss2
        return loss

class nll_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3, pred4, pred5 = tuple(preds)


        loss1 = super(nll_SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(nll_SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(nll_SegmentationMultiLosses, self).forward(pred3, target)
        loss4 = super(nll_SegmentationMultiLosses, self).forward(pred4, target)
        loss5 = super(nll_SegmentationMultiLosses, self).forward(pred5, target)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        return loss

class nll4_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll4_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3, pred4 = tuple(preds)


        loss1 = super(nll4_SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(nll4_SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(nll4_SegmentationMultiLosses, self).forward(pred3, target)
        loss4 = super(nll4_SegmentationMultiLosses, self).forward(pred4, target)

        loss = loss1 + loss2 + loss3 + loss4
        return loss

class nll44_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll44_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3, pred4 = tuple(preds)


        loss1 = super(nll44_SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(nll44_SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(nll44_SegmentationMultiLosses, self).forward(pred3, target)
        loss4 = super(nll44_SegmentationMultiLosses, self).forward(pred4, target)

        loss = loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4
        return loss

class nll41_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll41_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3, pred4 = tuple(preds)


        loss1 = super(nll41_SegmentationMultiLosses, self).forward(pred1, target)
        # loss2 = super(nll4_SegmentationMultiLosses, self).forward(pred2, target)
        # loss3 = super(nll4_SegmentationMultiLosses, self).forward(pred3, target)
        # loss4 = super(nll4_SegmentationMultiLosses, self).forward(pred4, target)
        #
        # loss = loss1 + loss2 + loss3 + loss4
        loss =loss1
        return loss

class nll5_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll5_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3, pred4, pred5 = tuple(preds)


        loss1 = super(nll5_SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(nll5_SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(nll5_SegmentationMultiLosses, self).forward(pred3, target)
        loss4 = super(nll5_SegmentationMultiLosses, self).forward(pred4, target)
        loss5 = super(nll5_SegmentationMultiLosses, self).forward(pred5, target)
        loss = loss1 + loss2 + loss3 + loss4 + 0.4*loss5
        return loss

class nll1_SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(nll1_SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        # *preds, target = tuple(inputs)
        # pred1 = preds[0][0]
        # loss = super(SegmentationMultiLosses, self).forward(pred1, target)

        *preds, target = tuple(inputs)
        pred1= tuple(preds)[0]


        loss = super(nll1_SegmentationMultiLosses, self).forward(pred1, target)

        return loss



class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)



