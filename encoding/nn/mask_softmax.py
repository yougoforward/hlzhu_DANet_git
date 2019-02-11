import torch

from torch.nn.modules.module import Module
import torch.nn.functional as F

class Mask_Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    def __init__(self, mask=None, dim=None):
        super(Mask_Softmax, self).__init__()
        self.dim = dim
        self.mask = mask

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return mvmask_softmax(input, self.mask, self.dim)

class gauss_Mask_Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    def __init__(self, mask=None, dim=None):
        super(Mask_Softmax, self).__init__()
        self.dim = dim
        self.mask = mask

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return gauss_mask_softmax(input, self.mask, self.dim)

def gauss_mask_softmax(input, mask=None, dim=-1):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax function.

        Softmax is defined as:

        :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `(0, 1)` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Arguments:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.


        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).

        """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=5)
    else:
        # zero_mask = torch.zeros(input.size()).to(device=input.device)
        max_input = input.max(dim=dim, keepdim=True)
        mask_exp_input = torch.exp(input - max_input[0]+mask)
        # mask_exp_input = torch.mul(exp_input, mask)
        # mask = mask.to(device=input.device)
        # mask_exp_input = torch.where(mask, exp_input, zero_mask)
        sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim=True) + 1e-10

        # return torch.where(mask, torch.div(mask_exp_input, sum_mask_exp_input), zero_mask)
        return torch.div(mask_exp_input, sum_mask_exp_input)


def mask_softmax(input, mask=None, dim=-1):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax function.

        Softmax is defined as:

        :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `(0, 1)` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Arguments:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.


        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).

        """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=5)
    else:
        # zero_mask = torch.zeros(input.size()).to(device=input.device)
        max_input = input.max(dim=dim, keepdim=True)
        exp_input = torch.exp(input - max_input[0])
        mask_exp_input = torch.mul(exp_input, mask)
        # mask = mask.to(device=input.device)
        # mask_exp_input = torch.where(mask, exp_input, zero_mask)
        sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim=True) + 1e-10

        # return torch.where(mask, torch.div(mask_exp_input, sum_mask_exp_input), zero_mask)
        return torch.div(mask_exp_input, sum_mask_exp_input)


def mvmask_softmax(input, mask=None, dim=-1):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax function.

        Softmax is defined as:

        :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `(0, 1)` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Arguments:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.


        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).

        """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=5)
    else:
        if torch.is_tensor(mask):
            return mask_softmax(input, mask=mask, dim=dim)

        mask=[mask[0].to(device=input.device), mask[1].to(device=input.device)]
        N,H,W = mask[0].size()
        max_input = input.max(dim=dim, keepdim=True)
        exp_input = torch.exp(input - max_input[0])
        # zero_mask = torch.zeros(input.size()).to(device=input.device)
        if N==1:
            mask_exp_input = torch.mul(exp_input, mask[0])
            # mask_exp_input = torch.where(mask[0], exp_input, zero_mask)
            sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim=True) + 1e-10
            # return torch.where(mask[0], torch.div(mask_exp_input, sum_mask_exp_input), zero_mask)
            return torch.div(mask_exp_input, sum_mask_exp_input)
        else:
            Sm = 0
            for i in range(N):
                mask_exp_input = torch.mul(exp_input, mask[0][i])
                # mask_exp_input =torch.where(mask[0][i], exp_input, zero_mask)
                sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim=True) + 1e-10
                # Sm = Sm + torch.where(mask[0][i], torch.div(mask_exp_input, sum_mask_exp_input), zero_mask)
                Sm = Sm + torch.div(mask_exp_input, sum_mask_exp_input)
            return torch.mul(Sm, mask[1])
            # return Sm