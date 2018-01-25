# define a conv2d module where only a fraction of the features is learned.
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class Conv2D_partial(nn.Module):
    def split_it(self, A, n_to_fix, n_to_learn, zero_fixed_part=False):
        """Splits a weight matrix to two parts of given sizes and returns each part as a Parameter."""
        A_fixed = None
        A_learn = None
        if n_to_fix == 0:  # all learnable
            A_learn = Parameter(A)
            A_learn.requires_grad = True
            A = [A_learn]

        elif n_to_learn == 0:  # all fixed
            A_fixed = Parameter(A)
            A_fixed.requires_grad = False
            A = [A_fixed]
        else:
            A_fixed = Parameter(A[:n_to_fix])
            A_learn = Parameter(A[n_to_fix:])
            A_learn.requires_grad = True
            A = [A_fixed, A_learn]
            
        if zero_fixed_part and n_to_fix > 0:
            A_fixed.data = 0*A_fixed.data
        if A_fixed is not None:
            A_fixed.requires_grad = False
        return A, A_fixed, A_learn

    def __init__(self, aConv, part=1.0, zero_fixed_part=False):
        super(Conv2D_partial, self).__init__()
        # make a convolution, just to get the weight matrix.

        self.part = part
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = \
            aConv.in_channels, aConv.out_channels, aConv.kernel_size, aConv.stride, aConv.padding, aConv.dilation, aConv.groups, aConv.bias
        # aConv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        W = aConv.weight.data.clone()
        nPlanes = W.shape[0]  # number of convolution filters.

        assert part >=0, 'part cannot be < 0'

        if part < 1:
            n_to_learn = int(nPlanes * part)  # number of filters to actually learn
        else:
            part = int(part)
            assert part <= nPlanes, 'part cannot be larger than number of planes in convolution: {}'.format(nPlanes)
            n_to_learn = part
            #print 'learning',part,'filters for this layer'

        n_to_fix = nPlanes - n_to_learn  # number of filters to keep fixed
        self.n_to_fix = n_to_fix
        self.n_to_learn = n_to_learn

        self.W, self.W_fixed, self.W_learn = self.split_it(W, n_to_fix, n_to_learn,zero_fixed_part)

        self.has_bias = bias is not None
        if self.has_bias:
            b = bias.data.clone()
            self.b, self.b_fixed, self.b_learn = self.split_it(b, n_to_fix, n_to_learn,zero_fixed_part)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.zero_fixed_part=False

    def forward(self, x):

        W = torch.cat(self.W, 0)
        if self.has_bias:
            b = torch.cat(self.b, 0)
        else:
            b = None
        return F.conv2d(x, W, b, self.stride, self.padding, self.dilation, self.groups)