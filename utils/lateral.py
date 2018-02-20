# a simple module to implement lateral inhibition.
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn import Parameter
class LateralInhibition(nn.Module):
    def __init__(self, aConv, lateral_inhibition='none',learnable=False):
        super(LateralInhibition, self).__init__()
        # for now, support just one kind of inhibition.
        self.lateral_inhibition = lateral_inhibition
        if lateral_inhibition=='default':
            out_channels = aConv.weight.shape[0]
            f = torch.zeros(3, 3)
            f[1, 1] = 1
            basic_m = torch.zeros(3, 3)
            basic_m[1, 1] = 1
            basic_m -= basic_m.mean()
            f = torch.zeros(out_channels, 1, 3, 3)

            myConv = nn.Conv2d( out_channels,out_channels,(3,3), padding=1,groups=out_channels,bias=False)
            myConv.weight.data = f
            myConv.weight.requires_grad=False

            self.my_fun = myConv
        elif lateral_inhibition=='none':
            print 'not doing any inhibition'
            self.my_fun = nn.Sequential()
        else:
            raise Exception('unexpected type',lateral_inhibition,'for lateral inhibition')

        for p in self.my_fun.parameters():
            p.requires_grad=learnable

    def forward(self, x):
        #print('LATERAL, size of input:',x.shape,'size of weights:',self.my_fun.weight.shape)
        return self.my_fun(x)
