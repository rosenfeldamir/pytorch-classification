# a simple module to implement lateral inhibition.
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn import Parameter
from utils.misc import  str2bool

class LateralInhibition(nn.Module):
    def __init__(self, aConv, lateral_inhibition='none',learnable=False):
        lateral_specs = lateral_inhibition.split('_')
        lateral_inhibition = lateral_specs[0]
        self.lateral_inhibition = lateral_inhibition
        if len(lateral_specs) == 1:
            self.repeat=False
        else:
            self.repeat = str2bool(lateral_specs[1])
        super(LateralInhibition, self).__init__()
        # for now, support just one kind of inhibition.

        out_channels = aConv.weight.shape[0]
        self.out_channels = out_channels
        if lateral_inhibition=='none':
            print 'not doing any inhibition'
            self.my_fun = nn.Sequential()
        elif lateral_inhibition in ['rand','default']:
            # Implement as a grouped convolution.

            if self.repeat == False:

                myConv = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels, bias=False)
                if lateral_inhibition == 'rand': # random init, learn as usual
                    pass
                elif lateral_inhibition == 'default': #initialized with preset filter.
                    #f = torch.zeros(3, 3)
                    #f[1, 1] = 1
                    basic_m = torch.zeros(3, 3)
                    basic_m[1, 1] = 1
                    basic_m -= basic_m.mean()
                    myConv.weight.data = basic_m.expand_as(myConv.weight.data)
                #f = torch.zeros(out_channels, 1, 3, 3)
                #myConv.weight.requires_grad=learnable
                self.my_fun = myConv
            else: # define a single filter which will be repeated.
                if lateral_inhibition == 'rand':
                    basic_m = (torch.rand(3,3)-.5)*0.01
                else:
                    basic_m = torch.zeros( 3, 3)
                    basic_m[1, 1] = 1
                    basic_m -= basic_m.mean()

                self.V = Parameter(basic_m)




        elif lateral_inhibition == 'full': # this is not really lateral inhibition, its really just adding more
            # filters. It effectively makes the filters in each later larger by consecutively using two convolutions.
            self.my_fun = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1,  bias=False)
        #else:
        #    raise Exception('unexpected type',lateral_inhibition,'for lateral inhibition')


        if self.repeat:
            self.V.requires_grad=learnable
        else:
            for p in self.my_fun.parameters():
                p.requires_grad = learnable



    def forward(self, x):
        #print('LATERAL, size of input:',x.shape,'size of weights:',self.my_fun.weight.shape)
        if self.repeat:
            weight= self.V.expand(self.out_channels,1,3,3).contiguous()
            return nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, dilation=1, groups=self.out_channels)
        else:
            return self.my_fun(x)
