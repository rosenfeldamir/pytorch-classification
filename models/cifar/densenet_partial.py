import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.split_conv import Conv2D_partial

__all__ = ['densenet_partial']


class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0,part=1.0,zero_fixed_part=False,
                 do_init=False,split_dim=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = Conv2D_partial(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                    part,zero_fixed_part,do_init,split_dim)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2D_partial(nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False),part,zero_fixed_part,do_init,split_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0,part=1.0,zero_fixed_part=False,
                 do_init=False,split_dim=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = Conv2D_partial(nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False),part,zero_fixed_part,do_init,split_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes,part=1.0,zero_fixed_part=False,do_init=False,split_dim=0):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = Conv2D_partial(nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False),part,zero_fixed_part,do_init,split_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_partial(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2,part=1.0,zero_fixed_part=False,
                 do_init=False,split_dim=0):
        super(DenseNet_partial, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        #print 'BAH'
        self.conv1 = Conv2D_partial(nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False),part,zero_fixed_part,do_init,split_dim)
        #print 'BOO'


        self.dense1 = self._make_denseblock(block, n,part,zero_fixed_part,do_init,split_dim)
        self.trans1 = self._make_transition(compressionRate,part,zero_fixed_part,do_init,split_dim)
        self.dense2 = self._make_denseblock(block, n,part,zero_fixed_part,do_init,split_dim)
        self.trans2 = self._make_transition(compressionRate,part,zero_fixed_part,do_init,split_dim)
        self.dense3 = self._make_denseblock(block, n,part,zero_fixed_part,do_init,split_dim)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks,part,zero_fixed_part,do_init,split_dim):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, 
                dropRate=self.dropRate,part=part,zero_fixed_part=zero_fixed_part,do_init=do_init,split_dim=split_dim))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate,part,zero_fixed_part,do_init,split_dim):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes,part,zero_fixed_part,do_init=do_init,split_dim=split_dim)


    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def densenet_partial(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet_partial(**kwargs)