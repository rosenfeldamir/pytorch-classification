import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['wrn_primed']

# apply controller to a single convolutional

def applyController(x, hint, controller):

    if controller is None or hint is None:
        return x
    if type(controller) is nn.Sequential and len(controller) == 0:  # this is an empty controller, for prettier code...
        return x
    S = x.size()
    h = controller(hint)
    h = h.view(S[0], S[1], 1, 1).contiguous()
    x_result = x * h
    x = x_result + x

    return x

def makeControllerLayer(m, hint_size):
    #print(self.controllerBias)
    print 1
    print m
    print 2
    print  hint_size
    L = [nn.Linear(hint_size, m.num_features, bias=False)]
    return nn.Sequential(*L)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, hint_size = None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_controller = makeControllerLayer(self.bn1, hint_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2_controller = makeControllerLayer(self.bn2, hint_size)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x, hint):
        bn1_res = self.bn1(x)
        bn1_res = applyController(bn1_res, hint, self.bn1_controller)
        if not self.equalInOut:
            x = self.relu1(bn1_res )
        else:
            out = self.relu1(bn1_res )
        bn2_res = self.bn2(self.conv1(out if self.equalInOut else x))
        bn2_res = applyController(bn2_res, hint, self.bn2_controller)
        out = self.relu2(bn2_res)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, hint_size=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, hint_size=hint_size)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, hint_size):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, hint_size))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet_primed(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, hint_size = None):
        super(WideResNet_primed, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, hint_size)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, hint_size)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, hint_size)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.bn_controller = makeControllerLayer(self.bn1, hint_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
                
    def forward(self, x, hint = None):
        out = self.conv1(x, hint)
        out = self.block1(out, hint)
        out = self.block2(out, hint)
        out = self.block3(out, hint)
        bn1_out = self.bn1(out)
        bn1_out = applyController(bn1_out,hint,self.bn_controller)
        out = self.relu(bn1_out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn_primed(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet_primed(**kwargs)
    return model