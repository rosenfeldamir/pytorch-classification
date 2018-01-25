# Define a Renset model that outputs intermediate layers as well.
from torch import nn
import torch
import torch.nn.functional as F
def pool_me(x,poolingtype='max'):
    if poolingtype=='avg':
        return F.adaptive_avg_pool2d(x,output_size=1)
    else:
        return F.adaptive_max_pool2d(x,output_size=1)
    
def pureBiasLayer(n,pureBias=True):
    L = nn.Linear(n,n)
    if pureBias:
        L.weight.data=torch.eye(n)
        #L.weight.requires_grad=True
    return L
class ResNet_with_middle(nn.Module):
    def __init__(self, rr,pooling_type='max',addLinear=False,pureBias=False):
        super(ResNet_with_middle, self).__init__()        
        self.resnet = rr                
        self.pooling_type = pooling_type
        if addLinear:
            self.L0 = pureBiasLayer(64,pureBias=pureBias)
            self.L1 = pureBiasLayer(64,pureBias=pureBias)
            self.L2 = pureBiasLayer(128,pureBias=pureBias)
            self.L3 = pureBiasLayer(256,pureBias=pureBias)
            self.L4 = pureBiasLayer(512,pureBias=pureBias)
        else:
            self.L0 = nn.Sequential()
            self.L1 = nn.Sequential()
            self.L2 = nn.Sequential()
            self.L3 = nn.Sequential()
            self.L4 = nn.Sequential()
    def forward(self, x):
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        poolingType = self.pooling_type
        x0 = pool_me(x,poolingType)#F.adaptive_avg_pool2d(x,output_size=1)
        #print x0.size()
        x0 =  self.L0(x0.squeeze()).view(-1,64,1,1);
        #print 'booya'
        x = r.layer1(x)
        #print 'bayoo'
        x1 =  pool_me(x,poolingType)
        x1 =  self.L1(x1.squeeze()).view(-1,64,1,1);
        x = r.layer2(x)
        x2 =  pool_me(x,poolingType)     
        x2 =  self.L2(x2.squeeze()).view(-1,128,1,1);
        x = r.layer3(x)
        x3 =  pool_me(x,poolingType)   
        x3 =  self.L3(x3.squeeze()).view(-1,256,1,1);
        x = r.layer4(x)
        x4 =  pool_me(x,poolingType)        
        #print '--->',x4.size()
        x4 =  self.L4(x4.squeeze()).view(-1,512,1,1);
        x = r.avgpool(x)
        x = x.view(x.size(0), -1)
        x = r.fc(x)
        return x,x0.squeeze(),x1.squeeze(),x2.squeeze(),x3.squeeze(),x4.squeeze()

