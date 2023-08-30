from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
import numpy as np
from einops import rearrange


np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)
class IEFgClassifier(nn.Module):
    def __init__(self, num_class=7,num_head=4):
        super(IEFgClassifier, self).__init__()
        
        resnet = models.resnet18(True)
        
        checkpoint = torch.load('./models/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.num_head = num_head
        self.CrossAttnBlock = CrossAttention(512,self.num_head,False)

        for i in range(num_head):
            setattr(self,"head%d" %i, Attention())
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x, gen_x ):
        
        x = self.features(x)
        gen_x = self.features(gen_x)
        x = self.CrossAttnBlock(x,gen_x)
        attnhead = []
        for i in range(self.num_head):
            attnhead.append(getattr(self,"head%d" %i)(x))
        attnhead = torch.stack(attnhead).permute([1,0,2])

        if attnhead.size(1)>1:
            attnhead = F.log_softmax(attnhead,dim=1)
           
        out = self.fc(attnhead.sum(dim=1))
        out = self.bn(out)
        return out

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out 

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.Q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.KV = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.Q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.KV_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x,gen_x):
        b,c,h,w = x.shape

        kv = self.KV_dwconv(self.KV(x))
        k,v = kv.chunk(2, dim=1)   
        q = self.Q_dwconv(self.Q(gen_x))   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

