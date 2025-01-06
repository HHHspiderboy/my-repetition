import torch
import torch.nn as nn
from direction import *

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 4, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)

class ReducedBlock(nn.Module):
    def __init__(self,in_channels, stride,kernel_size,padding) :
        super(ReducedBlock,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels,int(in_channels/2),kernel_size,padding,stride),
            nn.BatchNorm2d(int(in_channels/2)),
            nn.LeakyReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(int(in_channels/2),in_channels,kernel_size,padding,stride),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        result=self.layer1(x)
        result=self.layer2(result)
        return result+x

class RDIAN(nn.Module):
    def __init__(self) :
        super(RDIAN,self).__init__()
        #特征提取过程1->16
        self.LR1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.LR2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        #残差结构 四个有不同kernel size（kernel size=2k-1 k=1,2,3,4）的残差块生成四个Feature maps
        self.residual0=self.pileBlock(16,ReducedBlock(),1,1,0,1)
        self.residual1=self.pileBlock(32,ReducedBlock(),3,1,1,2)
        self.residual2=self.pileBlock(32,ReducedBlock(),5,1,2,2)
        self.residual3=self.pileBlock(32,ReducedBlock(),7,1,3,2)
        #通道注意力和空间注意力
        self.attention=AttentionSum(32,32)
        #连接四个feature maps
        self.contation=nn.Sequential(
            nn.Conv2d(4*32,32,kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.residualSum=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=1,padding=0,stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.d11=Conv_d11()
        self.d12=Conv_d12()
        self.d13=Conv_d13()
        self.d14=Conv_d14()
        self.d15=Conv_d15()
        self.d16=Conv_d16()
        self.d17=Conv_d17()
        self.d18=Conv_d18()
        self.head=FCNHead(32,1)

    def forward(self,input):
        #求不同方向的结果
        out11=self.d11(input)
        out12=self.d12(input)
        out13=self.d13(input)
        out14=self.d14(input)
        out15=self.d15(input)
        out16=self.d16(input)
        out17=self.d17(input)
        out18=self.d18(input)
        #把相反方向的结果相乘求和
        out=out11.mul(out15)+out12.mul(out16)+out13.mul(out17)+out14.mul(out18)
        out=nn.Sigmoid(out)#得到热力图DT 作为attention的weights
        
        out1=self.LR1(input)
        result=out1.mul(out)
        result=self.LR2(out1+result)
        #结果回传到ResNet中
        residual0=self.residual0(result)
        residual1=self.residual1(result)
        residual2=self.residual2(result)
        residual3=self.residual3(result)
        #连接Feature map并加入注意力机制
        result_cancatation=self.contation(torch.cat((residual0,residual1,residual2,residual3),dim=1))
        result_attention=self.attention(result_cancatation)

        result_t=nn.functional.interpolate(result_attention,size=[input.size(2),input.size(3)],mode='bilinear')
        result_tt=self.residualSum(result_t)
        result_altra=nn.ReLU(result_t+result_tt,inplace=True)
        pred=self.head(result_altra)
        return pred

    #构建一个由多个block堆叠的网络层 默认由一个block组成
    def pileBlock(self,in_channels,block,kernel_size,stride,padding,blockNumber=1):
        netLayer=[]
        for i in range(blockNumber):
            netLayer.append(block(in_channels,stride,kernel_size,padding))
        return nn.Sequential(*netLayer)
#注意力机制  maxpool+MLP  avgpool+MLP
class ChannelAttention(nn.Module):
    def __init__(self,in_channels,reduction_ratio=16) :
        super(ChannelAttention,self).__init__()
        #定义MLP
        self.MLP=nn.Sequential(
            #展平
            nn.Lambda(lambda x:x.view(x.size(0),-1)),
            nn.Linear(in_channels,in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio,in_channels)
        )
        
    def forward(self,input):
        avg_output=nn.functional.avg_pool2d(input,(input.size(2),input.size(3)),stride=(input.size(2),input.size(3)))
        avg_output=self.MLP(avg_output)

        max_output=nn.functional.max_pool2d(input,(input.size(2),input.size(3)),stride=(input.size(2),input.size(3)))
        max_output=self.MLP(max_output)

        scale=nn.functional.sigmoid(avg_output+max_output).unsqueeze(2).unsqueeze(3).expand_as(input)
        return input*scale
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.compress=ChannelPool()
        self.spatial=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3,dilation=1,bias=False,groups=1),
            nn.BatchNorm2d(1,eps=1e-5,momentum=0.01,affine=True)
        )
    def forward(self,input):
        output=self.spatial(self.compress(input))
        output=nn.functional.sigmoid(output)
        return input*output

class AttentionSum(nn.Module):
    def __init__(self,channels,reduction_ratio=16):
        super(AttentionSum,self).__init__()
        self.channelAttention=ChannelAttention(channels,reduction_ratio)
        self.spatialAttention=SpatialAttention()
    def forward(self,input):
        out=self.spatialAttention(self.channelAttention(input))
        return out
        

