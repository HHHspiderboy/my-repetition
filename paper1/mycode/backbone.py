import torch.nn as nn
import torch
#CBAM
#通道注意力 全局Avgpool+conv+relu+conv 全局Maxpool+conv+relu+conv->sigmoid
#ratio是通道压缩比例
class ChannelAttention(nn.Module):
    def __init__(self,in_channels,ratio=16):
        super(ChannelAttention,self).__init__(self)
        self.avgPool=nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.conv2=nn.Conv2d(in_channels//16,in_channels,1,bias=False)
        self.maxPool=nn.AdaptiveMaxPool2d(1)
        self.relu=nn.ReLU()
        
    def forword(self,input):
        avgPoolOut=self.conv2(self.relu(self.conv1(self.avgPool(input))))
        maxPoolOut=self.conv2(self.relu(self.conv1(self.maxPool(input))))

        out=avgPoolOut+maxPoolOut
        return nn.Sigmoid(out)

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 输入通道数为2，输出通道数为1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # shape从[b,c,h,w]变为[b,1,h,w]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # shape从[b,c,h,w]变为[b,1,h,w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后变为[b,2,h,w]
        out = torch.cat([avg_out, max_out], dim=1)
        # 输出shape为[batch_size, 1, height, width]
        out = self.sigmoid(self.conv1(out))
        return out * x


#定义ResNet残差网络3x3 conv+ bn+relu  *2
#在ResNet基础上加上了CBAM通道空间注意力机制
class ResNet(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResNet,self).__init__()
        self.conv_1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride)
        self.bn_1=nn.BatchNorm2d(out_channels)
        self.bn_2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        #当输出维度与输入维度不匹配需要使用1x1来进行调整
        if stride!=1 or in_channels!=out_channels:
            self.residual_trans=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels=out_channels)
            )
        else:
            self.residual_trans=None
        self.channelAttentionCount=ChannelAttention(out_channels)  #通道注意力
        self.spatialAttentionCount=SpatialAttention()
    def forword(self,input):
        self.residual=input#保留输入 
        if self.residual_trans!=None:#进行了1x1 conv的调整情况
            self.residual=self.residual_trans(input)
        net=nn.Sequential(
            self.conv_1,self.bn_1,self.relu,self.conv_2,self.bn_2
        )
        out=net(input)
        #补充空间注意力机制和通道注意力机制进入
        out=self.channelAttentionCount(out)*out
        out=self.spatialAttentionCount(out)*out
        out+=self.residual
        out=nn.ReLU(out)
        return out

class My_MSHNet(nn.Module):
    def __init__(self, inChannels,block=ResNet):
        super().__init__()
        #不同尺度通道
        param_channels=[16,32,64,128,256]
        param_block=[2,2,2,2]
        #网络设计
        #backbone UNet将层分离
        self.pooling=nn.MaxPool2d(2,2)
        
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)#scale_factor 表示为输出为输入的倍数
        self.up4=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.up8=nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.up16=self.up4=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)

        self.conv_entry=nn.Conv2d(inChannels,param_channels[0],1,1)

        #encoder-decoder结构
        self.encoder0=self.pileBlock(param_channels[0],param_channels[0],block)
        self.encoder1=self.pileBlock(param_channels[0],param_channels[1],block,param_block[0])
        self.encoder2=self.pileBlock(param_channels[1],param_channels[2],block,param_block[1])
        self.encoder3=self.pileBlock(param_channels[2],param_channels[3],block,param_block[2])

        self.midLayer=self.pileBlock(param_channels[3],param_channels[4],block,param_block[3])

        self.decoder3=self.pileBlock(param_channels[4]+param_channels[3],block,param_block[3])
        self.decoder2=self.pileBlock(param_channels[3]+param_channels[2],block,param_block[2])
        self.decoder1=self.pileBlock(param_channels[2]+param_channels[1],block,param_block[1])
        self.decoder0=self.pileBlock(param_channels[1]+param_channels[0],block,param_block[0])

        #
        self.output0=nn.Conv2d(param_channels[0],1,1)
        self.output1=nn.Conv2d(param_channels[1],1,1)
        self.output2=nn.Conv2d(param_channels[2],1,1)
        self.output3=nn.Conv2d(param_channels[3],1,1)
        #4->1 kernel_size=3 stride/padding=1
        #将设置的四个decoder的结果进行连接
        self.final=nn.Conv2d(4,1,3,1,1)

    #构建一个由多个block堆叠的网络层 默认由一个block组成
    def pileBlock(self,in_channels,out_channels,block,blockNumber=1):
        netLayer=[block(in_channels,out_channels)]
        for i in range(blockNumber-1):
            netLayer.append(block(out_channels,out_channels))
        return nn.Sequential(*netLayer)
    
    def forward(self,input,flag):
        # encoder2Mid=nn.Sequential(
        #     self.conv_entry(),self.encoder0(),
        #     self.pooling(),self.encoder1(),
        #     self.pooling(),self.encoder2(),
        #     self.pooling(),self.encoder3(),
        #     self.midLayer()
        # )
        encoder0Out=self.encoder0(self.conv_entry(input))
        encoder1Out=self.encoder1(self.pooling(encoder0Out))
        encoder2Out=self.encoder2(self.pooling(encoder1Out))
        encoder3Out=self.encoder3(self.pooling(encoder2Out))
        midOut=self.midLayer(self.pooling(encoder3Out))
        decoder3Out=self.decoder3(torch.cat([encoder3Out,self.up(midOut)],1))
        decoder2Out=self.decoder2(torch.cat([encoder2Out,self.up4(decoder3Out)],1))
        decoder1Out=self.decoder1(torch.cat([encoder1Out,self.up8(decoder2Out)],1))
        decoder0Out=self.decoder0(torch.cat([encoder0Out,self.up16(decoder1Out)],1))

        if flag:
            mask0=self.output0(decoder0Out)
            mask1=self.output1(decoder1Out)
            mask2=self.output2(decoder2Out)
            mask3=self.output3(decoder3Out)
            out=self.final(torch.cat([mask0,self.up(mask1),self.up4(mask2),self.up8(mask3)],dim=1))
            return [mask0,mask1,mask2,mask3],out
        else:
            out=self.output0(decoder0Out)
            return [],out
    

