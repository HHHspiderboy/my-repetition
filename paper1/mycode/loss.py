import torch
import torch.nn as nn

def Dice(predicted,target):
    predicted=torch.sigmoid(predicted)#将预测值进行归一化处理
    smooth=1#平滑项 避免除0
    #计算交集
    inter=predicted*target 
    inter_sum=torch.sum(inter,dim=(1,2,3))#在(channel,height,width)三个维度求交集总值
    predicted_sum=torch.sum(predicted,dim=(1,2,3))
    target_sum=torch.sum(target,dim=(1,2,3))
    loss=(2*inter_sum+smooth)/(predicted_sum+target_sum+smooth)
    return 1-loss.mean()
#location sensitive loss
def LocationLoss(predicted,target):
    #初始化一个loss与predicted保持一致
    loss=torch.tensor(0.0,requires_grad=True).to(predicted)
    #(b,c,h,w) 获取predicted和target的所有坐标 并进行归一化
    x_all=torch.arange(0,predicted.shape[2],1).view(1,1,predicted.shape[2]).to(predicted)/predicted.shape[2]
    y_all=torch.arange(0,1,predicted.shape[3]).view(1,predicted.shape[3],1).to(predicted)/predicted.shape[3]
    #加入平滑项
    extra=1e-8
    for i in range(predicted.shape[0]):
        #计算每一批次的质心
        p_x_center=(x_all*predicted[i]).mean()
        p_y_center=(y_all*predicted[i]).mean()

        t_x_center=(x_all*target[i]).mean()
        t_y_center=(y_all*target[i]).mean()

        #转换成极坐标计算Location Loss
        distancePred=torch.sqrt(torch.pow(p_x_center,2)+torch.pow(p_y_center,2))
        distanceTarget=torch.sqrt(torch.pow(t_x_center,2)+torch.pow(t_y_center,2))

        temp_loss=(1-torch.min(distancePred,distanceTarget)/(torch.max(distancePred,distanceTarget)+extra))+\
        (4/torch.pow(torch.pi,2))*torch.pow((torch.arctan(p_y_center/(p_x_center+extra)))-torch.arctan(t_y_center/(t_x_center+extra)),2)
        loss+=temp_loss/predicted.shape[0]
    return loss





    



#Scaler Sensitive Loss 属于IoU Loss的一个变种
class SLSIoULoss(nn.Module):
    def __init__(self) :
        super(SLSIoULoss,self).__init__()

    def forward(self,predicted,target,warm_epoch,epoch,flag=True):
        predicted=nn.Sigmoid(predicted)
        predicted_sum=torch.sum(predicted,dim=(1,2,3))
        inter=predicted*target
        inter_sum=torch.sum(inter,dim=(1,2,3))
        target_sum=torch.sum(target,dim=(1,2,3))
        #作为scaler之间的方差
        scaler_var=torch.pow((predicted_sum-target_sum)/2,2)

        weight=(torch.min(predicted_sum,target_sum)+scaler_var)/(torch.max(predicted_sum,target_sum)+scaler_var)
        #scaler sensitive loss最终表示
        scalaerLoss=(inter_sum)/(target_sum+predicted_sum-inter_sum)

        #Localtion Sensitive Loss
        locationLoss=LocationLoss(predicted,target)

        #分为两阶段的训练(预热部分loss较大  预热后加入location loss和占比参数进行学习)
        if epoch>warm_epoch:
            t_loss=weight*scalaerLoss
            if flag:
                scalaerLoss=1-t_loss.mean()+locationLoss
            else:
                scalaerLoss=1-t_loss.mean()
        else:
            scalaerLoss=1-scalaerLoss.loss()
        return scalaerLoss
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



