from backbone import My_MSHNet
import torch.utils.data as data
from data import *
from PIL import *
import tqdm
from torch.optim import Adagrad
import torch.nn as nn
from loss import *
class MyTrainer:
    def __init__(self):
        device = torch.device('cuda')
        self.device = device


        self.curr_epoch=0
        self.batchSize=100
        self.warm_epoch=5
        self.lr=0.05
        self.model=My_MSHNet(3)
        self.trainSet=IRSTD('./dataset/IRSTD-1k')
        self.valSet=IRSTD('./dataset/IRSTD-1k')
        self.trainLoader=data.DataLoader(self.trainSet,self.batchSize, shuffle=True, drop_last=True)
        self.valLoader=data.DataLoader(self.valSet,1,drop_last=True)
        self.optimizer=Adagrad(filter(lambda p:p.requires_grad,self.model.parameters()),lr=self.lr)#仅有梯度更新的参数被放入优化器中
        self.downSample=nn.MaxPool2d(2,2)
        self.loss=SLSIoULoss()
        #检测指标  precision FA PD..

        
    def train(self,epoch):
        self.model.train()
        processBar=tqdm(self.trainLoader)
        totalLoss=AverageMeter()
        tag=False
        for i,(data,mask) in enumerate(processBar):

            data=data.to(self.device)
            labels=data.to(self.device)

            if epoch>self.warm_epoch:
                tag=True
            masks,predicted=self.model(data,tag)
            loss=0
            loss+=self.loss(predicted,labels,self.warm_epoch,epoch)
            
            for j in range(len(masks)):
                if j>0:
                    labels=self.downSample(labels)
                loss+=self.loss(masks[j],labels,self.warm_epoch,epoch)

            loss/=(len(masks)+1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            totalLoss.update(loss.item(),predicted.size(0))
        



    def test(self,epoch):
        self.model.eval()
        tbar=tqdm(self.valLoader)
        tag=False
        with torch.no_grad():
            for i,(data,mask) in enumerate(tbar):
                data = data.to(self.device)
                mask = mask.to(self.device)
                if epoch>self.warm_epoch:
                    tag=True
                loss =0
                _,predicted=self.model(data,tag)

        #保存最佳的数据模型
        

if __name__=='__main__':
    curr_epoch=0
    trainer=MyTrainer()
    for epoch in range(curr_epoch,trainer.curr_epoch):
        trainer.train(epoch)
        trainer.test(epoch)

