import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from datasets import *
from model import *
from loss import *
from metrics import *
import time
import os
import tqdm
from utils import *

class my_args(object):
    def __init__(self):
        parser=ArgumentParser()
        #dataset_dir
        parser.add_argument('--data-dir',type=str,default='./datasets/')
        #训练epoch
        parser.add_argument('--epochs',type=int,default=100)
        #图片输入尺寸和剪切尺寸
        parser.add_argument('--base-size',type=int,default=512)
        parser.add_argument('--crop-size',type=int,default=480)
        #lr
        parser.add_argument('--lr',type=float,default=0.005)
    
        return parser.parse_args()
class Trainer(object):
    def __init__(self,args) :
        self.args=args

        #数据集
        self.trianDataLoader=DataLoader(dataset=IRDSTDataset(args,mode='train'),batch_size=args.batch_size,shuffle=True,drop_last=True)
        self.testDataLoader=DataLoader(dataset=IRDSTDataset(args,mode='test'),batch_size=1)
        #backbone和参数初始化
        self.backbone=RDIAN()
        self.backbone.apply(self.weight_init)
        #Loss
        self.criterion=SoftLoULoss()
        #优化器adagrad
        self.optimizer=torch.optim.Adagrad(self.backbone.parameters(),lr=args.lr,weight_decay=1e-4)
        #分析指标参数
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0
        #存储路径
        folder_name = '%s_RDIAN' % (time.strftime('SBC%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
        self.save_folder = os.path.join('resultmydata/', folder_name)
        self.save_pth = os.path.join(self.save_folder, 'checkpoint')
        if not os.path.exists('resultmydata'):
            os.mkdir('resultmydata')
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not os.path.exists(self.save_pth):
            os.mkdir(self.save_pth)

        print('folder: %s' % self.save_folder)
        print('Args: %s' % args)

    #初始化参数
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)

    def training(self,epoch):
        losses=[]
        self.backbone.train()
        self.iou_metric.reset()
        tbar=tqdm(self.trianDataLoader)
        for _,(data,mask) in enumerate(tbar):
            output=self.backbone(data)
            loss=self.criterion(output,mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.log_name = os.path.join(self.save_folder, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('Epoch:%3d,  train loss:%f'
                                 % (epoch,  np.mean(losses)))
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, np.mean(losses)))
        adjust_learning_rate(self.optimizer, epoch, self.args.epochs, self.args.learning_rate,
                             self.args.warm_up_epochs, 1e-6)
    def test(self,epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        eval_losses = []
        self.backbone.eval()
        tbar=tqdm(self.testDataLoader)
        for _,(data,mask) in enumerate(tbar):
            with torch.no_grad():
                output = self.backbone(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, mask)
            eval_losses.append(loss.item())
            
            self.iou_metric.update(output, mask)
            self.nIoU_metric.update(output, mask)
            pixAcc, IoU = self.iou_metric.get()
            _, nIoU,DR = self.nIoU_metric.get()

            tbar.set_description('  Epoch:%3d, eval loss:%f, pixAcc:%f, IoU:%f, nIoU:%f,'# DR:%f'
                                 %(epoch, np.mean(eval_losses), pixAcc, IoU, nIoU))

        pth_name = 'Epoch-%3d_pixAcc-%4f_IoU-%.4f_nIoU-%.4f.pth' % (epoch, pixAcc, IoU, nIoU)
        if IoU > self.best_iou:
            torch.save(self.net.state_dict(), os.path.join(self.save_pth, pth_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(self.net.state_dict(), os.path.join(self.save_pth, pth_name))
            self.best_nIoU = nIoU
        

if __name__=='__main__':
    myArgs=my_args()
     
    trainer=Trainer(myArgs)
    for epoch in range(1,myArgs.epoch+1):
        trainer.training(epoch)
        trainer.test(epoch)
    print('Best IoU: %.5f, best nIoU: %.5f' % (trainer.best_iou, trainer.best_nIoU))

