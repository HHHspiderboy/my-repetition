import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()
        self.unloader = transforms.ToPILImage()

    def IOU(self, pred, mask):
        smooth = 1

        intersection = pred * mask

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(mask, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
            (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        return loss
    def BBOX(self, img, gt):
        listx_pred = []
        listy_pred = []
        listx_gt = []
        listy_gt = []       
        
        pred = img.cpu().clone()
        pred = pred.squeeze(0)
        gt = gt.cpu().clone()
        gt = gt.squeeze(0)

        xy_pred = np.where(pred > 0.5)
        xy_gt = np.where(gt == 1)

        listx_pred = list(xy_pred)[2]
        listy_pred = list(xy_pred)[3]
        listx_gt = list(xy_gt)[2]
        listy_gt = list(xy_gt)[3]

        x_gt_min = min(listx_gt)
        y_gt_min = min(listy_gt)
        x_pred_min = min(listx_pred)
        y_pred_min = min(listy_pred)

        x_gt_max = max(listx_gt)
        y_gt_max = max(listy_gt)
        x_pred_max = max(listx_pred)
        y_pred_max = max(listy_pred)

        x_gt = (x_gt_max - x_gt_min)/2
        y_gt = (y_gt_max - y_gt_min)/2
        x_pred = (x_pred_max - x_pred_min)/2
        y_pred = (y_pred_max - y_pred_min)/2

        x = abs(x_pred - x_gt)
        y = abs(y_pred - y_gt)
        loss = x + y

        return loss


    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)

        img = pred.cpu().clone()
        img = img.squeeze(0)

        loss_iou = self.IOU(pred, mask)

        loss = loss_iou

        return loss