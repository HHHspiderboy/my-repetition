import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image,ImageOps as Image,ImageOps
import random


class IRDSTDataset(data.Dataset):
    def __init__(self, args,mode):
        self.args=args
        if mode=='train':
            self.fileList='train.txt'
        else:
            self.fileList='test.txt'
        self.fileList=args.data_dir+'/'+self.fileList
        self.imgsList=args.data_dir+'/images'
        self.labelsList=args.data_dir+'/labels'
        #从txt获取图片顺序
        self.names=[]
        with open(self.fileList,'r')as f:
            self.names.append(i.strip() for i in f.readlines())
        #参数设置
        self.mode=mode

        self.base_size=args.base_size
        self.crop_size=args.crop_size
        #BN层参数加速
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([-0.1246],[1.0923])
        ])

    def __getitem__(self, index):
        cur_name=self.names[index]
        img_path=self.imgsList+'/'+cur_name+'.bmp'
        label_path=self.labelsList+'/'+cur_name+'.bmp'

        img = Image.open(img_path).convert('L') #灰度模式
        mask = Image.open(label_path).convert('1')#二值模式

        if self.mode=='train':
            img,mask=self._sync_transform(img,mask)
        else:
            img=img.resize((self.base_size,self.base_size), Image.BILINEAR)
            mask=mask.resize((self.base_size,self.base_size),Image.NEAREST)
        img,mask=self.transform(img),transforms.ToTensor()(mask)
        return img,mask

    def __len__(self):
       return len(self.names) 

    #对img与mask进行处理
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img, mask

