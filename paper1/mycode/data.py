import torch.utils.data as data
import torchvision.transforms as tranform
from PIL import Image, ImageOps, ImageFilter
#创建数据集
class IRSTD(data.Dataset):
    def __init__(self,datadir) :
        self.dir=datadir#文件目录
        self.txtfile='train.txt'
        
        with open(self.txtfile,'r')as f:
            self.imagesName=[line.strip() for line in f.readlines()]
        
        self.baseSize=256
        self.cropSize=256
        #设置正则化 加快迭代速度
        self.transform=tranform.Compose([
            tranform.ToTensor(),
            tranform.Normalize([.485, .456, .406], [.229, .224, .225])
        ])


    def __getitem__(self, index) :
       imgPath=self.dir+'\images'+self.imagesName[index]+'.png'
       labelPath=self.dir+'\labels'+self.imagesName[index]+'.png'
       #为什么要RGB打开覆盖
       img=Image.open(imgPath).convert('RGB')
       mask=Image.open(labelPath)
       img,mask=self.testTransform(img,mask)
       return self.transform(img),tranform.ToTensor()(mask)
    



    #训练过程对图片和GT进行多种处理方式  测试过程则只需要将原始图像更改size至(256,256)
    #训练过程：随机镜像和随机缩放 随机裁剪 高斯模糊(暂未实现 将test和train使用相同处理resize) 
    def testTransform(self,img,mask):
        img = img.resize((self.baseSize, self.baseSize), Image.BILINEAR)
        mask = mask.resize((self.baseSize, self.baseSize), Image.NEAREST)
        return img,mask


        