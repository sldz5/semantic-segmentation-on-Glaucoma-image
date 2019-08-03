#from torch.utils import data

import os,sys
import numpy as np

import scipy.misc as m
from PIL import Image

import torch
from torch.utils.data import Dataset


EYE_PATH='./256fcn/'

# TODO
# o shuffle support

class EYEDataset(Dataset):
    def __init__(self, root_path, set='train', img_size=256):
        self.root_path = root_path
        self.set       = set
        self.img_size  = img_size
        self.n_classes = 3

        assert self.set in ['train', 'test', 'traintest']    #断言set在这三里面的其中一个


        self.files = []
        with open (self.root_path + self.set + '.txt', 'r') as f:   #with 事先需要设置，事后做清理工作一步到位，详见https://www.cnblogs.com/DswCnblog/p/6126588.html
            for line in f:
                self.files.append(line.rstrip())    #rstrip()：删除 string 字符串末尾的指定字符（默认为空格）

        self.files = sorted(self.files)             #排序


    def __len__(self):
        return len(self.files)


    def _get_image(self, path):
        img   = m.imread(path)
        npimg = np.array(img, dtype=np.uint8)

        # RGB => BGR
        npimg = npimg[:, :, ::-1] # make a copy of the same list in reverse order:
        npimg = npimg.astype(np.float64)                #转换类型

        npimg = m.imresize(npimg, (self.img_size, self.img_size))

        npimg = npimg.astype(float) / 255.0

        npimg = npimg.transpose(2,0,1) # （256，256，3）——> (3, 256, 256) 将下标（a[0],a[1],a[2]->a[2],a[0],a[1])

        return torch.from_numpy(npimg).float()


    def _get_pascal_labels(self):
        return np.asarray([[0,0,0], [255,0,0], [0,255,0]])


    def _encode_segmap(self, npgt):
        # npgt contains 255
        npgt = npgt.astype(int)
        

        npgt2 = np.zeros((npgt.shape[0], npgt.shape[1]), dtype=np.int16)    #假设npgt=（256，256）则npgt.shape[0]=256，创建一个和npgt一样大的数组
        for i, label in enumerate(self._get_pascal_labels()):
            npgt2[np.where(np.all(npgt == label, axis=-1))[:2]] = i      ##np.all:测试沿给定轴的所有数组元素是否为真。
                                                                        ##np.where:只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 
        npgt2 = npgt2.astype(int)
        return npgt2


    def _get_gt(self, path):
        gt = m.imread(path)
        npgt = np.array(gt, dtype=np.int32)
        npgt = self._encode_segmap(npgt)
        return torch.from_numpy(npgt).long()


    def __getitem__(self, index):
        base_name = self.files[index]

        img_file = self.root_path + self.set+'A/' + base_name + '.png'
        gt_file  = self.root_path + self.set+'B/' + base_name + '.png'

        img = self._get_image(img_file)
        label= self._get_gt(gt_file)

        return img, label, base_name


    def decode_segmap(self, temp, plot=False):
        label_colours = self._get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r/255.0)
        rgb[:, :, 1] = (g/255.0)
        rgb[:, :, 2] = (b/255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb



### EOF ###
