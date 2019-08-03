
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models


class FCN(nn.Module):
    def __init__(self,nb_classes=3):
        super(FCN, self).__init__()

        self.nb_classes = nb_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d( 3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, nb_classes, 1),)

    def init_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class FCN8s(FCN):
    def __init__(self,vgg16):
        super(FCN8s, self).__init__()

        self.vgg16 = vgg16

        self.score_pool4 = nn.Conv2d(512, self.nb_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.nb_classes, 1)


    def forward(self, x):
        self.init_params(self.vgg16)
        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(feat1)
        feat3 = self.conv_block3(feat2)
        feat4 = self.conv_block4(feat3)
        feat5 = self.conv_block5(feat4)

        score = self.classifier(feat5)
        
        score_pool4 = self.score_pool4(feat4)
        score_pool3 = self.score_pool3(feat3)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3

        # v0.2
        #out = F.Upsample(score, size=x.size()[2:], mode='bilinear')
        # v0.1.2
        out = F.upsample_bilinear(score, x.size()[2:])

        return out


class FCN16s(FCN):
    def __init__(self):
        super(FCN16s, self).__init__()

        self.score_pool4 = nn.Conv2d(512, self.nb_classes, 1)


    def forward(self, x):
        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(feat1)
        feat3 = self.conv_block3(feat2)
        feat4 = self.conv_block4(feat3)
        feat5 = self.conv_block5(feat4)

        score = self.classifier(feat5)
        score_pool4 = self.score_pool4(feat4)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4

        # v0.2
        #out = F.Upsample(score, size=x.size()[2:], mode='bilinear')
        # v0.1.2
        out = F.upsample_bilinear(score, x.size()[2:])

        return out  # [1,21,256,256]


class FCN32s(FCN):
    def __init__(self):
        super(FCN32s, self).__init__()

    def forward(self, x):
        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(feat1)
        feat3 = self.conv_block3(feat2)
        feat4 = self.conv_block4(feat3)
        feat5 = self.conv_block5(feat4)

        score = self.classifier(feat5)

        # v0.2
        #out = F.Upsample(score, size=x.size()[2:], mode='bilinear')
        # v0.1.2
        out = F.upsample_bilinear(score, x.size()[2:])
        del score,x
        return out  # [1,3,256,256]


### EOF ###
