import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[0:3])
        self.conv2 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[4])
        self.ds_2 = nn.Conv2d(256, 128, 1, 1, 0) # using 128
        nn.init.normal_(self.ds_2.weight, std=0.01)
        nn.init.constant_(self.ds_2.bias, 0)
        self.conv3 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[5])
        self.ds_3 = nn.Conv2d(512, 256, 1, 1, 0)
        nn.init.normal_(self.ds_3.weight, std=0.01)
        nn.init.constant_(self.ds_3.bias, 0)
        self.conv4 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[6])
        self.ds_4 = nn.Conv2d(1024, 512, 1, 1, 0)
        nn.init.normal_(self.ds_4.weight, std=0.01)
        nn.init.constant_(self.ds_4.bias, 0)
        self.conv5 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[7])

    def forward(self, x):
        x = self.conv1(x)
        B2_C2 = self.conv2(x)
        B3_C3 = self.conv3(B2_C2)
        B4_C3 = self.conv4(B3_C3)
        B5_C3 = self.conv5(B4_C3)
        return B5_C3, self.ds_4(B4_C3), self.ds_3(B3_C3), self.ds_2(B2_C2)

