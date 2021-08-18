import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.UNet.Encoder import Encoder
from net.UNet.BasicConv2d import BasicConv2d
from net.UNet.Decoder import Decoder


class UNet(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, leaky_relu=False, is_aspp=False, n_stack=1):
        super(UNet, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = Decoder(IF_BN=True, leaky_relu=leaky_relu, is_aspp=is_aspp, n_stack=n_stack)

    def forward(self, x):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2)
        return output
