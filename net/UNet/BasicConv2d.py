import torch
import torch.nn as nn
# from op_wrapper.adaptive_dilated_conv2d_wrapper import BasicAdaptiveDilatedConv2D

class BasicConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 pad, 
                 if_Bn=True,
                 if_Bias=True,
                 activation=nn.ReLU(inplace=True)):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.if_Bn = if_Bn
        if self.if_Bn:
            self.Bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    
    def forward(self, x):
        x = self.conv2d(x)
        if self.if_Bn:
            x = self.Bn(x)
        if not(self.activation == None):
            x = self.activation(x)
        return x