import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG19(nn.Module):
    def __init__(self, features):
        super(VGG19, self).__init__()
        self.features = features
        self.down = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.aspp = nn.ModuleList(aspp(in_channel=128))

        self.final_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.down(x)
        x = F.upsample_bilinear(x, scale_factor=2)

        aspp_out = []
        for k, v in enumerate(self.aspp):
            if k%2 == 0:
                aspp_out.append(self.aspp[k+1](v(x)))
            else:
                continue
        # Using Aspp concat, and relu inside
        for i in range(4):
            x = x + aspp_out[i] * 0.25
        x = F.relu_(x)

        x = self.final_layer(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512]
}

def aspp(aspp_num=4, aspp_stride=2, in_channel=512, use_bn=True):
    aspp_list = []
    for i in range(aspp_num):
        pad = (i+1) * aspp_stride
        dilate = pad
        conv_aspp = nn.Conv2d(in_channel, in_channel, 3, padding=pad, dilation=dilate)
        aspp_list.append(conv_aspp)
        if use_bn:
            aspp_list.append(nn.BatchNorm2d(in_channel))
    return aspp_list

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG19(make_layers(cfg['E'], True))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
    return model
