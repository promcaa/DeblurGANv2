import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import InceptionResNetV2

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class FPN(nn.Module):
    def __init__(self, norm_layer, num_filters=256):
        super().__init__()

        self.backbone = InceptionResNetV2(pretrained=True)
        self.norm_layer = norm_layer
        self.num_filters = num_filters

        self.top_layer = ConvBlock(self.backbone.last_linear.in_features, num_filters,
                                    kernel_size=1, norm_layer=norm_layer)
        self.lateral_layer = ConvBlock(self.backbone.layer3.out_channels, num_filters,
                                        kernel_size=1, norm_layer=norm_layer)
        self.fpn_layer = ConvBlock(num_filters, num_filters,
                                    kernel_size=3, padding=1, norm_layer=norm_layer)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone.layer2(x), self.backbone.layer3(x), \
                         self.backbone.layer4(x), self.backbone.layer5(x)
        p5 = self.top_layer(self.backbone.avgpool(c5))
        p4 = self.lateral_layer(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_layer(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_layer(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        p2 = self.fpn_layer(p2)
        p3 = self.fpn_layer(p3)
        p4 = self.fpn_layer(p4)
        p5 = self.fpn_layer(p5)

        return [p2, p3, p4, p5]

class FPNInception(nn.Module):
    def __init__(self, output_ch=3, num_filters=128):
        super().__init__()

        self.fpn = FPN(nn.BatchNorm2d, num_filters=num_filters)

