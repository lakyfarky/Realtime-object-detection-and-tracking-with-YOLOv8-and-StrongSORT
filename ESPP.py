import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ESPP(nn.Module):
    def __init__(self, c1, c2=512, a=4):
        super().__init__()
        c_ = c1 // a

        self.conv1x1 = nn.Conv2d(c1, c_, kernel_size=1)
        self.conv3x3_initial = nn.Conv2d(c1, c_, kernel_size=3, padding=1)
        self.conv3x3 = nn.Conv2d(c_, c_, kernel_size=3, padding=1)
        self.conv1x1_shortcut = nn.Conv2d(c_, c1, kernel_size=1)

        self.conv_atrous_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, dilation=1)
        self.conv_atrous_3 = nn.Conv2d(c_, c_, kernel_size=3, padding=3, dilation=3)
        self.conv_atrous_5 = nn.Conv2d(c_, c_, kernel_size=3, padding=5, dilation=5)
        self.conv_atrous_7 = nn.Conv2d(c_, c_, kernel_size=3, padding=7, dilation=7)

        self.conv_final = nn.Conv2d(c_ * 4, c2, kernel_size=1, padding=1)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_initial(x)
        out3 = self.conv3x3(out2)
        out4 = self.conv3x3(out3)

        out5 = self.conv_atrous_1(out1)
        out6 = self.conv_atrous_3(out2)
        out7 = self.conv_atrous_5(out3)
        out8 = self.conv_atrous_7(out4)

        out9 = torch.cat([out5, out6, out7, out8], dim=1)
        shortcut = self.conv3x3_shortcut(out1)
        net = 0.8 * out9 + shortcut
        out = F.relu(net)
        return out

# Example usage:
c1 = 64
c2 = 256
a = 4
input = torch.randn(1, c1, 640, 640)
model = ESPP(c1, c2, a)
output = model(input)
print(output.shape)  # torch.Size([1, 64, 640, 640])