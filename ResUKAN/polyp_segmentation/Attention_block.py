import torch.nn as nn
import torch
from kan import KAN, KANLinear



def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


class CA_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(CA_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        # self.kan_conv1 = FastKANConv2DLayer(input_dim=inplanes, output_dim=planes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        # self.kan_conv2 = FastKANConv2DLayer(input_dim=planes, output_dim=planes * 2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        # self.kan_conv3 = FastKANConv2DLayer(input_dim=planes * 2, output_dim=planes, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)

        # self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        # self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.kan = KAN([planes * 2, round(planes / 2), planes * 2])
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.kan_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.kan_conv2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.global_avg_pooling(out)
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        out = self.kan(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.global_max_pooling(out1)
        out1 = out1.view(out1.size(0), -1)
        # out1 = self.fc1(out1)
        # out1 = self.relu(out1)
        # out1 = self.fc2(out1)
        out1 = self.kan(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)
        out = self.sigmoid(out)

        # out = self.kan_conv3(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out




