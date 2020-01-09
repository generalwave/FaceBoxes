import torch.nn as nn
import torch
import torch.nn.functional as functional


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.02)
        else:
            m.weight.data.normal_(0, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation_fn=nn.ReLU()):
        super(BasicConv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        return x


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat([x, -x], dim=1)
        x = functional.relu(x, inplace=True)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, out_channels, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, reduce_channels, kernel_size=1),
            BasicConv2d(reduce_channels, out_channels, kernel_size=3)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, reduce_channels, kernel_size=1),
            BasicConv2d(reduce_channels, out_channels, kernel_size=3),
            BasicConv2d(out_channels, out_channels, kernel_size=3)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


class RDCL(nn.Module):
    def __init__(self):
        super(RDCL, self).__init__()
        self.conv1 = BasicConv2d(in_channels=3, out_channels=24, kernel_size=7, stride=4, activation_fn=CReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(in_channels=48, out_channels=64, kernel_size=5, stride=2, activation_fn=CReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class RDCL2(nn.Module):
    def __init__(self):
        super(RDCL2, self).__init__()
        self.conv1 = BasicConv2d(in_channels=3, out_channels=12, kernel_size=7, stride=2, activation_fn=nn.ReLU())
        self.pool1 = BasicConv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2, activation_fn=CReLU())
        self.conv2 = BasicConv2d(in_channels=48, out_channels=32, kernel_size=3, stride=2, activation_fn=nn.ReLU())
        self.pool2 = BasicConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, activation_fn=CReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class MSCL(nn.Module):
    def __init__(self):
        super(MSCL, self).__init__()
        self.inception1 = Inception(in_channels=128, out_channels=32, reduce_channels=24)
        self.inception2 = Inception(in_channels=128, out_channels=32, reduce_channels=24)
        self.inception3 = Inception(in_channels=128, out_channels=32, reduce_channels=24)
        self.conv3_1 = BasicConv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv3_2 = BasicConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4_1 = BasicConv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv4_2 = BasicConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        feature_maps = [x]

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        feature_maps.append(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        feature_maps.append(x)

        return feature_maps


class MultiBox(nn.Module):
    def __init__(self, num_classes, phase):
        super(MultiBox, self).__init__()
        self.num_classes = num_classes
        self.phase = phase

        self.conf0 = nn.Conv2d(in_channels=128, out_channels=num_classes * 21, kernel_size=3, padding=1)
        self.loc0 = nn.Conv2d(in_channels=128, out_channels=4 * 21, kernel_size=3, padding=1)

        self.conf1 = nn.Conv2d(in_channels=256, out_channels=num_classes * 1, kernel_size=3, padding=1)
        self.loc1 = nn.Conv2d(in_channels=256, out_channels=4 * 1, kernel_size=3, padding=1)

        self.conf2 = nn.Conv2d(in_channels=256, out_channels=num_classes * 1, kernel_size=3, padding=1)
        self.loc2 = nn.Conv2d(in_channels=256, out_channels=4 * 1, kernel_size=3, padding=1)

    def forward(self, feature_maps):
        conf, loc = [], []

        conf.append(self.conf0(feature_maps[0]).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc0(feature_maps[0]).permute(0, 2, 3, 1).contiguous())

        conf.append(self.conf1(feature_maps[1]).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc1(feature_maps[1]).permute(0, 2, 3, 1).contiguous())

        conf.append(self.conf2(feature_maps[2]).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc2(feature_maps[2]).permute(0, 2, 3, 1).contiguous())

        conf = torch.cat([x.view(x.size(0), -1, self.num_classes) for x in conf], dim=1)
        loc = torch.cat([x.view(x.size(0), -1, 4) for x in loc], dim=1)

        if self.phase == "test":
            conf = functional.softmax(conf, dim=-1)

        return conf, loc


class FaceBoxes(nn.Module):
    def __init__(self, num_classes, phase):
        super(FaceBoxes, self).__init__()
        self.rdcl = RDCL()
        self.mscl = MSCL()
        self.multibox = MultiBox(num_classes, phase)
        self.apply(weight_init)

    def forward(self, x):
        conf, loc = self.multibox(self.mscl(self.rdcl(x)))
        return conf, loc
