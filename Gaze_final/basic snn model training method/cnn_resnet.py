from math import sqrt
import torch.nn.functional as F
import torch.nn as nn
import torch

TimeStep = 4

# 基础残差块，通常包含两个卷积层
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, modified=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)

        # self.bn1 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)

        # self.bn2 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.bn2 = nn.BatchNorm2d(planes)

        self.spike_func = nn.ReLU()
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    # tdBatchNorm(nn.BatchNorm2d(planes)),
                    nn.BatchNorm2d(planes),
                    nn.ReLU()
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    # tdBatchNorm(nn.BatchNorm2d(planes)),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.modified:
            out = F.relu(out)
            out = out + self.shortcut(x)          # Equivalent to union of all spikes
        else:
            out = out + self.shortcut(x)
            out = F.relu(out)
        return out

# 残差块的组合：由多个残差块组成，第一个残差块可能进行下采样操作(downsample)使x高宽减半
class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, downsample, modified):
        super(BLock_Layer, self).__init__()
        layers = []

        # 根据是否下采样，改变步长，2为下采样1维持
        if downsample:
            layers.append(block(in_planes, planes, 2, modified))
        else:
            layers.append(block(in_planes, planes, 1, modified))
        for _ in range(1, num_block):
            layers.append(block(planes, planes, 1, modified))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class ResNet(nn.Module):
    """ Establish ResNet.
     Spiking DS-ResNet with “modified=True.”
     Spiking ResNet with “modified=False.”
     """
    def __init__(self, block, num_block_layers, num_classes=16):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        # self.bn1 = tdBatchNorm(nn.BatchNorm2d(64))
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = BLock_Layer(block, 64, 64, num_block_layers[0], False, modified=True)
        self.layer2 = BLock_Layer(block, 64, 128, num_block_layers[1], True, modified=True)
        self.layer3 = BLock_Layer(block, 128, 256, num_block_layers[2], True, modified=True)
        # self.layer3 = BLock_Layer(block, 256, 512, num_block_layers[3], True, modified=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.voting = nn.Linear(256, num_classes)
        self.spike_func = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # out = self.spike_func(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)

        out = self.voting(out)

        return out


def cnn_resnet14():
    return ResNet(BasicBlock, [2, 2, 2], num_classes=2)

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet68():
    return ResNet(BasicBlock, [11, 11, 11])

# net = cnn_resnet14()
# print(net)
# print(net(torch.rand(128, 1, 36, 60)))