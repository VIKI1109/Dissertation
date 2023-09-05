import torch
import torch.nn as nn

TimeStep = 4

class SNN_VGG16_MLF(nn.Module):
    def __init__(self, num_classes=2):
        super(SNN_VGG16_MLF, self).__init__()

        # VGG16共有3个VGG块，一个块由两到四个卷积构成
        # 这是第一个块，有两个卷积，通道由3升到64, (3, 32, 32)-->(64, 32, 32)
        self.conv_01 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_01 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_01 = activate_func()

        self.conv_02 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_02 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_02 = activate_func()

        self.pooling_01 = nn.AvgPool2d(2)

        # 这是第二个块，有两个卷积，通道由64升到128, (64, 16, 16)-->(128, 16, 16)
        self.conv_03 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_03 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_03 = activate_func()

        self.conv_04 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_04 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_04 = activate_func()

        self.pooling_02 = nn.AvgPool2d(2)

        # 这是第三个块，有三个卷积，通道由128升到256, (128, 16, 16)-->(256, 8, 8)
        self.conv_05 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_05 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_05 = activate_func()

        self.conv_06 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_06 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_06 = activate_func()

        self.conv_07 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_07 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_07 = activate_func()

        self.pooling_03 = nn.AvgPool2d(2)

        # 这是第四个块，有三个卷积，通道由256升到512, (512, 4, 4)-->(512, 4, 4)
        self.conv_08 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_08 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_08 = activate_func()

        self.conv_09 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_09 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_09 = activate_func()

        self.conv_10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_10 = activate_func()

        self.pooling_04 = nn.AvgPool2d(2)

        # 这是第五个块，有三个卷积，通道512, (512, 2, 2)-->(512, 2, 2)
        self.conv_11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_11 = activate_func()

        self.conv_12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_12 = activate_func()

        self.conv_13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.spike_13 = activate_func()

        self.pooling_05 = nn.AvgPool2d(2)

        # 到这里就没有卷积层了，后面是三个用于投票的全连接层:512 --> 256 --> 10
        self.fc_14 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.spike_14 = activate_func()

        self.fc_15 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.spike_15 = activate_func()

        self.fc_16 = nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = data_temp

        # 这是第一个块，有两个卷积
        out = self.conv_01(data)
        out = self.bn_01(out)
        out = self.spike_01(out)

        out = self.conv_02(out)
        out = self.bn_02(out)
        out = self.spike_02(out)

        out = self.pooling_01(out)

        # 这是第二个块，有两个卷积
        out = self.conv_03(out)
        out = self.bn_03(out)
        out = self.spike_03(out)

        out = self.conv_04(out)
        out = self.bn_04(out)
        out = self.spike_04(out)

        out = self.pooling_02(out)

        # 这是第三个块，有三个卷积
        out = self.conv_05(out)
        out = self.bn_05(out)
        out = self.spike_05(out)


        out = self.conv_06(out)
        out = self.bn_06(out)
        out = self.spike_06(out)

        out = self.conv_07(out)
        out = self.bn_07(out)
        out = self.spike_07(out)

        out = self.pooling_03(out)

        # 这是第四个块，有三个卷积
        out = self.conv_08(out)
        out = self.bn_08(out)
        out = self.spike_08(out)


        out = self.conv_09(out)
        out = self.bn_09(out)
        out = self.spike_09(out)

        out = self.conv_10(out)
        out = self.bn_10(out)
        out = self.spike_10(out)


        out = self.pooling_04(out)

        # 这是第五个块，有三个卷积
        out = self.conv_11(out)
        out = self.bn_11(out)
        out = self.spike_11(out)


        out = self.conv_12(out)
        out = self.bn_12(out)
        out = self.spike_12(out)


        out = self.conv_13(out)
        out = self.bn_13(out)
        out = self.spike_13(out)


        out = self.pooling_05(out)

        # 后面是用于投票的全连接层
        # 前面卷积层输出的是有多个通道的二维矩阵，全连接层只能接受一维输入，所以需要将矩阵拉平
        # view相当于reshape，调整张量的形状，out.shape[0]是一次输入所包含的样本数量, 后面-1表示自动计算
        out = out.view(out.shape[0], -1)

        # out = self.classifier(out)
        out = self.fc_14(out)
        out = self.spike_14(out)


        out = self.fc_15(out)
        out = self.spike_15(out)


        out = self.fc_16(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o += out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

Vth = 0.6
Vth2 = 1.0
Vth3 = 1.5
a = 1.0
tau = 0.25
momentum_SGD = 0.9

class MLF(nn.Module):
    def __init__(self):
        super(MLF, self).__init__()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        u2 = torch.zeros((bs,) + x.shape[1:], device=x.device)
        u3 = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            u2 = tau * u2 * (1 - spikefunc2(u2)) + x[t * bs:(t + 1) * bs, ...]
            u3 = tau * u3 * (1 - spikefunc3(u3)) + x[t * bs:(t + 1) * bs, ...]
            o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2) + spikefunc3(u3)

        return o

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth) < (a/2)) / a
        return grad_input * hu


spikefunc = SpikeFunction.apply


class SpikeFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth2)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth2) < (a/2)) / a
        return grad_input * hu

spikefunc2 = SpikeFunction2.apply

class SpikeFunction3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth3)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth3) < (a/2)) / a
        return grad_input * hu

spikefunc3 = SpikeFunction3.apply

activate_func = MLF