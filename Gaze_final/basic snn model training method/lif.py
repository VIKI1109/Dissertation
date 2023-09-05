import torch
import torch.nn as nn
import torch.nn.functional as F

Vth = 0.6

# 代理梯度相关参数
a = 1.0
TimeStep = 4

tau = 0.25

class Lif(nn.Module):
    def __init__(self):
        super(Lif, self).__init__()

    def forward(self, x):
        # 四维张量：(b, c, h, w) = (batch, channel, height, width)
        # bs：batch, 每一批次所含样本数量，这里把输入层编码得到的多个时间步长
        # 对应的数据也放到了batch里，因此真实样本数量为第一维的size除以时间步长
        # 注：也可创造一个五维张量,即时间步长单独一个维度，这样的话，需要forward的次数
        # 等于时间步长，而本方案中，无论有多少个时间步长，只需forward一次
        bs = int(x.shape[0] / TimeStep)

        # 初始化电压，输出等变量
        # (bs,) + x.shape[1:], 经测试这是类似于数组拼接的语法
        # 最终得到的u是x剔除时间步长后的size，而输出o则与x的size保持一致
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        for t in range(TimeStep):
            # 根据对x的切片，可知不同样本同一时间步长的数据是紧挨的
            input = x[t * bs:(t + 1) * bs, ...]
            u = tau * u + input
            out = spikefunc(u)
            u = u - out * Vth
            o[t * bs:(t + 1) * bs, ...] = out

        return o

# 继承了torch.autograd.Function，表明这是自定义求导操作
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    # 该静态方法必须接受第一个参数为ctx(上下文)，类似于self可以用于保留到backward阶段使用的变量
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # torch.gt：比较input与Vth大小，大于为1否则为0
        output = torch.gt(input, Vth)
        # alpha = 8
        # sgx = torch.sigmoid(alpha*input)
        # grad = alpha * sgx * (1 - sgx)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 代理梯度：LIF激活函数不可导，故用此替代
        hu = (abs(input - Vth) < (a/2)) / a
        # alpha = 8
        # sgx = torch.sigmoid(alpha*input)
        # grad = alpha * sgx * (1 - sgx)
        return grad_input * hu

# Function里调用apply相当于Module里forward
spikefunc = SpikeFunction.apply