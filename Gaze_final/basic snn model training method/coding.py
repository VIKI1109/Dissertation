import torch.nn.functional as F
import torch.nn as nn
import torch
from lif import Lif
import numpy as np
from spikingjelly.clock_driven import encoding

TimeStep = 16

# 基础CNN
class CNN_MLP(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_MLP, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2, 36*60=2160
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out

# SNN，不编码
class SNN_MLP(nn.Module):
    def __init__(self, num_classes=2):
        super(SNN_MLP, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2, 36*60=2160
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.spike = Lif()

    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = data_temp

        out = self.flatten(data)

        out = self.spike(self.fc1(out))
        out = self.spike(self.fc2(out))
        out = self.spike(self.fc3(out))
        out = self.spike(self.fc4(out))
        out = self.fc5(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

# SNN，泊松编码
class Poisson_MLP(nn.Module):
    def __init__(self, num_classes=2):
        super(Poisson_MLP, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2, 36*60=2160
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            # prop是生成的0-1的随机数，如果小于对应元素，则发放1
            prop = torch.rand_like(data_temp)
            data[t * bs:(t + 1) * bs, ...] = torch.where(prop < data_temp, 1.0, 0.0)

        out = self.flatten(data)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

# SNN，移位编码， Rank coding
class Rank_MLP(nn.Module):
    def __init__(self, num_classes=2):
        super(Rank_MLP, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2, 36*60=2160
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 编码规则0.1-->[1 0 0 0], 0.3-->[1 1 0 0], 0.66-->[1 1 1 0], 0.88-->[1 1 1 1], 超过多少个rank，就编码多少次
        rank = [0.0, 0.25, 0.5, 0.75]
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = torch.where(data_temp > rank[t], 1.0, 0.0)

        out = self.flatten(data)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

# SNN，高斯群编码， GRF coding
class GRF_MLP(nn.Module):
    def __init__(self, num_classes=2):
        super(GRF_MLP, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2, 36*60=2160
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def gaussian_receptive_field_encoding(data, num_neurons, beta):
        # Step 2: Calculate the center positions of the receptive field (c_i) and standard deviation (sigma)
        centers = [(2 * i - 3) / (2 * (num_neurons - 2)) for i in range(num_neurons)]
        sigma = 1 / (beta * (num_neurons - 2))

        # Step 3: Compute the probabilities for each neuron based on the Gaussian function
        probabilities = [np.exp(-((x - c) ** 2) / (2 * sigma ** 2)) for x in data for c in centers]

        # Step 4: Map the probabilities to spike times
        max_prob = max(probabilities)
        spike_times = [int(round(TimeStep * (1 - p / max_prob))) for p in probabilities]

        return spike_times

    def forward(self, x):
        # 编码规则0.1-->[1 0 0 0], 0.3-->[1 1 0 0], 0.66-->[1 1 1 0], 0.88-->[1 1 1 1], 超过多少个rank，就编码多少次
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = torch.where(data_temp == t, 1.0, 0.0)

        out = self.flatten(data)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

class CNN_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(CNN_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128, 2) # regression layer
        self.float()
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # op = 16x28x20
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # op = 6x12x50
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x)) # regression layer for real output
        x = self.reg(x)
        return x

# 不编码
class SNN_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(SNN_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 10, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(10, 20, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*20, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()
    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = data_temp

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2)) # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2)) # op = 6x12x50
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x)) # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o

# 泊松编码
class Poisson_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(Poisson_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()

    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            # prop是生成的0-1的随机数，如果小于对应元素，则发放1
            prop = torch.rand_like(data_temp)
            data[t * bs:(t + 1) * bs, ...] = torch.where(prop < data_temp, 1.0, 0.0)

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2))  # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2))  # op = 6x12x50
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x))  # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o

# SNN移位编码
class Rank_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(Rank_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()

    def forward(self, x):
        # 编码规则0.1-->[1 0 0 0], 0.3-->[1 1 0 0], 0.66-->[1 1 1 0], 0.88-->[1 1 1 1], 超过多少个rank，就编码多少次
        # rank = [0.0, 0.25, 0.5, 0.75]
        rank = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.6, 0.66, 0.72, 0.8, 0.86, 0.92, 0.96]
        data_temp = x
        bs = data_temp.shape[0]

        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = torch.where(data_temp > rank[t], 1.0, 0.0)

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2))  # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2))  # op = 6x12x50
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x))  # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o

# SNN，高斯群编码， GRF coding
class GRF_GazeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GRF_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5)  # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5)  # op = 12x24x50
        self.fc1 = nn.Linear(6 * 12 * 50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128, 2)  # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()

    def gaussian_receptive_field_encoding(data, num_neurons, beta):
        # Step 2: Calculate the center positions of the receptive field (c_i) and standard deviation (sigma)
        centers = [(2 * i - 3) / (2 * (num_neurons - 2)) for i in range(num_neurons)]
        sigma = 1 / (beta * (num_neurons - 2))

        # Step 3: Compute the probabilities for each neuron based on the Gaussian function
        probabilities = [np.exp(-((x - c) ** 2) / (2 * sigma ** 2)) for c in centers for x in data]

        return probabilities

    def forward(self, x):
        data = self.gaussian_receptive_field_encoding(data=x, num_neurons=TimeStep, beta=1)

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2))  # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2))  # op = 6x12x50
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x))  # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep

        return o

# SNN延迟编码
class Latency_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(Latency_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()

    def forward(self, x):
        data_temp = x
        bs = data_temp.shape[0]
        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)

        le = encoding.LatencyEncoder(TimeStep)
        for t in range(TimeStep):

            data[t * bs:(t + 1) * bs, ...] = le(x)

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2))  # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2))  # op = 6x12x50
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x))  # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o

# SNN带权相位编码
class Weighted_GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(Weighted_GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
        self.spike1 = Lif()
        self.spike2 = Lif()
        self.spike3 = Lif()

    def forward(self, x):
        data_temp = x

        # 带权相位编码器编码范围是0到1-2^(-T), 而输入的x范围是0到1，因此这里将超过的裁剪到1-2^T
        x = torch.where(x >= 1-2**(-TimeStep), 1-2**(-TimeStep), x)

        bs = data_temp.shape[0]
        data = torch.zeros((TimeStep * bs,) + data_temp.shape[1:], device=data_temp.device)

        weightedPhaseEncoder = encoding.WeightedPhaseEncoder(TimeStep)
        for t in range(TimeStep):
            data[t * bs:(t + 1) * bs, ...] = weightedPhaseEncoder(x)

        x = F.max_pool2d(self.spike1(self.conv1(data)), (2, 2))  # op = 16x28x20
        x = F.max_pool2d(self.spike2(self.conv2(x)), (2, 2))  # op = 6x12x50
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.spike3(self.fc1(x))  # regression layer for real output
        out = self.reg(x)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o = o + out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o

