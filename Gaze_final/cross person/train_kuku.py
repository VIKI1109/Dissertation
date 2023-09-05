import model
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from cnn_vgg16 import CNN_VGG16
from vgg16_base import SNN_VGG16_Base
from vgg_eca import SNN_VGG16_ECA
from vgg16_mlf import SNN_VGG16_MLF
from vgg16_tdbn import SNN_VGG16_TDBN

class GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # from maxpool2 dimension
        self.reg = nn.Linear(128,2) # regression layer
        self.float()
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # op = 16x28x20
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # op = 6x12x50
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x)) # regression layer for real output
        x = self.reg(x)
        return x

class lenet_model(nn.Module):
    def __init__(self):
        super(lenet_model, self).__init__()

        self.convNet = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, 5, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.FC = nn.Sequential(
            nn.Linear(50 * 6 * 12, 500),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(502, 2)

    def forward(self, x_in):
        feature = self.convNet(x_in['eye'])
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)

        feature = torch.cat((feature, x_in['head_pose']), 1)
        gaze = self.output(feature)

        return gaze

from lif import Lif
TimeStep = 4
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

if __name__ == "__main__":
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    config = config["train"]
    cudnn.benchmark = True

    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]

    folder = os.listdir(labelpath)
    folder.sort()

    # i represents the i-th folder used as the test set.
    for i in range(0, 15):

        if i in list(range(15)):
            trains = copy.deepcopy(folder)
            tests = trains.pop(i)
            print(f"Train Set:{trains}")
            print(f"Test Set:{tests}")

            trainlabelpath = [os.path.join(labelpath, j) for j in trains]

            savepath = os.path.join(config["save"]["save_path"], f"checkpoint/{tests}")
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

            print("Read data")
            dataset = reader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=6,
                                     header=True)

            print("Model building")
            # net = SNN_GazeNet()
            # net = CNN_VGG16()
            # net = SNN_VGG16_Base()
            # net = SNN_VGG16_TDBN()
            # net = SNN_VGG16_ECA()
            net = SNN_VGG16_MLF()
            net.train()
            net.to(device)

            print("optimizer building")
            lossfunc = config["params"]["loss"]
            loss_op = getattr(nn, lossfunc)().cuda()
            base_lr = config["params"]["lr"]

            decaysteps = config["params"]["decay_step"]
            decayratio = config["params"]["decay"]

            optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

            print("Traning")
            length = len(dataset)
            total = length * config["params"]["epoch"]
            cur = 0
            timebegin = time.time()
            with open(os.path.join(savepath, "train_log"), 'w') as outfile:
                for epoch in range(1, config["params"]["epoch"] + 1):
                    for i, (data, label) in enumerate(dataset):

                        # Acquire data
                        data = data["eye"][:, :1, :, :].to(device)
                        # data["eye"] = data["eye"][:, :1, :, :].to(device)
                        # data['head_pose'] = data['head_pose'].to(device)
                        label = label.to(device)

                        # forward
                        gaze = net(data)

                        # loss calculation
                        loss = loss_op(gaze, label)
                        optimizer.zero_grad()

                        # backward
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        cur += 1

                        # print logs
                        if i % 20 == 0:
                            timeend = time.time()
                            resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                            log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
                            print(log)
                            outfile.write(log + "\n")
                            sys.stdout.flush()
                            outfile.flush()

                    if epoch % config["save"]["step"] == 0:
                        torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

