import model
import reader
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
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

def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


if __name__ == "__main__":
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    config = config["test"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["load"]["model_name"]

    loadpath = os.path.join(config["load"]["load_path"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder = os.listdir(labelpath)
    i = int(sys.argv[2])

    if i in range(15):
        # tests = folder[i]
        tests = 'p14.label'
        print(f"Test Set: {tests}")

        savepath = os.path.join(loadpath, f"checkpoint/{tests}")
        # savepath = os.path.join(loadpath, f"checkpoint/p00.label")

        if not os.path.exists(os.path.join(loadpath, f"evaluation/{tests}")):
            os.makedirs(os.path.join(loadpath, f"evaluation/{tests}"))

        print("Read data")
        dataset = reader.txtload(os.path.join(labelpath, tests), imagepath, 10, shuffle=False, num_workers=4,
                                 header=True)

        begin = config["load"]["begin_step"]
        end = config["load"]["end_step"]
        step = config["load"]["steps"]

        for saveiter in range(begin, end + step, step):
            print("Model building")
            net = SNN_GazeNet()
            # net = CNN_VGG16()
            # net = SNN_VGG16_Base()
            # net = SNN_VGG16_TDBN()
            # net = SNN_VGG16_ECA()
            # net = SNN_VGG16_MLF()
            print(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))
            statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))

            net.to(device)
            net.load_state_dict(statedict)
            net.eval()

            print(f"Test {saveiter}")
            length = len(dataset)
            accs = 0
            count = 0
            with torch.no_grad():
                with open(os.path.join(loadpath, f"evaluation/{tests}/{saveiter}.log"), 'w') as outfile:
                    outfile.write("name results gts\n")
                    for j, (data, label) in enumerate(dataset):
                        # Acquire data
                        img = data["eye"][:, :1, :, :].to(device)
                        # data["eye"] = data["eye"][:, :1, :, :].to(device)
                        # data['head_pose'] = data['head_pose'].to(device)

                        # img = data["eye"].to(device)
                        # headpose = data["head_pose"].to(device)
                        names = data["name"]

                        # img = {"eye": img, "head_pose": headpose}
                        gts = label.to(device)

                        gazes = net(img)
                        for k, gaze in enumerate(gazes):
                            gaze = gaze.cpu().detach().numpy()
                            count += 1
                            accs += angular(gazeto3d(gaze), gazeto3d(gts.cpu().numpy()[k]))

                            name = [names[k]]
                            gaze = [str(u) for u in gaze]
                            gt = [str(u) for u in gts.cpu().numpy()[k]]
                            log = name + [",".join(gaze)] + [",".join(gt)]
                            outfile.write(" ".join(log) + "\n")

                    loger = f"[{saveiter}] Total Num: {count}, avg: {accs / count}"
                    outfile.write(loger)
                    print(loger)

