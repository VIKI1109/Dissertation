# -*- coding: utf-8 -*-
import torch
import numpy as np

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks/GazeNet

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import importlib
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
from cnn_vgg16 import CNN_VGG16
import math

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset):
  def __init__(self, path, root, train, num_lines,header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
              if train:
                line = f.readlines()[:num_lines]
                if header: line.pop(0)
              if not train:
                line = f.readlines()[num_lines:]

              self.lines.extend(line)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[0]
    gaze2d = line[5]
    head2d = line[6]
    eye = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

    img = cv2.imread(os.path.join(self.root, eye), 0)/255.0
    img = np.expand_dims(img, 0)

    info = {"eye":torch.from_numpy(img).type(torch.FloatTensor),
            "head_pose":headpose,
            "name":name}

    return info, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=6, header=True):
  train_set = loader(labelpath, imagepath, True, 2501, header)
  # test_set = loader(labelpath, imagepath,False, 2501, header)
  train = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
  # test = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
  print(f"[Read Data]: Train num: {len(train_set)}")
  # print(f"[Read Data]: Test num: {len(test_set)}")
  print(f"[Read Data]: Label path: {labelpath}")
  return train


def txtload_test(labelpath, imagepath, batch_size, shuffle=True, num_workers=6, header=True):
  # train_set = loader(labelpath, imagepath, True, 2501, header)
  test_set = loader(labelpath, imagepath,False, 2501, header)
  # train = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
  test = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
  # print(f"[Read Data]: Train num: {len(train_set)}")
  print(f"[Read Data]: Test num: {len(test_set)}")
  # print(f"[Read Data]: Label path: {labelpath}")
  return test

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
from snn_vgg16 import SNN_VGG16

class GazeNet(nn.Module): # LeNet based model
    def __init__(self):
        super(GazeNet, self).__init__()
        # 整个网络的input_size=1x36x60, output_size=2
        # 1 input image 36x60x1, channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
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
TimeStep = 8
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
  gaze_x = (-torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])).unsqueeze(1)
  gaze_y = (-torch.sin(gaze[:, 1])).unsqueeze(1)
  gaze_z = (-torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])).unsqueeze(1)
  gaze3d = torch.cat([gaze_x, gaze_y, gaze_z], 1)
  return gaze3d

def loss_op(gaze, label, device):
  totals = torch.sum(gaze*label, dim=1, keepdim=True)
  length1 = torch.sqrt(torch.sum(gaze*gaze, dim=1, keepdim=True))
  length2 = torch.sqrt(torch.sum(label*label, dim=1, keepdim=True))
  res = totals/(length1 * length2)
  angular = torch.mean(torch.acos(torch.min(res , torch.ones_like(res).to(device)*0.9999999)))*180/math.pi
  return angular

if __name__ == "__main__":
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    config = config["train"]
    cudnn.benchmark = True

    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]

    folder = os.listdir(labelpath)
    folder.sort()

    trains = copy.deepcopy(folder)
    trainlabelpath = [os.path.join(labelpath, j) for j in trains]

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint/test")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device("cuda:0")

    print("Read data")
    train_loader = txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=6,
                              header=True)

    print("Model building")
    net = SNN_GazeNet()
    net.train()
    net.to(device)

    print("optimizer building")
    loss_op = nn.MSELoss().cuda()
    # loss_op = getattr(nn, lossfunc)().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Training")
    length = len(train_loader)
    total = length * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            for i, (data, label) in enumerate(train_loader):
                # print(i)
                # Acquire data
                data = data["eye"][:, :1, :, :].to(device)

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

def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
    print("test")
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    config = config["test"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["load"]["model_name"]

    loadpath = os.path.join("./checkpoint/SNN_LeNet/checkpoint/test")
    device = torch.device("cuda:0")

    folder = os.listdir(labelpath)
    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]

    for saveiter in range(begin, end + step, step):
        print("Model building")
        net = SNN_GazeNet()
        savepath = config["load"]["load_path"]
        print(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))
        statedict = torch.load(os.path.join(loadpath, f"Iter_{saveiter}_{modelname}.pt"))
        folder = os.listdir(labelpath)
        folder.sort()

        tests = copy.deepcopy(folder)
        testlabelpath = [os.path.join(labelpath, j) for j in tests]
        test_loader = txtload_test(testlabelpath, imagepath, 256, num_workers=6, header=True)
        net.to(device)
        net.load_state_dict(statedict)
        net.eval()

        print(f"Test {saveiter}")
        length = len(test_loader)
        accs = 0
        count = 0
        with torch.no_grad():
            with open(os.path.join(loadpath, f"evaluation/{saveiter}.log"), 'w') as outfile:
                outfile.write("name results gts\n")
                for j, (data, label) in enumerate(test_loader):
                    # Acquire data
                    img = data["eye"][:, :1, :, :].to(device)
                    names = data["name"]

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