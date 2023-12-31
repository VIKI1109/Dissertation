import torch
import numpy as np
import math

import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import glob
import torchvision
import scipy.io

# import cv2
# from PIL import Image
# from torchvision.utils import make_grid
# from torchvision.io import read_image
# import torchvision.transforms.functional as Fv
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# 这是作为VGG和Resnet基线存在的四个网络
from snn_vgg16 import SNN_VGG16
from cnn_vgg16 import CNN_VGG16
from snn_resnet import snn_resnet18
from cnn_resnet import cnn_resnet14

# 这是编码方式实验涉及到的MLP
from coding import *
import time

TimeStep = 4

def convert_polar_vector_np(angles):
    y = -1 * math.sin(angles[0]) # first column is theta
    x = -1 * math.cos(angles[0]) * math.sin(angles[1])
    z = -1 * math.cos(angles[0]) * math.cos(angles[1])

    mag_v = math.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return np.array([x, y, z])

#####
## Convert polar angles of tensor batch [[theta, phi] (nx2)] into Cartesian unit vector tensor
#####
def convert_polar_vector(angles):
    y = -1 * torch.sin(angles[:,0]) # first column is theta
    x = -1 * torch.cos(angles[:,0]) * torch.sin(angles[:,1])
    z = -1 * torch.cos(angles[:,0]) * torch.cos(angles[:,1])

    mag_v = torch.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return x, y, z


#####
## Convert polar angles of single tensor [theta, phi] into Cartesian unit vector tensor
#####
def convert_polar_vector_single(angles):
    y = -1 * torch.sin(angles[0]) # first column of angles is pitches
    x = -1 * torch.cos(angles[0]) * torch.sin(angles[1])
    z = -1 * torch.cos(angles[0]) * torch.cos(angles[1])

    mag_v = torch.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return torch.tensor([x, y, z])
### END IMPORT


### CONFIG ###
data_path = "./MPIIGaze/Data/Normalized/" # training data path
save_path = "./models/lenet.pt" # save path for trained model
if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"
criterion = nn.MSELoss() # mean square loss
epochs = 50 # no of training epochs

### END CONFIG ###


### DATA LOADER ###
class MPIIGaze(Dataset): # custom data loader for MPIIGAZE normalized data
    def __init__(self, data_path, transform=None):
        self.image_list = []
        self.label_list = []
        self.pose_list = []
        for mat_file in glob.glob(data_path + "*/*.mat"):
            mat = scipy.io.loadmat(mat_file)
            gaze_list_arr = mat["data"][0].item()[0].item()[0] # gaze
            pose_list_arr = mat["data"][0].item()[0].item()[2]  # pose
            img_list_arr = mat["data"][0].item()[0].item()[1] # image
            for i in range(len(img_list_arr)):
                img_arr = img_list_arr[i]
                gaze_arr = gaze_list_arr[i]
                pose_arr = pose_list_arr[i]
                self.image_list.append(img_arr)
                self.label_list.append(gaze_arr)
                self.pose_list.append(pose_arr)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = self.transform(self.image_list[index])
        x = torch.tensor(self.label_list[index][0])
        y = torch.tensor(self.label_list[index][1])
        z = torch.tensor(self.label_list[index][2])
        theta = torch.asin(-1*y)
        phi = torch.atan2(-1*x, -1*z)
        g  = torch.tensor([theta, phi])
        #h = torch.tensor(self.pose_list[index])
        gaze_vec = torch.tensor(self.label_list[index])
        return img, g, gaze_vec

dataset = MPIIGaze(data_path)

train_size = int(0.7 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + valid_size)
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size]) # create train, validation and test sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)
### END DATA LOADER ###


### MODEL DEFINITION ###
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

# gazenet = GazeNet()
gazenet = SNN_VGG16()

# gazenet = SNN_VGG16()
# gazenet = CNN_VGG16()
# gazenet = snn_resnet14()
# gazenet = cnn_resnet14()

# gazenet = SNN_GazeNet()
# gazenet = Poisson_GazeNet()
# gazenet = Rank_GazeNet()
# gazenet = Latency_GazeNet()
# gazenet = Weighted_GazeNet()
# gazenet = SnnTorch_Lenet()
# gazenet = SnnTorch_AlexNet()
# gazenet = SnnTorch_VGG16()
print(gazenet)
optimizer = optim.Adam(gazenet.parameters()) # use Adam optimizer
### END MODEL DEF ###

### TRAIN ###
for epoch in range(epochs):
    print("start")
    t1 = time.time()
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    gazenet.to(device) # move model to device
    gazenet.train()
    counter = 0
    for inputs, labels, labels_gvec in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # move to device
        # torch.save(inputs[0][0], 'img.pt')
        optimizer.zero_grad() # clear optimizer
        outputs = gazenet.forward(inputs) # compute outputs
        loss = criterion(outputs.double(), labels.double()) # compute loss
        loss.backward() # backpropagation
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        counter += 1
        #print(counter, "/", len(train_loader))

    # Validating the model
    gazenet.eval()
    counter = 0
    with torch.no_grad():
        for inputs, labels, labels_gvec in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = gazenet.forward(inputs)
            valloss = criterion(outputs.double(), labels.double())
            val_loss += valloss.item()*inputs.size(0)
            counter += 1
            #print(counter, "/", len(valid_loader))

    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(valid_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    t2 = time.time()
    cost_time = t2 - t1
    speed = len(train_loader.dataset)/cost_time
    print('Training cost time: {:.6f} \tTraining speed: {:.6f}img/s'.format(cost_time, speed))

torch.save(gazenet.state_dict(), save_path) # save trained model
### END TRAIN ###

### EVALUATE TRAINED MODEL ###
test_loss = 0
accuracy = 0
with torch.no_grad():
    t1 = time.time()
    gazenet.to(device)
    gazenet.eval()
    counter = 0
    error = 0
    for inputs, labels_g, labels_gvec in test_loader:
        inputs, labels_g = inputs.to(device), labels_g.to(device)
        outputs = gazenet.forward(inputs)
        loss = criterion(outputs.double(), labels_g.double())
        test_loss += loss.item()*inputs.size(0)
        ip_x, ip_y, ip_z = convert_polar_vector(labels_g) # find cartesian gaze vector from ground truth
        op_x, op_y, op_z = convert_polar_vector(outputs) # find cartesian gaze vector from output
        costheta = ip_x*op_x + ip_y*op_y + ip_z*op_z # cosine similarity
        costheta = torch.clamp(costheta, min=-1.0, max=1.0) # prevent singularity when costheta exceeds limits due to Python
        error_rad = torch.acos(costheta) # error between output and truth
        error_deg = error_rad * 180 / math.pi
        error += torch.mean(error_deg)
        counter += 1
    test_error = error / counter # mean angle error on test set
    test_loss = test_loss/len(test_loader.dataset)
    print("Test loss: {:.4f} \tTest error: {:.4f}".format(test_loss, test_error))
    t2 = time.time()
    cost_time = t2 - t1
    speed = len(train_loader.dataset)/cost_time
    print('Training cost time: {:.6f} \tTraining speed: {:.6f}img/s'.format(cost_time, speed))
### END EVAL  ###

### VISUALIZE TEST DATA ###
examples = enumerate(test_loader)
batch_id, (images, labels_g, labels_gvec) = next(examples) # read a batch from the test set
images = images[0:16].to(device)
outputs = gazenet(images)
output_gazevec = torch.stack(convert_polar_vector(outputs), dim = 1) # get 3D gaze Cartesian vector
plt.figure(figsize=(20,7))
plt.suptitle("Head pose independent, model: LeNet")

for i in range(0,16): # plot projection of 3D gaze on the XY plane
    plt.subplot(4,4,i+1)
    g_vec = output_gazevec[i].squeeze().detach().cpu().numpy()
    ip_g_vec = labels_gvec[i].cpu().numpy()
    img_gray = images[i].squeeze().cpu()
    img_len_x = img_gray.shape[1]
    img_len_y = img_gray.shape[0]
    center_x = img_len_x / 2
    center_y = img_len_y / 2
    plt.quiver([center_x],[center_y],g_vec[0],-1*g_vec[1], scale=0.2, scale_units='inches', color="green")
    plt.quiver([center_x],[center_y],ip_g_vec[0],-1*ip_g_vec[1], scale=0.2, scale_units='inches', color="blue")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_gray, cmap="gray")
### END EVAL  ###