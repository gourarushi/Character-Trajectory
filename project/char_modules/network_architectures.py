# for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define network
class simpleNet(nn.Module):
  def __init__(self):
    super(simpleNet, self).__init__()
    # 3 input channels, 8 output channels, row convolution kernel of size 3
    self.conv1 = nn.Conv1d(3, 8, 3)
    self.conv2 = nn.Conv1d(8, 16, 3)
    self.conv3 = nn.Conv1d(16, 32, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(768, 20)

  def forward(self, x):
    # output given by : math.floor((inp-(k-1)-1)/s+1)
    x = F.max_pool1d(F.relu(self.conv1(x)), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = F.max_pool1d(F.relu(self.conv3(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = self.fc1(x)
    return x

  def latent(self, x):
    # output given by : math.floor((inp-(k-1)-1)/s+1)
    x = F.max_pool1d(F.relu(self.conv1(x)), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = F.max_pool1d(F.relu(self.conv3(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# define network
class clustFitNet(nn.Module):
  def __init__(self):
    super(clustFitNet, self).__init__()
    self.fc1 = nn.Linear(380, 20)

  def forward(self, x):        
    x = x.view(-1, self.num_flat_features(x))
    x = self.fc1(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


 # define network
class fusionNet(nn.Module):
  def __init__(self):
    super(fusionNet, self).__init__()
    # 3 input channels, 8 output channels, row convolution kernel of size 3
    self.conv1 = nn.Conv1d(3, 8, 3)
    self.conv2 = nn.Conv1d(8, 16, 3)
    self.conv3 = nn.Conv1d(16, 32, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(1148, 20)

  def forward(self, x):
    x1, x2 = x
    x1, x2 = x1.to(device), x2.to(device)
    # output given by : math.floor((inp-(k-1)-1)/s+1)
    x = F.max_pool1d(F.relu(self.conv1(x1)), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = F.max_pool1d(F.relu(self.conv3(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x2 = x2.view(-1, self.num_flat_features(x2))
    x = torch.cat((x,x2),dim=1)
    x = self.fc1(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features    