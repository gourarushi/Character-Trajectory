import numpy as np

# to plot the data
import matplotlib.pyplot as plt

# for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# to save parameters
import copy

# for evaluating results
from sklearn.metrics import classification_report

# to track progress
from tqdm.notebook import tqdm



# train network
def trainNet(net,criterion,optimizer,scheduler,train_loader,val_loader,epochs,print_every=None,earlyStopping=None,approach="defailt"):

  print("training network")
  # early stopping parameter indicates how long to wait after last time validation loss improved.

  if not print_every:
      print_every = int(epochs / 10)

  stopCounter = 0
  avg_trainLosses = []
  avg_valLosses = []

  for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

    train_loss = []
    val_loss = []

    net.train()
    for i, (inputBatch,labelBatch) in enumerate(train_loader):

        if approach == "fusion":
          inputBatch = [ip.to(device).float() for ip in inputBatch]
        else:
          inputBatch = inputBatch.to(device).float()

        labelBatch = labelBatch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputBatch = net(inputBatch)
        loss = criterion(outputBatch, labelBatch)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss.append(loss.item())

    net.eval()
    for i, (inputBatch,labelBatch) in enumerate(val_loader):
      with torch.no_grad():

        if approach == "fusion":
          inputBatch = [ip.to(device).float() for ip in inputBatch]
        else:
          inputBatch = inputBatch.to(device).float()

        # forward + backward + optimize
        outputBatch = net(inputBatch)
        loss = criterion(outputBatch, labelBatch)
        val_loss.append(loss.item())

    avg_trainLoss = sum(train_loss) / len(train_loss)
    avg_valLoss = sum(val_loss) / len(val_loss)
    avg_trainLosses.append(avg_trainLoss)

    if (epoch > 0) and (avg_valLoss < min(avg_valLosses)):
        best_params = copy.deepcopy(net.state_dict())
        best_epoch, best_loss = epoch, avg_valLoss
        stopCounter = 0        
    else:
      stopCounter += 1
      if stopCounter == earlyStopping:
        break
    avg_valLosses.append(avg_valLoss)   

    # print statistics
    if epoch % print_every == print_every - 1:
      print('epoch: %d, train loss: %.3f, val loss: %.3f' % (epoch + 1, avg_trainLoss, avg_valLoss))

    scheduler.step(avg_valLoss)          

  print('Finished Training')
  plt.plot(avg_trainLosses, label='train loss')
  plt.plot(avg_valLosses, label='val loss')
  plt.plot([best_loss]*epoch, linestyle='dashed')
  plt.plot(best_epoch, best_loss, 'o')
  plt.legend()
  plt.show()

  return best_params



# evaluate and print
def evaluate(net,input1,output_true,classes=None, approach="default"):

  output_pred = netOutput(net,input1,outType="class", approach=approach)

  if classes is not None:
    print(classification_report(output_true, output_pred, target_names=classes, labels=range(len(classes)) ,digits=4))
  else:
    print(classification_report(output_true, output_pred, digits=4))


def netOutput(net, inputs, outType="default", approach="default"):
  net.eval()
  outputs = []

  with torch.no_grad():
    for input1 in tqdm(inputs):
      if outType == "latent":
        output = net.latent(torch.from_numpy(input1).unsqueeze(0).float()).numpy()

      elif outType == "class":
        if approach == "fusion":
          input1 = [torch.from_numpy(ip).unsqueeze(0).float() for ip in input1]
          output = net(input1).numpy()
        else:  
          output = net(torch.from_numpy(input1).unsqueeze(0).float()).numpy()
        output = np.argmax(output)

      else:
        output = net(torch.from_numpy(input1).unsqueeze(0).float()).numpy()

      outputs.append(output)

  return np.array(outputs)



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