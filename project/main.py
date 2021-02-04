import argparse

import numpy as np
import itertools

# to get and read data
import os
from scipy.io import loadmat

# to plot the data
import matplotlib
import matplotlib.pyplot as plt

# for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# for clustering
from sklearn.cluster import KMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device :",device)





# import modules
from common_modules import data_f, network_f, patches_f, clusters_f
from char_modules import preprocess_f, plotting_f

# reload module
import importlib
importlib.reload(network_f)




def fusionApproach(train_inputs,clustFit_train_inputs,train_labels, test_inputs,clustFit_test_inputs,test_labels, args):
  
  def fuse(inputs1, inputs2):
    fused = []
    for ip1,ip2 in zip(inputs1,inputs2):
      ip = (ip1,ip2)
      fused.append(ip)
    return fused

  fused_train_inputs = fuse(train_inputs, clustFit_train_inputs)
  fused_test_inputs = fuse(test_inputs, clustFit_test_inputs)

  #print(train_inputs.shape)
  #print(clustFit_train_inputs.shape)

  # define network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
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

  if args.eval == "fusion":
    net = Net().to(device)
    net.load_state_dict(torch.load(args.load))
  else:
    net = create_train(Net,20, fused_train_inputs, train_labels, fused_test_inputs, test_labels, approach="fusion")
  
  evaluate(net, fused_train_inputs, train_labels, fused_test_inputs, test_labels, approach="fusion")

  return net 


def clustFitApproach(train_inputs, train_labels, test_inputs, test_labels, net, args):
  
  # model based on clusterfit predictions
  clustFit_train_inputs = network_f.netOutput(net, train_inputs)
  clustFit_test_inputs = network_f.netOutput(net, test_inputs)

  # 19 patches for every sample
  clustFit_train_inputs = patches_f.mergePatches(clustFit_train_inputs,19)
  clustFit_test_inputs = patches_f.mergePatches(clustFit_test_inputs,19)

  # define network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
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

  if args.eval == "clustFit":
    net = Net().to(device)
    net.load_state_dict(torch.load(args.load))
  else:
    net = create_train(Net,20, clustFit_train_inputs, train_labels, clustFit_test_inputs, test_labels)

  evaluate(net, clustFit_train_inputs, train_labels, clustFit_test_inputs, test_labels)

  return net, clustFit_train_inputs, clustFit_test_inputs



def latentApproach(train_inputs, test_inputs, net, args):

  # extract and cluster latent representation from the trained network for patches
  latent_train_inputs = network_f.netOutput(net, train_inputs, outType="latent")
  latent_test_inputs = network_f.netOutput(net, test_inputs, outType="latent")

  # flatten inputs to 2d array
  latent_train_inputs2d = clusters_f.flatten_to_2d(latent_train_inputs)
  latent_test_inputs2d = clusters_f.flatten_to_2d(latent_test_inputs)

  print("forming clusters... ", end="")
  kmeans = KMeans(n_clusters=20, random_state=0).fit(latent_train_inputs2d)
  print("done")
  kmeans_train_labels = kmeans.labels_
  kmeans_test_labels = kmeans.predict(latent_test_inputs2d)

  kmeans_train_labels = np.array([int(label) for label in kmeans_train_labels])
  kmeans_test_labels = np.array([int(label) for label in kmeans_test_labels])

  
  # define network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
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

    def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features

  if args.eval == "clustFit":
    net = Net().to(device)
    net.load_state_dict(torch.load(args.load))
  else:
    net = create_train(Net,10, train_inputs, kmeans_train_labels, test_inputs, kmeans_test_labels)
    
  evaluate(net, train_inputs, kmeans_train_labels, test_inputs, kmeans_test_labels)

  return net



def simpleApproach(train_inputs, train_labels, test_inputs, test_labels, args):

  # define network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
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

  if args.eval == "simple":
    net = Net().to(device)
    net.load_state_dict(torch.load(args.load))
  else:
    net = create_train(Net,20, train_inputs, train_labels, test_inputs, test_labels)

  evaluate(net, train_inputs, train_labels, test_inputs, test_labels)

  return net



def create_train(Net, patience, train_inputs, train_labels,test_inputs,test_labels,approach=None):
  # create dataset and dataloader
  train_loader, val_loader, test_loader = data_f.createLoaders(train_inputs, train_labels,
                                                              test_inputs, test_labels,
                                                              batch_size=32)

  # create network
  net = Net().to(device)

  #Define a Loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  scheduler = ReduceLROnPlateau(optimizer, 'min') 

  # train network
  best_params = network_f.trainNet(net,criterion,optimizer, scheduler,
                                  train_loader, val_loader,
                                  epochs=150, earlyStopping=patience,
                                  approach=approach)

  net.load_state_dict(best_params)

  return net

def evaluate(net, train_inputs, train_labels,test_inputs,test_labels,approach=None):
  # evaluate network
  print("evaluation results on train data")
  network_f.evaluate(net,train_inputs,train_labels,approach=approach)
  print("evaluation results on test data")
  network_f.evaluate(net,test_inputs,test_labels,approach=approach)
  return

def printNetSizes():

  inp_size = 206; c0 = 3;   # c0 = 4 if indicator channel else 3
  k_conv = 4; k_pool = 2; c1 = 8; c2 = 16; c3 = 32;

  print("initial size  of  sample = %d x %d" % (c0,inp_size))
  conv1_outSize = inp_size-(k_conv-1)
  print("output  size after conv1 = %d x %d" % (c1,conv1_outSize))
  pool1_outSize = np.floor((conv1_outSize-(k_pool-1)-1)/k_pool + 1)
  print("output  size after pool1 = %d x %d" % (c1,pool1_outSize))

  conv2_outSize = pool1_outSize-(k_conv-1)
  print("output  size after conv2 = %d x %d" % (c2,conv2_outSize))
  pool2_outSize = np.floor((conv2_outSize-(k_pool-1)-1)/k_pool + 1)
  print("output  size after pool2 = %d x %d" % (c2,pool2_outSize))

  conv3_outSize = pool2_outSize-(k_conv-1)
  print("output  size after conv3 = %d x %d" % (c3,conv3_outSize))
  pool3_outSize = np.floor((conv3_outSize-(k_pool-1)-1)/k_pool + 1)
  print("output  size after pool3 = %d x %d" % (c3,pool3_outSize))

  return


def Clustering(inputs, labels):
  # flatten inputs to 2d array
  inputs2d = clusters_f.flatten_to_2d(inputs)

  # plot silhoutte index for number of cluster 2 and 20
  #_ = clusters_f.form_clusters(inputs2d, "KMeans", list(range(2,26)), labels)   #list(range(2,31))

  # visualize cluster centers
  print("forming clusters... ", end="")
  #kmeans_centers = clusters_f.form_clusters(inputs2d, "KMeans", [20], labels)
  kmeans = KMeans(n_clusters=20, random_state=0).fit(inputs2d)
  print("done")
  kmeans_centers = kmeans.cluster_centers_

  nsamples, nx, ny = inputs.shape
  sample_shape = nx, ny

  print("\ncluster centers visualized")
  plotting_f.plotClusters(kmeans_centers, sample_shape)  

  return

def Patches(train_inputs, train_labels, test_inputs, test_labels, vis=False):

  train_data = list(zip(train_inputs, train_labels))
  test_data = list(zip(test_inputs, test_labels))

  kwargs = {'window_size':20, 'stride':10}

  print("creating patches")
  kwargs['data'] = train_data
  patch_train_inputs, patch_train_labels, patch_train_indexes = patches_f.dataToPatches(**kwargs)
  kwargs['data'] = test_data
  patch_test_inputs, patch_test_labels, patch_test_indexes = patches_f.dataToPatches(**kwargs)
  
  if vis:
    # visualize patches
    plotting_f.plotTimeSeries(train_data, patch_train_indexes, patch_train_inputs, char='a', index=0)
    plotting_f.plotChar(train_data, patch_train_indexes, patch_train_inputs, char='a', index=0)

    plt.show()

  return patch_train_inputs, patch_train_labels, patch_train_indexes, patch_test_inputs, patch_test_labels, patch_test_indexes  


def printLengths(train_inputs,test_inputs):
  # distribution of sample lenghts
  lengths = []

  for sample in itertools.chain(train_inputs,test_inputs):
    input = sample[0]
    input = np.array(input)
    lengths.append(len(input))

  print('max length =',np.max(lengths))
  _ = plt.hist(lengths)
  plt.show()


def getRead_data():

  # get data
  fsource = "https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/mixoutALL_shifted.mat"
  fname = fsource[fsource.rindex('/')+1:] # fname = "mixoutALL_shifted.mat"
  data_f.download_file(url = fsource,
                        saveAs = fname)

  #load the file
  mat = loadmat('mixoutALL_shifted.mat')
  #print(mat.keys())


  # read data
  consts = mat['consts'][0][0]
  #print(consts)

  classes = [char[0] for char in consts[3][0]]
  #print(classes)
  #print('number of classes :',len(classes))

  #subtract 1 since np array indexing is from 0
  labels = consts[4][0] - 1
  inputs = mat['mixout'][0]

  train_inputs, test_inputs, train_labels, test_labels = data_f.train_test_split(inputs, labels, test_size=0.25, random_state=0)

  train_labels = np.array([int(label) for label in train_labels])
  test_labels = np.array([int(label) for label in test_labels])

  #append zeroes to resize
  train_inputs, target_len = patches_f.append_defaults(train_inputs, 206)
  test_inputs, _ = patches_f.append_defaults(test_inputs, 206)

  return train_inputs, test_inputs, train_labels, test_labels









def main(args):

  print()
  train_inputs, test_inputs, train_labels, test_labels = getRead_data()
  print()
  #printLengths(train_inputs,test_inputs)
  patch_train_inputs, patch_train_labels, patch_train_indexes, patch_test_inputs, patch_test_labels, patch_test_indexes = Patches(train_inputs, train_labels, test_inputs, test_labels)


  if args.saveModel and not os.path.exists("models/"):
    os.mkdir("models")    
  
  if args.clustering:
    Clustering(patch_train_inputs, patch_train_labels)

  if (args.train is not None) or (args.eval is not None):
    net = simpleApproach(train_inputs,train_labels, test_inputs,test_labels, args)
    torch.save(net.state_dict(), args.saveSimple) if args.saveSimple else None

  if "latent" in (args.train, args.eval):
    net1 = latentApproach(patch_train_inputs, patch_test_inputs, net, args)
    torch.save(net1.state_dict(), args.saveLatent) if args.saveLatent else None
  
  if (args.train in ("clustFit", "fusion")) or (args.eval in ("clustFit", "fusion")):
    net2, clustFit_train_inputs, clustFit_test_inputs = clustFitApproach(patch_train_inputs,train_labels, patch_test_inputs,test_labels, net, args)
    torch.save(net2.state_dict(), args.saveClustFit) if args.saveClustFit else None
  
  if "fusion" in (args.train, args.eval):
    net3 = fusionApproach(train_inputs,clustFit_train_inputs,train_labels, test_inputs,clustFit_test_inputs,test_labels, args)
    torch.save(net3.state_dict(), args.saveFusion) if args.saveFusion else None

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'approaches')
    parser.add_argument('--clustering', type=bool, default=False)

    parser.add_argument('--train', type=str ,choices=["simple","latent","clustFit","fusion"])
    parser.add_argument('--load', type=str)
    parser.add_argument('--eval', type=str ,choices=["simple","latent","clustFit","fusion"])

    parser.add_argument('--saveSimple', type=str, default="models/simpleModel")
    parser.add_argument('--saveLatent', type=str, default="models/latentModel")
    parser.add_argument('--saveClustFit', type=str, default="models/clustFitModel")
    parser.add_argument('--saveFusion', type=str, default="models/fusionModel")
    args = parser.parse_args()
    main(args)