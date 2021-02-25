import numpy as np
from scipy.io import loadmat
import pickle

# to read data
import requests

# to create dataset and dataloaders
from torch.utils.data import DataLoader, Dataset

# for splitting data
from sklearn.model_selection import train_test_split


# download file from url
def download_file(url,saveAs):
  print("download file from",url)
  if not os.path.exists(saveAs):
      r = requests.get(url, allow_redirects=True)
      open(saveAs, 'wb').write(r.content)
      print('file downloaded')
  else:
      print('file already exists')

# align train_data and test_data lengths
def append_defaults(series, target=None, default=0, extraChannel=False):
  if target is None:
      target = np.max([len(d[0]) for d in series])
  result = []
  for d in series:
      if extraChannel:
        tmp = np.zeros((d.shape[0]+1, target))
      else:
        tmp = np.zeros((d.shape[0], target))
      for i, c in enumerate(d):
          tmp[i, :len(c)] = c
      result.append(tmp)
  return np.array(result), target        

def getRead_data(dataset):

  # get data
  if dataset == "Character Trajectories":
    fsource = "https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/mixoutALL_shifted.mat"
    fname = fsource[fsource.rindex('/')+1:] # fname = "mixoutALL_shifted.mat"
    download_file(url = fsource,
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

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.25, random_state=0)

    train_labels = np.array([int(label) for label in train_labels])
    test_labels = np.array([int(label) for label in test_labels])

    #append zeroes to resize
    train_inputs, target_len = append_defaults(train_inputs, 206)
    test_inputs, _ = append_defaults(test_inputs, 206)

  elif dataset == "Anomaly":
    download_file(url = 'https://drive.google.com/u/0/uc?id=1CdYxeX8g9wxzSnz6R51ELmJJuuZ3xlqa&export=download',
                          saveAs = 'anomaly_dataset.pickle')

    infile = open('anomaly_dataset.pickle','rb')
    data = pickle.load(infile)
    infile.close()

    # read data
    train_inputs, train_labels = data[0], data[1]
    test_inputs, test_labels = data[2], data[3]

    train_inputs = [np.transpose(input) for input in train_inputs]
    test_inputs = [np.transpose(input) for input in test_inputs]

    train_data = list(zip(train_inputs, train_labels))
    test_data = list(zip(test_inputs, test_labels))

    classes = ["normal","anomaly"]
    sample_len = 50
    # print('number of classes :',len(classes))

    # print('\ntrain data contains',len(train_data),'samples')
    # print('test data contains',len(test_data),'samples')

    # print('\neach sample has 3 channels : x,y and force')
    # print('length of each channel is', sample_len)

    train_inputs = np.array(train_inputs)
    test_inputs = np.array(test_inputs)

    train_labels = np.array(train_labels, dtype=int)
    test_labels = np.array(test_labels, dtype=int)

  return train_inputs, test_inputs, train_labels, test_labels     

# create dataset and dataloaders
class mydataset(Dataset):
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    input = self.inputs[index]
    label = self.labels[index]
    return input,label

# function to create train, val and test loaders
def createLoaders(train_inputs, train_labels, test_inputs, test_labels, batch_size, val_percent=.25):
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=val_percent, random_state=0)

    train_dataset = mydataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = mydataset(val_inputs, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = mydataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader
