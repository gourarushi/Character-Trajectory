import os

# to read data
import requests

# to create dataset and dataloaders
from torch.utils.data import DataLoader, Dataset

# for splitting data
from sklearn.model_selection import train_test_split


# download file from url
def download_file(url,saveAs):
    if not os.path.exists(saveAs):
        r = requests.get(url, allow_redirects=True)
        open(saveAs, 'wb').write(r.content)
        print('file downloaded')
    else:
        print('file already exists')

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
