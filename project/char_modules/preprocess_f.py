import numpy as np

# function to remove nan values
def remove_nan(data):
  inputs = []
  labels = []

  for input,label in data:

    labels.append(int(label))

    # get length excluding nan values which indicate end of input
    lens = []
    for channel in input:
      channel = list(channel)
      len1 = np.where(np.isnan(list(channel)))[0][0] if any(np.isnan(channel)) else len(channel)
      lens.append(len1)
    inputLen = np.min(lens)

    channels = []
    for channel in input:
      channels.append(list(channel)[:inputLen])
    inputs.append(channels)

  data1 = list(zip(inputs,labels))
  return data1


def interpolate(arr,newSize):
  l = len(arr)
  indices = list(range(0,l))
  newIndices = np.linspace(0, l-1 , newSize)
  newArr = [np.interp(i,indices,arr) for i in newIndices]
  return newArr


# function to resize data samples
def resize_samples(data, resizeTo):
  inputs = []
  labels = []

  for input,label in data:

    labels.append(label)

    channels = []
    for channel in input:
      channels.append(interpolate(channel,resizeTo))
    inputs.append(channels)

  inputs, labels = np.array(inputs), np.array(labels)
  return inputs, labels