import numpy as np

# for filtering and normalization
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.preprocessing import MinMaxScaler

# to track progress
from tqdm.notebook import tqdm


def interpolate(arr,newSize):
  l = len(arr)
  indices = list(range(0,l))
  newIndices = np.linspace(0, l-1 , newSize)
  newArr = [np.interp(i,indices,arr) for i in newIndices]
  return newArr


# function to return list of patches for a given dataset
def dataToPatches(data, window_size, stride, resizeTo=False, medianFilter=False, gaussianFilter=False, normalize=False):
  inputs = []
  labels = []
  indexes = []

  for index,(input,label) in enumerate(tqdm(data)):

    inputLen = len(input[0])

    for i in range(0, inputLen, stride):
      channels = []
      # verify if last stride is possible
      if i + window_size in range(inputLen + 1):
        # 4th input channel is useless
        for channel in input[:3]:
          values  = [0]*i
          values += list(channel)[i:i+window_size]
          values += [0]*(inputLen - i - window_size)
          if resizeTo:
            values = interpolate(values,resizeTo)  
          # apply gaussian filter for smoothing and reducing noise
          if medianFilter:
              values = median_filter(values, size=3)
          # apply gaussian filter for smoothing and reducing noise
          if gaussianFilter:
              values = gaussian_filter(values, sigma=1)

          indicator  = [0]*i
          indicator += [1]*window_size
          indicator += [0]*(inputLen - i - window_size)
          channels.append(values)
          
        # Normalize between 0 and 1
        if normalize:
            channels = np.array(channels)
            shape = channels.shape
            channels = list(MinMaxScaler(normalize).fit_transform(channels.reshape(-1,1)).reshape(shape))

        if resizeTo:
          indicator = interpolate(indicator,resizeTo) 
        
        #channels.append(indicator)

        inputs.append(channels)
        labels.append(label)
        indexes.append(index)

  inputs, labels, indexes = np.array(inputs), np.array(labels, dtype=int), np.array(indexes)
  return inputs,labels,indexes

def mergePatches(inputs, patchesPerSample):

  merged = []
  for index in range(0,len(inputs),patchesPerSample):
    input1 = np.array([x[0] for x in inputs[index : index+patchesPerSample]])
    input1 = np.transpose(input1)
    merged.append(input1)

  return np.array(merged)  