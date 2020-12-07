import numpy as np

# to plot the data
import matplotlib
import matplotlib.pyplot as plt

classes = ['a','b','c','d','e','g','h','l','m','n','o','p','q','r','s','u','v','w','y','z']

def toPlot_char(char):
    
    xVel  = char[0]
    yVel  = char[1]
    force = char[2]

    xPos  = np.cumsum(xVel)
    yPos  = np.cumsum(yVel)

    #normalize force between 0 and 1
    color = (force - np.min(force)) / (np.max(force)-np.min(force))
    #define color based on force
    colormap = matplotlib.cm.inferno

    X=[]; Y=[]; C=[]
    for i,c in enumerate(color[:-1]):
      #_ = plt.plot([xPos[i],xPos[i+1]], [yPos[i],yPos[i+1]], color=colormap(c),
      #             marker='o', markersize=3.5, markerfacecolor='black')
      X.append([xPos[i],xPos[i+1]])
      Y.append([yPos[i],yPos[i+1]])
      C.append(colormap(c))

    return X,Y,C


def plotChar(train_data,train_indexes,train_inputs,char,index=None):

  np.random.seed(0)

  if not index:
    # random sample of manually selected character
    index = classes.index(char)
    indexes = np.where(np.array(train_data)==index)[0]
    index = np.random.choice(indexes)

  input, label = train_data[index]

  print(classes[int(label)])
  print('original sample')
  X,Y,C = toPlot_char(input)
  for x,y,c in zip(X,Y,C):
    plt.plot(x, y, color=c, marker='o', markersize=3.5, markerfacecolor='black')
  plt.show()

  indexes = np.where(train_indexes == index)[0]
  n_patches = len(indexes)
  print(n_patches,'patches')
  cols = int(np.ceil(n_patches/2))
  fig, axs = plt.subplots(2,cols, figsize=(15,6))

  for i,index in enumerate(indexes):
    input = train_inputs[index]
    indicator = input[3]
    indicator = np.where(indicator==1)[0]
    input = [np.take(channel,indicator) for channel in input[:3]]

    X,Y,C = toPlot_char(input)
    for x,y,c in zip(X,Y,C):
      axs[0 if i<cols else 1,i%cols].plot(x, y, color=c, marker='o', markersize=3.5, markerfacecolor='black')
