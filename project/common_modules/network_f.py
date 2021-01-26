import numpy as np

# to plot the data
import matplotlib.pyplot as plt

# for neural network
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# to save parameters
import copy

# for evaluating results
from sklearn.metrics import classification_report

# to track progress
from tqdm.notebook import tqdm



# train network
def trainNet(net,criterion,optimizer,scheduler,train_loader,val_loader,epochs,print_every=None,earlyStopping=None,approach="defailt"):

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