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
def trainNet(net,criterion,optimizer,train_loader,val_loader,epochs,print_every=None):

    if not print_every:
        print_every = int(epochs / 10)

    avg_trainLosses = []
    avg_valLosses = []

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        train_loss = []
        val_loss = []

        net.train()
        for i, (inputBatch,labelBatch) in enumerate(train_loader):

            inputBatch, labelBatch = inputBatch.to(device), labelBatch.to(device)
            inputBatch = inputBatch.float()

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

            inputBatch, labelBatch = inputBatch.to(device), labelBatch.to(device)
            inputBatch = inputBatch.float()

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
        avg_valLosses.append(avg_valLoss)

        # print statistics
        if epoch % print_every == print_every - 1:
          print('epoch: %d, train loss: %.3f, val loss: %.3f' % (epoch + 1, avg_trainLoss, avg_valLoss))

    print('Finished Training')
    plt.plot(avg_trainLosses, label='train loss')
    plt.plot(avg_valLosses, label='val loss')
    plt.plot([best_loss]*epochs, linestyle='dashed')
    plt.plot(best_epoch, best_loss, 'o')
    plt.legend()

    return best_params



# evaluate and print
def evaluate(net,data_loader,classes=None):
  y_true= []
  y_pred = []
  net.eval()

  for _, (inputBatch,labelBatch) in enumerate(tqdm(data_loader)):
    with torch.no_grad():
      inputBatch, labelBatch = inputBatch.to(device), labelBatch.to(device)
      inputBatch = inputBatch.float()
      outputBatch = net(inputBatch)

      for output,label in zip(outputBatch,labelBatch):
        output, label = output.cpu(), label.cpu()
        y_true.append(label)
        pred = np.argmax(output)
        y_pred.append(pred)

  if classes is not None:
    print(classification_report(y_true, y_pred, target_names=classes, labels=range(len(classes)) ,digits=4))
  else:
    print(classification_report(y_true, y_pred, digits=4))


def getLatentFeatures(net, inputs):
  latent_inputs = []
  for input1 in tqdm(inputs):
    lat_featues = net.latent(torch.from_numpy(input1).unsqueeze(0).float())
    lat_featues = lat_featues.detach().numpy()
    latent_inputs.append(lat_featues)

  return np.array(latent_inputs)    