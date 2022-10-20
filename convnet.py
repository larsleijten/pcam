import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    # Build a small Convolutional Neural Network
    self.conv = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(4,4),

        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(4,4),

        nn.Flatten(),
        nn.Linear(2304,1024),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,1)
    )
  
  def forward(self, x):
    logit = self.conv(x)
    #p = nn.Sigmoid(logit)
    return logit


class Experiment():
    def __init__(self, epochs, restart):
        self.num_epochs = epochs
        self.restart = restart
        if self.restart:
            self.train_loss = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.validation_loss = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.train_accuracy = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.validation_accuracy = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
        else:
            self.train_loss = pd.read_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
            self.validation.loss = pd.read_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
            self.train_accuracy = pd.read_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
            self.validation_accuracy = pd.read_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

    def run(self, train_data_loader, conv_model, ft_resnet50, pt_resnet50, resnet50, loss_fn, conv_optimizer, ft_resnet50_optimizer, pt_resnet50_optimizer, resnet50_optimizer, device, validation_data_loader):
        fit_convnet(train_data_loader, conv_model, loss_fn, conv_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_ft_resnet(train_data_loader, ft_resnet50, loss_fn, ft_resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_pt_resnet(train_data_loader, pt_resnet50, loss_fn, pt_resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_resnet(train_data_loader, resnet50, loss_fn, resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)


def fit_convnet(train_data_loader, model, loss_fn, optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    # Load the model from the latest trained epoch  
    if restart:
        with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt", 'w') as f:
            f.write(str(0))
  
    with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_conv_epoch_" + str(start_epoch)
        model.load_state_dict(torch.load(load_model_path))

    # Train up untill the required epoch
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_accuracy = train_epoch(train_data_loader, model, loss_fn, optimizer, device)    
        model_path = "/content/gdrive/My Drive/colab/models/pcam_conv_epoch_" + str(t+1)
        torch.save(model.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt", 'w') as f:
            f.write(str(t+1))
    
        v_loss, v_accuracy = validation(validation_data_loader, model, loss_fn, device)
        train_loss['conv_model'][t] = t_loss
        validation_loss['conv_model'][t] = v_loss
        train_accuracy['conv_model'][t] = t_accuracy
        validation_accuracy['conv_model'][t] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")


def fit_ft_resnet(train_data_loader, model, loss_fn, optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    if restart:
        with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(0))
  
    with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_epoch_" + str(start_epoch)
        model.load_state_dict(torch.load(load_model_path))

    # Train up untill the required epoch
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_accuracy = train_epoch(train_data_loader, model, loss_fn, optimizer, device)    
        model_path = "/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_epoch_" + str(t+1)
        torch.save(model.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        v_loss, v_accuracy = validation(validation_data_loader, model, loss_fn, device)
        train_loss['ft_resnet'][t] = t_loss
        validation_loss['ft_resnet'][t] = v_loss
        train_accuracy['ft_resnet'][t] = t_accuracy
        validation_accuracy['ft_resnet'][t] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")


def fit_pt_resnet(train_data_loader, model, loss_fn, optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    if restart:
        with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(0))
  
    with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_epoch_" + str(start_epoch)
        model.load_state_dict(torch.load(load_model_path))

    # Train up untill the required epoch
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_accuracy = train_epoch(train_data_loader, model, loss_fn, optimizer, device)    
        model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_epoch_" + str(t+1)
        torch.save(model.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        v_loss, v_accuracy = validation(validation_data_loader, model, loss_fn, device)
        train_loss['pt_resnet'][t] = t_loss
        validation_loss['pt_resnet'][t] = v_loss
        train_accuracy['pt_resnet'][t] = t_accuracy
        validation_accuracy['pt_resnet'][t] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

def fit_resnet(train_data_loader, model, loss_fn, optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    if restart:
        with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(0))
  
    with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_resnet50_epoch_" + str(start_epoch)
        model.load_state_dict(torch.load(load_model_path))

    # Train up untill the required epoch
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_accuracy = train_epoch(train_data_loader, model, loss_fn, optimizer, device)    
        model_path = "/content/gdrive/My Drive/colab/models/pcam_resnet50_epoch_" + str(t+1)
        torch.save(model.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        v_loss, v_accuracy = validation(validation_data_loader, model, loss_fn, device)
        train_loss['resnet'][t] = t_loss
        validation_loss['resnet'][t] = v_loss
        train_accuracy['resnet'][t] = t_accuracy
        validation_accuracy['resnet'][t] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss, total_correct = 0, 0    
    for batch, (X, y) in enumerate(dataloader):
        # Make sure the tensors are set to be processed by the correct device
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1).float())
        
        
        # Performance tracking
        m = torch.sigmoid(pred)
        bin_pred = torch.round(m).transpose(0,1)
        total_correct += (bin_pred == y).type(torch.float).sum().item()
        total_loss += loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    total_loss/=len(dataloader)
    total_correct/=size

    return total_loss, (total_correct*100)
            


def validation(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            m = torch.sigmoid(pred)
            bin_pred = torch.round(m).transpose(0,1)
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, (100*correct)


def test_convnet(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            m = torch.sigmoid(pred)
            bin_pred = torch.round(m).transpose(0,1)

            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")