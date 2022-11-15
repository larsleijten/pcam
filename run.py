import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import shutil
import ttach as tta
import random
import ConvNet

from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from pathlib import Path
from torchmetrics.classification import BinaryAUROC
from train import Experiment, train_epoch, fit_convnet, fit_ft_resnet, fit_pt_resnet, fit_resnet
from data import load_data
from test import validation, tta_test

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define and initialize the different models

# Define a loss function (general across all models)
loss_fn = nn.BCEWithLogitsLoss()

# Model 1: A small, custom Convolutional Neural Network
conv_model = ConvNet().to(device)

# Optimize the parameters with a learning rate of 1e-3 and default values of beta
conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr=1e-4, weight_decay=1e-5)

# Model 2: Fine tuning the FC-layers of a ResNet50 pretrained on ImageNet
ft_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
# Freeze the parameters
for param in ft_resnet50.parameters():
  param.requires_grad = False

# Add FC layers we can finetune
ft_resnet50.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128,1)
).to(device)
# Define optimizer on the finetuning layers
ft_resnet50_optimizer = torch.optim.Adam(ft_resnet50.fc.parameters(), lr = 1e-3, weight_decay=1e-5)

# Model 3: Retraining a ResNet50 pretrained on ImageNet
pt_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
# Add FC layers to fit our binary classification problem
pt_resnet50.fc = nn.Sequential(
    # Dropout layers are included as a regularization method
    nn.Dropout(p=0.6),
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(128,1)
).to(device)
# Define optimizer on all layers
pt_resnet50_optimizer = torch.optim.Adam(pt_resnet50.parameters(), lr = 1e-4, weight_decay = 1e-5)

# Model 4: Training a ResNet50 with randomly initialized weights
resnet50 = models.resnet50(weights=None).to(device)
# Add FC layers to fit our binary classification problem
resnet50.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128,1)
).to(device)
# Define optimizer
resnet50_optimizer = torch.optim.Adam(resnet50.parameters(), lr = 1e-3, weight_decay=1e-5)

# Load data 
train_data_loader, validation_data_loader, test_data_loader = load_data()

# Define if the training process should restart or use the latest saved model. 
restart = True
epochs = 20

# For reproducability
random.seed(128)

# Create and run experiment
experiment = Experiment(epochs, restart)
experiment.run(train_data_loader, conv_model, ft_resnet50, pt_resnet50, resnet50, loss_fn, conv_optimizer, ft_resnet50_optimizer, pt_resnet50_optimizer, resnet50_optimizer, device, validation_data_loader)

# Load the parameters from the best performing model
load_model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_best_epoch"
pt_resnet50.load_state_dict(torch.load(load_model_path))

# Run inference with Test Time Augmentation
test_predictions = tta_test(test_data_loader, pt_resnet50, loss_fn, device)

# Save the predictions for submission to the Grand Challenge leaderboard
test_csv = pd.DataFrame(columns=["case", "prediction"])
for i in range(0,32768):
  test_csv.loc[i, 'case',] = str(i)
  test_csv.loc[i, 'prediction'] = f"{test_predictions[i]:>8f}"

test_csv.reset_index(drop=True)
test_csv.to_csv("/content/gdrive/My Drive/colab/results/test_submission.csv")