# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d, ReLU, Sequential
import torchvision.transforms as transforms
import os

transform = transforms.Compose([
    transforms.PILToTensor()
])

x, y = np.array([]), np.array([])

for filename in os.listdir('garbage_classification/glass')[:10]:
    img = Image.open('garbage_classification/glass/'+filename)
    img = transform(img)
for filename in os.listdir('garbage_classification/cardboard')[:10]:
    img = Image.open('garbage_classification/cardboard/'+filename)
    img = transform(img)
for filename in os.listdir('garbage_classification/metal')[:10]:
    img = Image.open('garbage_classification/metal/'+filename)
    img = transform(img)
for filename in os.listdir('garbage_classification/plastic')[:10]:
    img = Image.open('garbage_classification/plastic/'+filename)
    img = transform(img)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
