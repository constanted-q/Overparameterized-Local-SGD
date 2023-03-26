#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden_1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden_2 = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        #x = F.dropout(x, training=self.training)
        x = self.relu(x)
        x = self.layer_hidden_1(x)
        #x = F.dropout(x, training=self.training)
        x = self.relu(x)
        x = self.layer_hidden_2(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, num_channels,num_classes):
        super(CNNMnist, self).__init__()
        #self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5,padding=2)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(3136, 512)
        #self.fc1 = nn.Linear(1600, 512)
        self.fc1 = nn.Linear(36864, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

