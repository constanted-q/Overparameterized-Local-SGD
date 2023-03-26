#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 23:19:18 2022

@author: qintiancheng
"""
#from tqdm import tqdm
import matplotlib.pyplot as plt
from resnet import ResNet50,ResNet18
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
from GoogleNet import GoogLeNet


from models import MLP, CNNMnist
from getdata import get_dataset
from train import global_train

dataset = 'cifar10'
label_per_user = 1
num_user = 20
model = 'CNN'
local_batch_size = 16
T = 10e+4

train_dataset, test_dataset,dict_users = get_dataset(dataset,num_user,label_per_user)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=local_batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=local_batch_size,
                                          shuffle=False)

if model =='MLP':
    dim_in = 28*28
    dim_hidden = 200
    dim_out = 10
    global_model = MLP(dim_in, dim_hidden,dim_out)
    test_thrs = 0.97
elif model =='CNN':
    num_channels = 3
    num_classes = 10
    global_model = CNNMnist(num_channels,num_classes)
    test_thrs = 0.985
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


local_steps_list = np.array([10,2,5,10,20,50,100])
#lr_list = np.array([0.01,0.2,0.2,0.15,0.1,0.1,0.075])
lr_list = [0.1,0.1,0.1]
p_list = [0.5,0.5,0.8]
criterion = torch.nn.CrossEntropyLoss()
communication_rounds = 20000
loss_list_1 = []
rounds_list_1 = []
for i in range(1):
    for lr in lr_list:
        print(f'lr={lr}')
        p = p_list[0]
        #global_model = CNNMnist(num_channels,num_classes)
        #global_model = torchvision.models.resnet50(pretrained=False)
        global_model = ResNet18()
        #global_model = GoogLeNet()
        global_model.train()
        print(sum(p.numel() for p in global_model.parameters() if p.requires_grad))
        local_steps = local_steps_list[i]
        #lr = lr_list[i]
        loss_list,rounds = global_train(global_model,train_dataset,test_dataset,dict_users,num_user,
                     lr,local_steps,communication_rounds,criterion,test_thrs,local_batch_size,device,p)
        loss_list_1.append(loss_list)
        rounds_list_1.append(rounds)

np.save('local_step_10_repeat',loss_list_1)
'''
plt.figure()
plt.title('Local step = 20')
plt.plot(range(0,20*len(loss_list_1[0]),20), loss_list_1[0], color='r',label = 'lr = 1')
plt.plot(range(0,20*len(loss_list_1[1]),20), loss_list_1[1], color='g',label = 'lr = 0.3')
plt.plot(range(0,20*len(loss_list_1[2]),20), loss_list_1[2], color='y',label = 'lr = 0.1')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds: R')
plt.ylim(0, 2)
plt.legend()
plt.show()'''

#np.save('local_step_1',loss_list_1)






















