#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:16:29 2022

@author: qintiancheng
"""
import copy
import random

import torch
import time
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def global_train(global_model,train_dataset,test_dataset,dict_users,num_user,
                 lr,local_steps,communication_rounds,criterion,test_thrs,local_batch_size,device,p):
    S = max(1,int(p*num_user))
    agent_list = list(range(num_user))
    random.shuffle(agent_list)
    global_model.to(device)
    rounds = communication_rounds
    loss_list = []
    start_time = time.time()
    eval_batch = 500
    evalloader = DataLoader(train_dataset, batch_size=eval_batch,shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=500,shuffle=False)
    for i in range(int(communication_rounds)):
        local_weights = []
        for s in range(S):
        #for j in range(int(num_user)):
            j = agent_list[s]
            local_dataset = DatasetSplit(train_dataset,dict_users[j])
            local_trainloader = DataLoader(local_dataset, batch_size=local_batch_size,shuffle=True)
            local_model = copy.deepcopy(global_model)
            local_weights.append(local_round(local_model,local_trainloader,lr,local_steps,criterion,device))
            del local_model
        for j in range(num_user-S):
            local_model = copy.deepcopy(global_model)
            local_weights.append(local_model.state_dict())
            del local_model
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        if (i+1) % 20 ==0:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=lr)
            inputs, labels = next(iter(evalloader))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #with torch.no_grad():
            outputs = global_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            gradient_norm = gra_norm(global_model)
            parameter_norm = weight_norm(global_model)
            loss_list.append(loss.detach().cpu())
            print(f'Round: [{i + 1:5d}] Loss: {loss:.5f}')
            print(f'Round: [{i + 1:5d}] GradNorm: {gradient_norm:.5f}')
            print(f'Round: [{i + 1:5d}] WeightNorm: {parameter_norm:.5f}')
        if (i+1) % 100 ==0:
            global_model.eval()
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
            print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
            if correct / total >test_thrs:
                rounds = i+1
                break
    return loss_list,rounds
        
        

def local_round(local_model,dataloader,lr,local_steps,criterion,device):
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    for i in range(int(local_steps)):
        inputs, labels = next(iter(dataloader))
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return local_model.state_dict()

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def gra_norm(w):
    total_norm = 0
    for p in w.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def weight_norm(w):
    total_norm = 0
    for p in w.parameters():
        if p.grad is not None:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm