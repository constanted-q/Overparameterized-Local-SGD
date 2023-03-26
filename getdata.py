#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:51:26 2022

@author: qintiancheng
"""

import numpy as np
from torchvision import datasets, transforms

def cifar10_noniid(labels, num_users,label_per_user):
    N = np.size(labels)
    num_imgs = int(N/(num_users*label_per_user))
    dict_users = {i: np.array([],int) for i in range(num_users)}
    idxs_shards = np.arange(num_users*label_per_user)
    np.random.shuffle(idxs_shards)
    idxs_labels = np.argsort(labels)
    for i in range(num_users):
        for j in range(label_per_user):
            idx_start = idxs_shards[i*label_per_user+j]*num_imgs
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_labels[idx_start:idx_start+num_imgs]), axis=0)
    return dict_users

def get_dataset(dataset,num_users,label_per_user):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'cifar10':
        #data_dir = '../data/mnist/'

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=False, transform=transform)

        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=False, transform=transform)
        labels = np.array(train_dataset.targets)
        dict_users = cifar10_noniid(labels, num_users,label_per_user)

    return train_dataset, test_dataset, dict_users
        

    
    