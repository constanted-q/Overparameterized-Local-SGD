#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:23:26 2022

@author: qintiancheng
"""

import matplotlib.pyplot as plt
import numpy as np

loss_list_1_2NN=np.load('/Users/qintiancheng/UIUC/12.npy',allow_pickle=True)
loss_list_2_2NN=np.load('/Users/qintiancheng/UIUC/13.npy',allow_pickle=True)
loss_list_10_2NN=np.load('/Users/qintiancheng/UIUC/14.npy',allow_pickle=True)

plt.figure()
plt.title('1/2 digits per node, 2NN')
plt.plot(range(0,len(loss_list_1_2NN[0])), loss_list_1_2NN[0], color='r',label = 'Local steps K= 1')
plt.plot(range(0,len(loss_list_1_2NN[1])), loss_list_1_2NN[1], color='g',label = 'Local steps K= 2')
plt.plot(range(0,len(loss_list_1_2NN[2])), loss_list_1_2NN[2], color='y',label = 'Local steps K= 5')
plt.plot(range(0,len(loss_list_1_2NN[3])), loss_list_1_2NN[3], color='b',label = 'Local steps K= 10')
plt.plot(range(0,len(loss_list_1_2NN[4])), loss_list_1_2NN[4], color='orange',label = 'Local steps K= 20')
plt.plot(range(0,len(loss_list_1_2NN[5])), loss_list_1_2NN[5], color='navy',label = 'Local steps K= 50')
plt.plot(range(0,len(loss_list_1_2NN[6])), loss_list_1_2NN[6], color='maroon',label = 'Local steps K= 50')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds: R')
plt.ylim(0, 2) 
plt.legend()
plt.show()

plt.figure()
plt.title('2/3 digits per node, 2NN')
plt.plot(range(0,len(loss_list_2_2NN[0])), loss_list_2_2NN[0], color='r',label = 'Local steps K= 1')
plt.plot(range(0,len(loss_list_2_2NN[1])), loss_list_2_2NN[1], color='g',label = 'Local steps K= 2')
plt.plot(range(0,len(loss_list_2_2NN[2])), loss_list_2_2NN[2], color='y',label = 'Local steps K= 5')
plt.plot(range(0,len(loss_list_2_2NN[3])), loss_list_2_2NN[3], color='b',label = 'Local steps K= 10')
plt.plot(range(0,len(loss_list_2_2NN[4])), loss_list_2_2NN[4], color='orange',label = 'Local steps K= 20')
plt.plot(range(0,len(loss_list_2_2NN[5])), loss_list_2_2NN[5], color='navy',label = 'Local steps K= 50')
plt.plot(range(0,len(loss_list_2_2NN[6])), loss_list_2_2NN[6], color='maroon',label = 'Local steps K= 50')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds: R')
plt.ylim(0, 2) 
plt.legend()
plt.show()

plt.figure()
plt.title('10 digits per node, 2NN')
plt.plot(range(0,len(loss_list_10_2NN[0])), loss_list_10_2NN[0], color='r',label = 'Local steps K= 1')
plt.plot(range(0,len(loss_list_10_2NN[1])), loss_list_10_2NN[1], color='g',label = 'Local steps K= 2')
plt.plot(range(0,len(loss_list_10_2NN[2])), loss_list_10_2NN[2], color='y',label = 'Local steps K= 5')
plt.plot(range(0,len(loss_list_10_2NN[3])), loss_list_10_2NN[3], color='b',label = 'Local steps K= 10')
plt.plot(range(0,len(loss_list_10_2NN[4])), loss_list_10_2NN[4], color='orange',label = 'Local steps K= 20')
plt.plot(range(0,len(loss_list_10_2NN[5])), loss_list_10_2NN[5], color='navy',label = 'Local steps K= 50')
plt.plot(range(0,len(loss_list_10_2NN[6])), loss_list_10_2NN[6], color='maroon',label = 'Local steps K= 50')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds: R')
plt.ylim(0, 2) 
plt.legend()
plt.show()