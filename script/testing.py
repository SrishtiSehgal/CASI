#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:56:36 2019

@author: srishtisehgal
"""

#CASI Architecture

###############################################################

#IMPORTS
import os, time, math, torch
import torch.nn as nn
import torch.utils.data as data_utils
import SparseAutoEncoder as SAE
import CNN 
import visdom_tutorial as utils
import numpy as np
from itertools import chain
#import EarlyStoppingCriteria as ESC
from sklearn.metrics import confusion_matrix

###############################################################

#RANDOM SEED SETTING
torch.manual_seed(0)
np.random.seed(0)

###############################################################
class test2(nn.Module):
	def __init__(self):
		super(test2, self).__init__()
		self.fc1 = nn.Linear(3,2, bias= False)
		self.fc2 = nn.Linear(2,1, bias= False)
	def forward(self, x):
		y = self.fc1(x)
#		print(y.size())
		y = self.fc2(y)
#		print(y.size())
		return y

class test1(nn.Module):
	def __init__(self):
		super(test1, self).__init__()
		self.fc1 = nn.Linear(4,3, bias= False)
		self.fc2 = nn.Linear(3,1, bias= False)
	def forward(self, x):
		y = self.fc1(x)
#		print(y.size())
		z = self.fc2(y)
#		print(y.size())
		return y, z

def model_training(net1, net2, X, Y1, Y2):
	#INITIALIZE LOSS AND OPTIMIZERS
	net1_loss, net2_loss = nn.MSELoss(), nn.MSELoss()
	optimizer_AE = torch.optim.Adam(	chain(net1.parameters(),net2.parameters()), lr=0.01)
	net1.train()
	net2.train()
	print('BEFORE TRAINING')
	print('net1', list(net1.parameters()),'\n')
	print('net1 grad', [x.grad for x in list(net1.parameters())],'\n')
	print('net2', list(net2.parameters()),'\n')
	print('net2 grad', [x.grad for x in list(net2.parameters())],'\n')
	#check direction gradients are updated, how shared layers are updated    
	optimizer_AE.zero_grad()
	latent, outputs = net1(X)
	class_pred = net2(latent)

	#CALCULATE LOSSES
	mse_loss = net1_loss(outputs, Y1)
	bce_loss = net2_loss(class_pred, Y2)
	loss =  bce_loss #+mse_loss
				
	#UPDATE
	print('LOSS UPDATED')
	loss.backward()
	print('net1 grad', [x.grad for x in list(net1.parameters())],'\n')
	print('net2 grad', [x.grad for x in list(net2.parameters())])
#	print('GRADIENTS APPLIED')
	optimizer_AE.step()
#	print('net1', list(net1.parameters()))
#	print('net1 grad', [x.grad for x in list(net1.parameters())])
#	print('net2', list(net2.parameters()))
#	print('net2 grad', [x.grad for x in list(net2.parameters())])
	
################################################################
	
net1 = test1()
net2 = test2()
	
################################################################
	
X = torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]])
Y1 = torch.tensor([[100.],[150.]])
Y2 = torch.tensor([[20.],[40.]])

################################################################

model_training(net1, net2, X, Y1, Y2)