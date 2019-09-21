#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:41:19 2019

@author: srishtisehgal
"""

#Typical CNN network
#IMPORTS
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import CASI
#SET RANDOM SEED
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

#class Net(nn.Module):
#    def __init__(self, channels, C_out, kernel_conv, 
#                 stride_conv, padding_conv,
#                 fc_in, fc_out):
#        
#        super(Net, self).__init__()
#
#        self.conv1 = nn.Conv1d(channels[0], C_out[0], 
#                               kernel_size = kernel_conv[0], 
#                               stride=stride_conv[0], 
#                               padding = padding_conv[0], 
#                               bias = False)
#        
#        
#        self.conv2 = nn.Conv1d(channels[1], C_out[1], 
#                               kernel_size = kernel_conv[1],
#                               stride=stride_conv[1], 
#                               padding = padding_conv[1],
#                               bias = False)
#        self.fc1 = nn.Linear(fc_in[0], fc_out[0], bias = False)
#        self.fc2 = nn.Linear(fc_in[1], fc_out[1], bias = False)
#        self.fc3 = nn.Linear(fc_in[2], fc_out[2], bias = False)        
#        self.input_fc = fc_in[0]
#    def forward(self,x):
#        print(x.size())
#        out = F.relu(self.conv1(x))
#        print(out.size())
#        out = F.relu(self.conv2(out))
#        print(out.size())
#        out = out.view(-1, self.input_fc)
#        print(out)
#        out = F.relu(self.fc1(out))
#        out = F.leaky_relu(self.fc2(out))
#        out = self.fc3(out)
#        print(out)
#        return out
##CNN PARAMETERS
#samples = 300
#features = 51
#channels_in=[features,20]
#channels_out = [20,15]
#kernel_conv=[1,1]
#stride_conv = [1,1]
#padding_conv = [0,0]
#fc_in = [channels_out[-1],8,4]
#fc_out = [fc_in[1], fc_in[2], 1]
#
#import math
#def cal_shape(prev, ker, pad=0, dil=1, stride=1):
#    return math.floor(1 + ( (prev + 2*pad - dil*(ker - 1) - 1) / stride) )
#def create_datasets(HX, WX, channelsX=1):
#    x = torch.randn((channelsX, HX, WX))
#    target = torch.zeros(1, WX//4, dtype=torch.float)
#    target = torch.cat((target,torch.ones(1, WX-(WX//4))),1)
#    target = target.view(1, *target.size())
#    return x, target
#
#first_input, target = create_datasets(1, samples, channelsX=features)
#print(first_input.size(), target.size())
#
#net = Net(channels_in, channels_out, 
#          kernel_conv, stride_conv, padding_conv, 
#          fc_in, fc_out)
#
#import torch.optim as optim
#optimizer = optim.SGD(net.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss() #sigmoid+BCELoss in one!
#
#for i in range(1):
#    accumulated_loss = 0 #actual loss
#    optimizer.zero_grad() #zero gradient buffers
#    for i in range(300):
#        if i == 299:
#            print(first_input[:,:,i].view(1, *first_input[:,:,i].size()).size())
#            print(first_input[:,:,i].view(1, *first_input[:,:,i].size()))
#        outputs = net(first_input[:,:,i].view(1, *first_input[:,:,i].size()))
#        loss = criterion(outputs, target[:,:,i])
#        if i ==299:
#            print(outputs)
#            print(loss)
#        accumulated_loss +=loss
#    accumulated_loss.backward()
#    print(accumulated_loss)
#    optimizer.step()
    
#    print(list(net.parameters()))
#    print([x.grad for x in list(net.parameters())][-1])
#    print(list(net.fc4.weight))

class CNN(nn.Module):
    def __init__(self, channels, features ,
                 kernel_conv, stride_conv, padding_conv,
                 fc, filename):
        
        super(CNN, self).__init__()
        #1 input channel (Depth)
        #n_features as the steps (Width)
        #None as the height (batch size)
        self.file = filename
        self.conv1 = nn.Conv1d(channels[0], channels[1], 
                               kernel_size = kernel_conv[0], 
                               stride=stride_conv[0], 
                               padding = int((kernel_conv[0] - 1)/2), 
                               bias = False)
        self.conv2 = nn.Conv1d(channels[1], int(channels[2]/2), 
                               kernel_size = int(kernel_conv[0]/2),
                               stride=stride_conv[1], 
                               padding = int((int(kernel_conv[0]/2) - 1)/2),
                               bias = False)
        self.conv3 = nn.Conv1d(int(channels[2]/2), int(channels[3]/2), 
                               kernel_size = int(kernel_conv[0]/2),
                               stride=stride_conv[1], 
                               padding = int((int(kernel_conv[0]/2) - 1)/2),
                               bias = False)
        
        self.fc1 = nn.Linear(int(channels[3]/2)*features, fc[0], bias = False)
        self.fc2 = nn.Linear(fc[0], fc[1], bias = False)
#        self.fc1 = nn.Linear(int(channels[3]/2)*features, fc[0], bias = False)
#        self.fc2 = nn.Linear(fc[0], fc[1], bias = False)
  
    def forward(self,x):
        x=x.view(x.size()[0],1,x.size()[1])
        CASI.write_progress_to_file(self.file, 'CNN-new-input-size', str(x.size()))
        
        out = F.relu(self.conv1(x))
        CASI.write_progress_to_file(self.file, 'CNN-1st-lyr', str(out.size()))
        
        out = F.relu(self.conv2(out))
        CASI.write_progress_to_file(self.file, 'CNN-2nd-lyr', str(out.size()))
        
        out = F.relu(self.conv3(out))
        CASI.write_progress_to_file(self.file, 'CNN-3rd-lyr', str(out.size()))
        
#        out = out.view(-1, out.size()[0]*out.size()[1]*out.size()[2])
        out = out.view(-1, out.size()[1]*out.size()[2])
        CASI.write_progress_to_file(self.file, 'CNN-new_size', str(out.size()))
        
        out = F.relu(self.fc1(out))
        CASI.write_progress_to_file(self.file, 'CNN-5th-lyr', str(out.size()))
        
        out = self.fc2(out)
        CASI.write_progress_to_file(self.file, 'CNN-6th-lyr', str(out.size()))
        
#        print(out)
        return out

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.fc1 = nn.Linear(20,1, bias= False)
    def forward(self, x):
        y = self.fc1(x)
        print(y.size())
        return y

#CNN PARAMETERS
#samples = 300
#features = 30
#channels=[1,20,14,10]
#kernel_conv=[23]
#stride_conv = [1,1]
#padding_conv = [0,0]
#fc= [6, 1]
#
#import math
#def cal_shape(prev, ker, pad=0, dil=1, stride=1):
#    return math.floor(1 + ( (prev + 2*pad - dil*(ker - 1) - 1) / stride) )
#def create_datasets(HX, WX, channelsX=1):
#    x = torch.randn((channelsX, HX, WX))
#    target = torch.zeros(channelsX//4,1, dtype=torch.float)
#    target = torch.cat((target,torch.ones(channelsX-(channelsX//4),1)),0)
##    target = target.view(*target.size(), 1)
#    return x, target
#
#first_input, target = create_datasets(1,features, channelsX=samples)
#net = CNN(channels, features,
#          kernel_conv, stride_conv, padding_conv, 
#          fc)
#
#import torch.optim as optim
#optimizer = optim.SGD(net.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss() #sigmoid+BCELoss in one!
#
#for i in range(1):
#    optimizer.zero_grad() #zero gradient buffers
#    outputs = net(first_input) #supply input like this torch.Size([300, 1, 30])
#    loss = criterion(outputs, target)#target should be like thistorch.Size([300, 1])
#    print('loss',loss)
#    loss.backward()
#    optimizer.step()
##    for i,inp in enumerate(first_input):
##        outputs = net(inp.view(1,*inp.size()))
##        loss = criterion(outputs, target[i])
##        print('loss',loss)
##        loss.backward()
##        optimizer.step()