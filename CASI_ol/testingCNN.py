#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:56:22 2019

@author: srishtisehgal
"""
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

class Net(nn.Module):
    def __init__(self, channels, C_out, kernel_conv, 
                 stride_conv, padding_conv,
                 kernel_pool, stride_pool, 
                 padding_pool, fc_in, fc_out):
        
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(channels[0], C_out[0], 
                               kernel_size = kernel_conv[0], 
                               stride=stride_conv[0], 
                               padding = padding_conv[0], 
                               bias = False)
        
        self.pool = nn.MaxPool2d(kernel_pool, 
                                 stride = stride_pool, 
                                 padding = padding_pool)
        
        self.conv2 = nn.Conv2d(channels[1], C_out[1], 
                               kernel_size = kernel_conv[1],
                               stride=stride_conv[1], 
                               padding = padding_conv[1],
                               bias = False)
        
        self.fc1 = nn.Linear(fc_in[0], fc_out[0], bias = False)
        self.fc2 = nn.Linear(fc_in[1], fc_out[1], bias = False)
        self.fc3 = nn.Linear(fc_in[2], fc_out[2], bias = False)
        self.fc4 = nn.Linear(fc_in[3], fc_out[3], bias = False)
        
    def forward(self,x,y):
        print(x.size())
        
        out = F.relu(self.conv1(x))
        print(out.size())
        
        out = self.pool(out)
        print(out.size())

        out = F.relu(self.conv2(out))
        print(out.size())

        out = self.pool(out)
        print(out.size())

        out = out.view(-1, 15*7*7)
        print(out.size())

        out = F.relu(self.fc1(torch.cat((out,y),1)))
        print(out.size())

        out = F.relu(self.fc2(out))
        print(out.size())

        out = F.relu(self.fc3(out))
        print(out.size())

        out = self.fc4(out)
        print(out.size())

        return out
    
    def create_datasets(channelsX, HX, WX, WY):
        x = torch.randn((channelsX, HX, WX))
        y = torch.randn((1,WY))
        target = torch.tensor([10.3])
        return x, y, target

inputsizes = (3,64,64,15)
epochs = 20
channels=[inputsizes[0],10]
C_out = [channels[-1],15]
kernel_conv=[3, 5]
stride_conv = [1, 2]
padding_conv = [0, 0]
kernel_pool = 4
stride_pool = 2
padding_pool = 1
fc_in = [C_out[1]*7*7+inputsizes[-1], 500, 200, 20]
fc_out = [fc_in[1], fc_in[2], fc_in[3], 1]

net = Net(channels, C_out, kernel_conv, stride_conv, 
          padding_conv, kernel_pool, stride_pool, 
          padding_pool, fc_in, fc_out)

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

first_input, second_input, target = Net.create_datasets(*inputsizes)
np.savetxt('first_input0.csv', first_input.numpy()[0,:,:], delimiter=',', fmt='%f', comments='')
np.savetxt('first_input1.csv', first_input.numpy()[1,:,:], delimiter=',', fmt='%f', comments='')
np.savetxt('first_input2.csv', first_input.numpy()[2,:,:], delimiter=',', fmt='%f', comments='')
np.savetxt('second_input.csv', second_input.numpy(), delimiter=',', fmt='%f', comments='')
np.savetxt('target.csv', target.numpy(), delimiter=',', fmt='%f', comments='')
first_input = first_input.view(-1, *tuple(first_input.size()))
    
for i in range(1):
    
    optimizer.zero_grad() #zero gradient buffers
#    print(list(net.parameters()))
    outputs = net(first_input,second_input)
#    print('output @ epoch: ', str(i), outputs, '\n\n')
    loss = criterion(outputs, target)
    loss.backward()
#    print([x.grad for x in list(net.parameters())][-1])
#    print(list(net.fc4.weight))
    optimizer.step()