#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:33:02 2019

@author: srishtisehgal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import CASI
#AutoEncoder Class
class AutoEncoder(nn.Module):
    def __init__(self, fc_in, fc_out, sparsity):
        super(AutoEncoder, self).__init__()
        #Encoding Layers
        self.sparsity = sparsity
        self.fce1 = nn.Linear(fc_in[0], fc_out[0], bias = False)
        #Decoding Layers
        self.fcd1 = nn.Linear(fc_out[0], fc_in[1], bias = False)
        self.fcd2 = nn.Linear(fc_in[1], fc_out[1], bias = False)
        
    def initialize(self):
        nn.init.xavier_uniform(self.linear.weight.data)
        self.linear.bias.data.zero_()
        
    def forward(self,x):
        kl= torch.tensor([0.])
        def avg_activation(tensor):
            funcs = nn.Sigmoid()
            p_hat = torch.mean(funcs(tensor), 1)
            #row vector of the sparsity value
            p_tensor = p_hat.new_full(tuple(p_hat.size()), 
                                      self.sparsity, 
                                      requires_grad=False)
#            KL_div = F.kl_div(p_hat, p_tensor)
            KL_div = torch.sum(
                        p_tensor * torch.log(p_tensor) -
                        p_tensor * torch.log(p_hat) + 
                        (1 - p_tensor) * torch.log((1 - p_tensor)) -
                        (1 - p_tensor) * torch.log((1 - p_hat))
                        )
            return KL_div #applied in the encoding layer only

        out = F.relu(self.fce1(x))
        kl += avg_activation(out)
        latent = (self.fcd1(out))
        out = (self.fcd2(latent))        
        return kl, latent, out

class AutoEncoderOrig(nn.Module):
    def __init__(self, fc_in, fc_out, DISTRIBUTION_VAL, filename):    
        super(AutoEncoderOrig, self).__init__()
        #Parameters
        self.sparsity = DISTRIBUTION_VAL
        self.file = filename
        #Encoding Layers
        self.fce1 = nn.Linear(fc_in[0], fc_out[0], bias = False) #51-60
        self.fce2 = nn.Linear(fc_in[1], fc_out[1], bias = False) #60-40
        self.fce3 = nn.Linear(fc_in[2], fc_out[2], bias = False) #40-30
        self.fce4 = nn.Linear(fc_in[3], fc_out[3], bias = False) #30-20
        #Decoding Layers
        self.fcd6 = nn.Linear(fc_out[3], fc_out[4], bias = False)#20-10
        self.fcd5 = nn.Linear(fc_out[4], fc_out[3], bias = False)#10-20
        self.fcd4 = nn.Linear(fc_out[3], fc_in[3], bias = False)#20-30
        self.fcd3 = nn.Linear(fc_out[2], fc_in[2], bias = False)#30-40
        self.fcd2 = nn.Linear(fc_out[1], fc_in[0], bias = False)#40-51
#        self.fcd1 = nn.Linear(fc_out[0], fc_in[0], bias = False)
    def initialize(self):
        nn.init.xavier_uniform(self.linear.weight.data)
        self.linear.bias.data.zero_()
    def forward(self,x):
        kl= torch.tensor([0.])
        def avg_activation(tensor):
#            print(tensor)
            funcs = nn.Sigmoid()
            p_hat = torch.mean(funcs(tensor), 1)
#            print(p_hat)
            #row vector of the sparsity value
            p_tensor = p_hat.new_full(tuple(p_hat.size()), 
                                      self.sparsity, 
                                      requires_grad=False)
#            KL_div = F.kl_div(p_hat, p_tensor)
            KL_div = torch.sum(
                        p_tensor * torch.log(p_tensor) -
                        p_tensor * torch.log(p_hat) + 
                        (1 - p_tensor) * torch.log((1 - p_tensor)) -
                        (1 - p_tensor) * torch.log((1 - p_hat))
                        )
#            print('KL', KL_div)
            return KL_div #applied in the encoding layer only
    
        out = self.fce1(x)#linear
#        CASI.write_progress_to_file(self.file, 'AE-1st-lyr', str(out.size()))
#        print(out)
        kl += avg_activation(out)
#        print(kl)
        
        out = self.fce2(out)
#        CASI.write_progress_to_file(self.file, 'AE-2nd-lyr', str(out.size()))
        kl += avg_activation(out)
        
        out = F.relu(self.fce3(out))
#        CASI.write_progress_to_file(self.file, 'AE-3rd-lyr', str(out.size()))
        kl += avg_activation(out)
        
        latent_space = F.relu(self.fce4(out))
#        CASI.write_progress_to_file(self.file, 'AE-latent-lyr', str(latent_space.size()))    
        kl += avg_activation(latent_space)
        
        out = F.relu(self.fcd6(latent_space))
#        CASI.write_progress_to_file(self.file, 'AE-5th-lyr', str(out.size()))
        out = F.relu(self.fcd5(out))
        out = F.relu(self.fcd4(out))
        out = F.relu(self.fcd3(out))
#        CASI.write_progress_to_file(self.file, 'AE-6th-lyr', str(out.size()))

        out = self.fcd2(out)
#        CASI.write_progress_to_file(self.file, 'AE-7th-lyr', str(out.size()))
        
#        out = self.fcd1(out) #linear
#        CASI.write_progress_to_file(self.file, 'AE-8th-lyr', str(out.size()))
        
        return kl, latent_space, out