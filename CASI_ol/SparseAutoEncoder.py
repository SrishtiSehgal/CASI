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
    def __init__(self, fc_in, fc_out, DISTRIBUTION_VAL, filename):    
        super(AutoEncoder, self).__init__()
        #Parameters
        self.sparsity = DISTRIBUTION_VAL
        self.file = filename
        #Encoding Layers
        self.fce1 = nn.Linear(fc_in[0], fc_out[0], bias = False)
        self.fce2 = nn.Linear(fc_in[1], fc_out[1], bias = False)
        self.fce3 = nn.Linear(fc_in[2], fc_out[2], bias = False)
        self.fce4 = nn.Linear(fc_in[3], fc_out[3], bias = False)
        #Decoding Layers
        self.fcd4 = nn.Linear(fc_out[3], fc_in[3], bias = False)
        self.fcd3 = nn.Linear(fc_out[2], fc_in[2], bias = False)
        self.fcd2 = nn.Linear(fc_out[1], fc_in[1], bias = False)
        self.fcd1 = nn.Linear(fc_out[0], fc_in[0], bias = False)
    
    def forward(self,x):
        def avg_activation(tensor):
            funcs = nn.Sigmoid()
            p_hat = torch.mean(funcs(tensor), 1)
            #row vector of the sparsity value
            p_tensor = p_hat.new_full(tuple(p_hat.size()), self.sparsity)
            KL_div = torch.sum(
                        p_tensor * torch.log(p_tensor/p_hat) + 
                        (1 - p_tensor) * torch.log((1 - p_tensor)/(1 - p_hat)))
            return KL_div #applied in the encoding layer only
        
        kl = torch.tensor([0.], requires_grad=False)
        kl_layer_loss = torch.tensor([0.], requires_grad=False)

        out = F.relu(self.fce1(x))
        CASI.write_progress_to_file(self.file, 'AE-1st-lyr', str(out.size()))
        with torch.no_grad(): 
            calculate_loss = out.detach()
            kl_layer_loss = avg_activation(calculate_loss)
            kl += kl_layer_loss
        
        out = F.relu(self.fce2(out))
        CASI.write_progress_to_file(self.file, 'AE-2nd-lyr', str(out.size()))
        with torch.no_grad(): 
            calculate_loss = out.detach()
            kl_layer_loss = avg_activation(calculate_loss)
            kl += kl_layer_loss
        
        out = F.relu(self.fce3(out))
        CASI.write_progress_to_file(self.file, 'AE-3rd-lyr', str(out.size()))
        with torch.no_grad(): 
            calculate_loss = out.detach()
            kl_layer_loss = avg_activation(calculate_loss)
            kl += kl_layer_loss
        
        latent_space = F.relu(self.fce4(out))
        CASI.write_progress_to_file(self.file, 'AE-latent-lyr', str(latent_space.size()))
        with torch.no_grad():     
            calculate_loss = latent_space.detach()
            kl_layer_loss = avg_activation(calculate_loss)
            kl += kl_layer_loss
            kl_layer_loss, calculate_loss = None, None
        
        out = F.relu(self.fcd4(latent_space))
        CASI.write_progress_to_file(self.file, 'AE-5th-lyr', str(out.size()))
        
        out = F.relu(self.fcd3(out))
        CASI.write_progress_to_file(self.file, 'AE-6th-lyr', str(out.size()))
        
        out = F.relu(self.fcd2(out))
        CASI.write_progress_to_file(self.file, 'AE-7th-lyr', str(out.size()))
        
        out = F.relu(self.fcd1(out))
        CASI.write_progress_to_file(self.file, 'AE-8th-lyr', str(out.size()))
        
        return kl, latent_space, out