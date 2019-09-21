#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:50:30 2019

@author: srishtisehgal
"""
import numpy as np
import matplotlib.pyplot as plt

class ColourData():
    def __init__(self, first_batch):
        self.reset()
        self.dataset = first_batch

    def reset(self):
        self.dataset = None

    def update(self, new_batch):
        self.dataset = np.concatenate((self.dataset, new_batch), axis = 0)

    def plot(self, maxrows, val_type, colour_map=plt.cm.Blues):
        mat = self.dataset[-maxrows:,:]
        fig, ax = plt.subplots(figsize=[10, 12])
        im = ax.imshow(mat, interpolation='nearest', cmap=colour_map)
        ax.figure.colorbar(im, ax=ax)
        ax.set(title='Data matrix visualization',
           ylabel='Rows',
           xlabel='Features')
        
#        thresh = mat.max() / 2.
#        for i in range(mat.shape[0]):
#            for j in range(mat.shape[1]):
#                ax.text(j, i, format(mat[i, j], '.2f'),
#                        ha="center", va="center",
#                        color="white" if mat[i, j] > thresh else "black")
        fig.tight_layout()
#        plt.show()
#        plt.axis('off')
#        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(val_type + '_Data_matrix_visualization.png')
        plt.savefig(val_type + '_Data_matrix_visualization.svg')
        plt.show()
        
some_batch = np.loadtxt('output_dataspace-test.csv', delimiter=',', dtype=float)
obj = ColourData(some_batch)
obj.plot(100, 'model')

whole_data = np.loadtxt('pseudonormalized_test_file.csv', delimiter=',', dtype=float)
obj2 = ColourData(whole_data)
obj2.plot(100, 'actual')

some_batch = np.loadtxt('output_dataspace.csv', delimiter=',', dtype=float)
obj = ColourData(some_batch)
obj.plot(100, 'tra-model')

whole_data = np.loadtxt('normalized_train_file.csv', delimiter=',', dtype=float)
obj2 = ColourData(whole_data)
obj2.plot(100, 'tra-actual')