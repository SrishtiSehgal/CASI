import numpy as np
import matplotlib.pyplot as plt
import os, sys, csv
import pickle
from scipy import stats
#import easygui
from PIL import Image
#import tflearn
#import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import random
#from deap import base
#from deap import creator
#from deap import tools
import itertools
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# 0 corresponds to healthy, 1 to failed

#######################################################################################
# METRICS
#######################################################################################
#min=4613
#max=53710
def plot_confusion_matrix(classes=['failed','healthy'], min=0, max=53710, dataset='testing', name='confusion_matrix',normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	
	def ROC():
		TN = 53649#cm[0][0]
		FN = 136#cm[1][0]
		TP = 4477#cm[1][1]
		FP = 43#cm[0][1]
		return np.array([[TP,FP],[FN,TN]]), round(TP/(TP+FN),4), round(FP/(FP+TN),4), round(TP/(TP+FP),4), round(FN/(FN+TP),4)

	new_confusion_matrix, TPR, FPR, PPV, FNR = ROC()
	
	fig = plt.figure(figsize=(10,10))	
	plt.imshow(new_confusion_matrix, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.clim(min,max)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = np.amax(new_confusion_matrix) / 2.
	for i, j in itertools.product(range(new_confusion_matrix.shape[0]), range(new_confusion_matrix.shape[1])):
		plt.text(j, i, format(new_confusion_matrix[i, j], fmt),
				 horizontalalignment="center", 
				 color="white" if new_confusion_matrix[i, j] > thresh else "black")

	plt.xlabel('True label')
	plt.ylabel('Predicted label')
	table = plt.table(cellText=[[TPR, FPR, PPV, FNR]], cellLoc='center', colLabels=['TPR','FPR','PPV','FNR'], loc='bottom', bbox=(0,-0.2,1,0.08))
	#bbox: The first coordinate is a shift on the x-axis, 
	#second coordinate is a gap between plot and text box (table in your case), 
	#third coordinate is a width of the text box, fourth coordinate is a height of text box
	table.auto_set_font_size(False)
	table.set_fontsize(12)
	# plt.show()
	plt.tight_layout()
	plt.savefig(name+dataset+'.png')

	return output

if __name__ == "__main__":
	plot_confusion_matrix()