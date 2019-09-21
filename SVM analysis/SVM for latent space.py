import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import itertools
import pickle
from sklearn.metrics import confusion_matrix
 
def plot_confusion_matrix(cm, name, classes, min, max, dataset, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	def ROC(cm):
		TN = cm[0][0]
		FN = cm[1][0]
		TP = cm[1][1]
		FP = cm[0][1]
		return np.array([[TP,FP],[FN,TN]]), round(TP/(TP+FN),4), round(FP/(FP+TN),4), round(TP/(TP+FP),4), round(FN/(FN+TP),4)

	new_confusion_matrix, TPR, FPR, PPV, FNR = ROC(cm)
	
	plt.figure(figsize=(10,10))	
	plt.imshow(new_confusion_matrix, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.clim(min,max)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' #if normalize else 'd'
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
	plt.show()
	plt.tight_layout()
	plt.savefig(name+dataset+'.png')
def print_tp_tn(output, cm):
	tn = cm[0][0]
	fn = cm[1][0]
	tp = cm[1][1]
	fp = cm[0][1]
	print("\t \t | failure | healthy |")
	print("predicted failure | " + str(tp) + "\t|\t" + str(fp) + "|")
	print("predicted healthy | " + str(fn) + "\t|\t" + str(tn) + "|")
	print()

	output.write("\t \t | failure | healthy |\n")
	output.write("predicted failure | " + str(tp) + "\t|\t" + str(fp) + "|\n")
	output.write("predicted healthy | " + str(fn) + "\t|\t" + str(tn) + "|\n")

	return output
def print_metrics(output, train_predictions, train_labels, name, type="training"):
	train_cm = confusion_matrix(train_labels, train_predictions)
	plot_confusion_matrix(train_cm, name,  ['failed', 'healthy'], 0, train_labels.shape[0], type)	
	output.write(type+'\n')
	output = print_tp_tn(output, train_cm)
	output.write("-------------------------------------------------------------")
	output.write('\n')

######################################################################################
# SVM
######################################################################################
X = np.loadtxt('Latent_dataspace.csv', delimiter=',', dtype=float)
X_test = np.loadtxt('Latent_dataspace-test.csv', delimiter=',',dtype=float)
Y = np.loadtxt('Y_train_file.csv', delimiter=',', dtype=float)
Y_test = np.loadtxt('Y_test_file.csv', delimiter=',', dtype=float)

print()
print("SVM classifier")
print()
with open('SVM_results_latent.txt', 'w') as output:
    output.write('\n')
    output.write("===== SVM CLASSIFIER =====\n")
    output.write('\n')
    
    SVM_clf = svm.SVC(C=1,class_weight={0:1, 1:100},kernel='rbf',decision_function_shape='ovr', random_state=0)
    SVM_clf.fit(X, Y)
    
    with open('svm_ovr_str.pickle','wb') as f:
    	pickle.dump(SVM_clf,f)
    
    # predict
    train_predictions = SVM_clf.predict(X)
    test_predictions = SVM_clf.predict(X_test)
    
    print_metrics(output, train_predictions, Y, 'SVM Classifier')
    np.savetxt('SVM Classifier - training pred.csv', train_predictions, delimiter=',', comments='', fmt='%f')
    print_metrics(output, test_predictions, Y_test, 'SVM Classifier', type='testing')
    np.savetxt('SVM Classifier - test pred.csv', test_predictions, delimiter=',', comments='', fmt='%f')