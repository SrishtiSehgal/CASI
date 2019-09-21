import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import itertools

#######################################################################################
# METRICS
#######################################################################################
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
	
	fig = plt.figure(figsize=(10,10))	
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
	# plt.show()
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
def print_metrics(output, train_predictions, train_labels, name):
	train_cm = confusion_matrix(train_labels, train_predictions)
	plot_confusion_matrix(train_cm, name,  ['defaulted', 'passed'], 0, train_labels.shape[0], 'training')	
	output.write('TRAINING\n')
	output = print_tp_tn(output, train_cm)
	output.write("-------------------------------------------------------------")
	output.write('\n')

def anomaly_detection_error(y_pred, y_true, dataset, output, name, OneClassSVMMethod=False):
	output.write('size of Target: ' +str(y_true.shape[0]))
	if OneClassSVMMethod:
		TP = np.nan
		FP = np.sum((y_true==0) & (y_pred==-1))
		TN = np.sum((y_true==0) & (y_pred!=-1))
		FN = np.nan
	else:
		TP = np.sum((y_true==1) & (y_pred==-1))
		FP = np.sum((y_true==0) & (y_pred==-1))
		TN = np.sum((y_true==0) & (y_pred!=-1))
		FN = np.sum((y_true==1) & (y_pred!=-1))
		output.write(dataset+ ' set: ' + str(TP) +' True positives\n')
		output.write(dataset+ ' set: ' + str(FP) +' False positives\n')
		output.write(dataset+ ' set: ' + str(TN) +' True negatives\n')
		output.write(dataset+ ' set: ' + str(FN) +' False negatives\n')
	print()
	print_tp_tn(output, [[TN,FP],[FN,TP]])
	data = np.array([[TN,FP],[FN,TP]])
	if dataset=='testing':
		plot_confusion_matrix(data, name,  ['failed', 'healthy'], 0, 58305, dataset)
	else:
		plot_confusion_matrix(data, name,  ['failed', 'healthy'], 0, 171836, dataset)

#######################################################################################
# CLASSIFICATION METHODS
#######################################################################################
def classification(X, Y, X_test, failed_size, IDs):
	input_dimensions = str(X.shape[1]) #feature length
	samples_size =str(X.shape[0]) #number of rows
	input_dimensions_test = str(X_test.shape[1] )#feature length
	samples_size_test = str(X_test.shape[0]) #number of rows

	with open('classification_results.txt', 'w') as output:
		output.write("===== DATA INFORMATION =====\n")
		output.write('training data size: ' +samples_size +' by '+ input_dimensions+'\n')
		output.write('test data size: '  +samples_size_test +' by '+ input_dimensions_test+'\n')
		output.write('failed points in training: ' +str(failed_size)+'\n')

		#######################################################################################
		# SVM
		#######################################################################################
#		print()
#		print("SVM classifier")
#		print()
#
#		output.write('\n')
#		output.write("===== SVM CLASSIFIER =====\n")
#		output.write('\n')
#
#		SVM_clf = svm.SVC(C=1,class_weight={0:1, 1:4},kernel='rbf',decision_function_shape='ovr', random_state=0)
#		SVM_clf.fit(X, Y)
#
#		with open('svm_ovr_str.pickle','wb') as f:
#			pickle.dump(SVM_clf,f)
#
#		# predict
#		train_predictions = SVM_clf.predict(X)
#		test_predictions = SVM_clf.predict(X_test)
#
#		print_metrics(output, train_predictions, Y, 'SVM Classifier')
#
#		np.savetxt('SVM Classifier - test pred.csv', test_predictions, delimiter=',', comments='', fmt='%f')
#
#		#######################################################################################
#		# Gaussian NAIVE BAYES
#		#######################################################################################
#		print()
#		print("GaussianNB classifier")
#		print()
#
#		output.write('\n')
#		output.write("===== GAUSSIAN NB CLASSIFIER =====\n")
#		output.write('\n')
#
#		GNB_clf = GaussianNB()
#		GNB_clf.fit(X, Y)
#
#		with open('GNB_clf.pickle','wb') as f:
#			pickle.dump(GNB_clf,f)
#
#		train_predictions = GNB_clf.predict(X)
#		test_predictions = GNB_clf.predict(X_test)
#		print_metrics(output, train_predictions, Y, 'GaussianNB Classifier')
#
#		np.savetxt('GaussianNB Classifier - test pred.csv', test_predictions, delimiter=',', comments='', fmt='%f')

		#######################################################################################
		# RANDOM FOREST
		#######################################################################################
		print()
		print("RandomForest classifier")
		print()

		output.write('\n')
		output.write("===== RANDOM FOREST CLASSIFIER =====\n")
		output.write('\n')

		RF_clf = RandomForestClassifier(random_state=43)
		RF_clf.fit(X, Y)

		with open('RF_clf.pickle','wb') as f:
			pickle.dump(RF_clf,f)

		train_predictions = RF_clf.predict(X)
		test_predictions = RF_clf.predict(X_test)

		print_metrics(output, train_predictions, Y, 'Random Forest Classifier')
		np.savetxt('Random Forest Classifier - test pred.csv', np.column_stack((IDs,test_predictions)), delimiter=',', comments='', fmt='%f')

		#######################################################################################

		#######################################################################################
		# MultiLayerPerceptron
		#######################################################################################
#		print()
#		print("MultiLayer Perceptron")
#		print()
#
#		output.write('\n')
#		output.write("===== MULTILAYER PERCEPTRON CLASSIFIER =====\n")
#		output.write('\n')
#
#		MLP_clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(15, 1), random_state=0, max_iter=500)
#
#		MLP_clf.fit(X,Y)
#
#		with open('MLP_clf.pickle','wb') as f:
#			pickle.dump(MLP_clf,f)
#
#		train_predictions = MLP_clf.predict(X)
#		test_predictions = MLP_clf.predict(X_test)
#
#		print_metrics(output, train_predictions, Y, 'Neural Network Classifier')
#		np.savetxt('Neural Network Classifier - test pred.csv', test_predictions, delimiter=',', comments='', fmt='%f')
#
#
#		#######################################################################################
#		# kNN
#		#######################################################################################
#		for neighbours in range(400, 1000, 100):
#			print()
#			print("kNN {} neighbors".format(neighbours))
#			print()
#
#			output.write('\n')
#			output.write("===== kNN {} NEIGHBOURS CLASSIFIER =====\n".format(neighbours))
#			output.write('\n')
#
#			kNN_clf = KNeighborsClassifier(n_neighbors=neighbours)
#			kNN_clf.fit(X,Y)
#
#			with open('kNN_clf_' + str(neighbours) + '.pickle','wb') as f:
#				pickle.dump(kNN_clf,f)
#
#			train_predictions = kNN_clf.predict(X)
#			test_predictions = kNN_clf.predict(X_test)
#			print_metrics(output, train_predictions, Y, 'kNN {} Classifier'.format(str(neighbours)))
#			np.savetxt('kNN {} Classifier - test pred.csv'.format(str(neighbours)), test_predictions, delimiter=',', comments='', fmt='%f')


#######################################################################################
# ANOMALY DETECTION
#######################################################################################
def AnomalyDetection(filepath):
	train_X = np.loadtxt(filepath+'normalized_train_file.csv', delimiter=',', dtype=float, skiprows=1)
	test_X = np.loadtxt(filepath+'pseudonormalized_test_file.csv', delimiter=',',dtype=float, skiprows=1)
	train_Y = np.loadtxt(filepath+'Y_train_file.csv', delimiter=',',dtype=float, skiprows=1)
	test_Y = np.loadtxt(filepath+'Y_test_file.csv', delimiter=',', dtype=float, skiprows=1)
	input_dimensions = str(train_X.shape[1]) #feature length
	samples_size =str(train_X.shape[0]) #number of rows
	input_dimensions_test = str(test_X.shape[1] )#feature length
	samples_size_test = str(test_X.shape[0]) #number of rows
	num_failed_train = train_Y[train_Y==1].shape[0]
	num_failed_test = test_Y[test_Y==1].shape[0]

	with open(filepath+'outliers_new_results.txt', 'w') as output:
		output.write("===== DATA INFORMATION =====\n")
		output.write('training data size: ' +samples_size +' by '+ input_dimensions+'\n')
		output.write('test data size: '  +samples_size_test +' by '+ input_dimensions_test+'\n')
		output.write('failed points in training: ' + str(num_failed_train))
		output.write('failed points in testing: ' + str(num_failed_test))

		#change input data for this method:
		training = train_X[np.where(train_Y==0)]
		testing = np.concatenate((test_X,train_X[np.where(train_Y==1)]))
		testing_Y =  np.concatenate((test_Y,train_Y[np.where(train_Y==1)]))
		input_dimensions = str(training.shape[1]) #feature length
		samples_size =str(training.shape[0]) #number of rows
		input_dimensions_test = str(testing.shape[1] )#feature length
		samples_size_test = str(testing.shape[0]) #number of rows
		#####################################################################
		# ONE CLASS SVM
		#####################################################################
		print()
		print('One Class SVM') # healthy data to train only
		print()

		output.write("\n===== ONE CLASS SVM =====\n")
		output.write("===== DATA INFORMATION FOR THIS METHOD 	=====\n")
		output.write('training data size: ' +samples_size +' by '+ input_dimensions+'\n')
		output.write('test data size: '  +samples_size_test +' by '+ input_dimensions_test+'\n')
		output.write('training set is all healthy data, testing set contains other data and all failed points\n')

		clf = svm.OneClassSVM(nu=0.15, kernel='rbf', gamma=0.75) # nu=0.15
		clf.fit(training)
		with open(filepath+'svm_one_class.pickle','wb') as f:
			pickle.dump(clf,f)
		y_pred_train = clf.predict(training)
		y_pred_test = clf.predict(testing)
		anomaly_detection_error(y_pred_train, train_Y[train_Y==0], "training", output, filepath+'OneClassSVM', OneClassSVMMethod=True)
		anomaly_detection_error(y_pred_test, testing_Y, "testing", output, filepath+'OneClassSVM', OneClassSVMMethod=True)

		#####################################################################
		# ISOLATION FOREST
		#####################################################################
		print()
		print('IsolationForest')
		print()

		output.write("\n===== ISOLATION FOREST =====\n")

		# Example settings
		n_samples = 100
		samples_max = 0.7336951612320737
		contamination_fraction = 0.11294048783176784

		clf = IsolationForest(n_estimators=n_samples,
								max_samples=samples_max,
								contamination=contamination_fraction,
								random_state=0)
		clf.fit(train_X)
		with open(filepath+'IsolationForest.pickle','wb') as f:
			pickle.dump(clf,f)
		y_pred_train = clf.predict(train_X)
		y_pred_test = clf.predict(test_X)
		anomaly_detection_error(y_pred_train, train_Y, "training", output, filepath+'Isolation Forest')
		anomaly_detection_error(y_pred_test, test_Y, "testing", output, filepath+'Isolation Forest')
					
		#####################################################################
		# ELLIPTIC ENVELOPE
		#####################################################################
		print()
		print('Elliptic Envelope')
		print()

		output.write("\n===== ELLIPTIC ENVELOPE =====\n")

		clf = EllipticEnvelope(contamination=0.175, random_state=0)
		clf.fit(train_X)
		with open(filepath+'EllipticEnvelope.pickle','wb') as f:
			pickle.dump(clf,f)
		y_pred_train = clf.predict(train_X)
		y_pred_test = clf.predict(test_X)
		anomaly_detection_error(y_pred_train, train_Y, "training", output, filepath+'EE')
		anomaly_detection_error(y_pred_test, test_Y, "testing", output, filepath+'EE')
		
		#####################################################################
		# LOCAL OUTLIER FACTOR
		#####################################################################
		print()
		print('Local Outlier Factor')
		print()

		output.write("\n=====LOCAL OUTLIER FACTOR =====\n'")

		for i in [100, 150, 200, 500, 1000]:
			clf = LocalOutlierFactor(n_neighbors=i, contamination=0.25)

			y_pred_train = clf.fit_predict(train_X)
			y_pred_test = clf._predict(test_X)
			anomaly_detection_error(y_pred_train, train_Y, "training", output, filepath+'LOF')
			anomaly_detection_error(y_pred_test, test_Y, "testing", output, filepath+'LOF')
			with open('R:\\SMM-Structures\\A1-010391 (Navy IPMS data analytics)\\Technical\\Data\\datafiles\\'+'LOF {} neighbours.pickle'.format(i),'wb') as f:
				pickle.dump(clf,f)
		print()
	