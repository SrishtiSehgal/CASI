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

#DEVICE SETTING
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

###############################################################

#RANDOM SEED SETTING
torch.manual_seed(0)
np.random.seed(0)

###############################################################

def model_training(autoencoder, neuralnet, 
				   train_loader, epoch, best_loss, 
				   filename, sparse, lr, weight_decay,
				   log, size_batch, loss_weights):
	
	#VISDOM LOSS INITIALIZER
	#objects to store & plot the losses
	Totallosses = utils.AverageMeter()
	MSElosses = utils.AverageMeter()
	Sparselosses = utils.AverageMeter()
	Classlosses = utils.AverageMeter()
	
	#INITIALIZE LOSS AND OPTIMIZERS
	total_loss, total_mse_loss, total_kl_loss, total_bce_loss = torch.tensor([0.], requires_grad=False),torch.tensor([0.], requires_grad=False),torch.tensor([0.], requires_grad=False), torch.tensor([0.], requires_grad=False)
	AE_loss, NN_loss = nn.MSELoss(), nn.MSELoss()#nn.BCEWithLogitsLoss()
	optimizer_AE = torch.optim.Adam(
		chain(autoencoder.parameters(),neuralnet.parameters()),
		lr=lr)
	autoencoder.train()
	neuralnet.train()
	# TRAIN IN MINI-BATCHES
	for i, data in enumerate(train_loader):
		#CLEAR GRADIENT BUFFERS
		optimizer_AE.zero_grad()
		
		#GET BATCH DATA FROM LOADER
		X, Y = data
		write_progress_to_file(filename, 'batch', str(i+1))
		
		#PREDICTION
		kl_loss, latent, outputs = autoencoder(X)
		class_pred = neuralnet(latent)
		predicted_labels = probability_label(class_pred)
		#CALCULATE LOSSES
		mse_loss = AE_loss(outputs, X)
		bce_loss = NN_loss(class_pred, Y)
		loss = kl_loss*sparse + mse_loss + bce_loss
		
		#SAVE LATENT SPACE FROM EACH BATCH
		if i==0:
			Latent_dataspace = latent.detach().numpy()
		else:
			Latent_dataspace = np.concatenate((Latent_dataspace,latent.detach().numpy()),axis=0)

		#SAVE PREDICTED TO COMPUTE ACCURACY
		if i==0:
			out = predicted_labels.numpy()
			label = Y.numpy()
		else:
			out = np.concatenate((out,predicted_labels.numpy()),axis=0)
			label = np.concatenate((label, Y.numpy()),axis=0)

		if i == 0:
			new_data = outputs.detach().numpy()
		else:
			new_data = np.concatenate((new_data,outputs.detach().numpy()), axis=0)
		
		#ACCUMULATING LOSS FROM EACH BATCH TO PLOT
		total_loss += loss 
		total_mse_loss += mse_loss 
		total_kl_loss += kl_loss* sparse 
		total_bce_loss += bce_loss

#		print('before: grad', [x.grad for x in list(autoencoder.parameters())[:1]],'\n')
#		print(list(autoencoder.parameters())[:1])

		#UPDATE
		loss.backward()
#		print('updated grad', [x.grad for x in list(autoencoder.parameters())[:1]],'\n')
		optimizer_AE.step()
#		print(list(autoencoder.parameters())[:1])
		# optimizer_CNN.step()
		if (i + 1) % log == 0:
#			print('Epoch [{}/{}] - Batch[{}/{}],\n\
#				  Total loss:{:.4f},\n\
#				  MSE loss:{:.4f},\n\
#				  Sparse loss:{:.4f},\n\
#				  Classification loss:{:.4f}'.format(
#				epoch + 1, 
#				EPOCHS, i + 1, 
#				len(train_loader.dataset) // size_batch, 
#				total_loss.item(), 
#				0.01*mse_loss.item(), 
#				kl_loss.item()* sparse,
#				10000*bce_loss.item()))
#				(10**magnitude(mse_loss.item()))*bce_loss.item()))
			
			#UPDATE PROGRESS REPORT
			write_progress_to_file(filename, 'Accumulated Loss', 
								   str(total_loss.item()))
			write_progress_to_file(filename, 'MSE Loss', 
								   str(total_mse_loss.item()))
			write_progress_to_file(filename, 'Sparse Loss', 
								   str(total_kl_loss.item()*sparse))
			write_progress_to_file(filename, 'Classification Loss', 
								   str(total_bce_loss.item())+'\n')

	#UPDATE LOSS OBJECTS WITH TOTAL_LOSS/NUM_BATCHES = avg batch loss for this epoch
	# PLOT LOSSES AFTER ALL MINI-BATCHES HAVE FINISHED TRAINING
	Totallosses.update(total_loss.data.numpy(), len(train_loader))
	plotter.plot('total loss', 'train-total', 'total loss', epoch, Totallosses.avg)       
	MSElosses.update(total_mse_loss.data.numpy(), len(train_loader))
	plotter.plot('total loss', 'train-mse', 'total loss', epoch, MSElosses.avg)       
	Sparselosses.update(total_kl_loss.data.numpy(), len(train_loader))
	plotter.plot('total loss', 'train-sparse', 'total loss', epoch, Sparselosses.avg)       
	Classlosses.update(total_bce_loss.data.numpy(), len(train_loader))
	plotter.plot('total loss', 'train-classification', 'total loss', epoch, Classlosses.avg)       

	# ACCURACY
	acc = np.sum(out == label)/len(out) #need better accuracy metric...consider confusion matrix here!
	TPR, FPR, PPV, FNR = ROC(label, out, open(filename,'a'))
	write_progress_to_file(filename, 'Train-Accuracy', str(acc)+'\n')

	plotter.plot('accuracy', 'train', 'Class Accuracy', epoch, acc)
	plotter.plot('rate', 'TPR', 'ROC metrics in training', epoch, TPR)
	plotter.plot('rate', 'FPR', 'ROC metrics in training', epoch, FPR)
	plotter.plot('rate', 'PPV', 'ROC metrics in training', epoch, PPV)
	plotter.plot('rate', 'FNR', 'ROC metrics in training', epoch, FNR)

	#IF TRAINING AND AVG BATCH LOSS BECOMES NEW BEST, SAVE MODELS
	with torch.no_grad(): 
		avg_loss = Totallosses.avg
		if avg_loss < best_loss:
			best_loss = avg_loss
	
			#SAVE MODELS
			torch.save(autoencoder.state_dict(), 'sparse_autoencoder_KL.pt')
			torch.save(neuralnet.state_dict(), 'CNN.pt')
			print('Saved Best Models at epoch {} in folder\n'.format(str(epoch+1)))
			np.savetxt('Latent_dataspace.csv', Latent_dataspace, delimiter=',', fmt='%f', encoding='UTF-8')
			np.savetxt('output_dataspace.csv', new_data, delimiter=',', fmt='%f', encoding='UTF-8')
	
		return best_loss

def evaluation(autoencoder, neuralnet, test_loader, epoch, filename, loss_weights):
	
	#VISDOM LOSS INITIALIZER
	# objects to store & plot the losses
	testTotallosses = utils.AverageMeter()
	Classlosstest = utils.AverageMeter()
	MSElosstest = utils.AverageMeter()
	autoencoder.eval()
	neuralnet.eval()
	with torch.no_grad():     
		
		#INITIALIZE LOSS
		AE_loss, NN_loss = nn.MSELoss(),nn.MSELoss() #nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([10.])) #play with this parameter
		total_loss, total_mse_loss, total_bce_loss = torch.tensor([0.], requires_grad=False),torch.tensor([0.], requires_grad=False),torch.tensor([0.], requires_grad=False)		
		#MINI-BATCHES
		for i, data in enumerate(test_loader):
			
			#GET BATCH DATA FROM LOADER
			X, Y = data
			write_progress_to_file(filename, 'Test-Batch', str(i+1))
			
			#PREDICTION
			_, latent, outputs = autoencoder(X)
			class_pred = neuralnet(latent)
#			class_pred = nn.Sigmoid()(class_pred)
			predicted_labels = probability_label(class_pred)
	
			#SAVE LATENT SPACE FROM EACH BATCH
			if i==0:
				Latent_dataspace = latent.detach().numpy()
			else:
				Latent_dataspace = np.concatenate((Latent_dataspace,latent.detach().numpy()),axis=0)

#			#SAVE PREDICTED TO COMPUTE ACCURACY
			if i==0:
				out = predicted_labels.numpy()
				label = Y.numpy()
			else:
				out = np.concatenate((out,predicted_labels.numpy()),axis=0)
				label = np.concatenate((label, Y.numpy()),axis=0)
			if i == 0:
				new_data = outputs.detach().numpy()
			else:
				new_data = np.concatenate((new_data,outputs.detach().numpy()), axis=0)
			#CALCULATE LOSSES
			mse_loss = AE_loss(outputs, X)
			total_mse_loss += mse_loss
			bce_loss = NN_loss(class_pred, Y)
			total_bce_loss += bce_loss
#			loss = 0.01*mse_loss + 10000*bce_loss #(10**magnitude(mse_loss))
			loss = mse_loss + bce_loss
			total_loss += loss

#			print('TEST')
#			print('Epoch [{}/{}],\n\
#				  Total loss:{:.4f},\n\
#				  MSE loss:{:.4f},\n\
#				  Classification loss:{:.4f}'.format(
#				epoch + 1, EPOCHS, 
#				total_loss.item(), 
#				0.01*mse_loss.item(), 
#				10000*bce_loss.item()))
##				(10**magnitude(mse_loss.item()))*bce_loss.item()))

			#UPDATE PROGRESS REPORT
			write_progress_to_file(filename, 'Test-Total Loss', 
								   str(total_loss.item()))
			write_progress_to_file(filename, 'Test-MSE Loss', 
								   str(mse_loss.item()))
			write_progress_to_file(filename, 'Test-Classification Loss', 
								   str((10**magnitude(mse_loss.item()))*bce_loss.item())+'\n')
		
		#UPDATE THE LOSS OBJECT
		testTotallosses.update(total_loss.data.numpy(), len(test_loader))
		Classlosstest.update(total_bce_loss.data.numpy(), len(test_loader))
		MSElosstest.update(total_mse_loss.data.numpy(), len(test_loader))
		
		# ACCURACY
		acc = np.sum(out == label)/len(out) #need better accuracy metric...consider confusion matrix here!
		TPR, FPR, PPV, FNR = ROC(label, out, open(filename,'a'))
		write_progress_to_file(filename, 'Test-Accuracy', str(acc)+'\n')
		
		print('Average total loss in testing: {}\n\
			  Accuracy in testing: {acc}'.format(str(testTotallosses.avg), acc=acc))
		write_progress_to_file(filename, 'Test-Average total loss', 
								   str(testTotallosses.avg)+'\n')
		np.savetxt('output_dataspace-test.csv', new_data, delimiter=',', fmt='%f', encoding='UTF-8')
		np.savetxt('Latent_dataspace-test.csv', Latent_dataspace, delimiter=',', fmt='%f', encoding='UTF-8')

		#PLOT TEST RESULTS
		plotter.plot('total loss', 'test-classification', 'total loss', epoch, Classlosstest.avg)
		plotter.plot('total loss', 'test-mse', 'total loss', epoch, MSElosstest.avg)
		plotter.plot('total loss', 'test-total', 'total loss', epoch, testTotallosses.avg)
		plotter.plot('accuracy', 'test', 'Class Accuracy', epoch, acc)
		plotter.plot('rate', 'TPR', 'ROC metrics in testing', epoch, TPR)
		plotter.plot('rate', 'FPR', 'ROC metrics in testing', epoch, FPR)
		plotter.plot('rate', 'PPV', 'ROC metrics in testing', epoch, PPV)
		plotter.plot('rate', 'FNR', 'ROC metrics in testing', epoch, FNR)
		
###############################################################
def ROC(true_val, predictions, output):
	cm = confusion_matrix(true_val, predictions)#, labels=['failed', 'healthy'])
	TN = cm[0][0]
	FN = cm[1][0]
	TP = cm[1][1]
	FP = cm[0][1]
	#NORMAL LABELS
	print("\t \t | failure | healthy |")
	print("predicted failure | " + str(TP) + "\t|\t" + str(FP) + "|")
	print("predicted healthy | " + str(FN) + "\t|\t" + str(TN) + "|")
	print()

	output.write("\t \t | failure | healthy |\n")
	output.write("predicted failure | " + str(TP) + "\t|\t" + str(FP) + "|\n")
	output.write("predicted healthy | " + str(FN) + "\t|\t" + str(TN) + "|\n")
	output.close()
	
	return round(TP/(TP+FN),4), round(FP/(FP+TN),4), round(TP/(TP+FP),4), round(FN/(FN+TP),4)

def norm(orig_csvfile, norm, avg=0, stdev=1, start_dig_signals=42):
	if norm:
		avg = np.mean(orig_csvfile[:,0:start_dig_signals], axis = 0, dtype=np.float64)
		stdev = np.std(orig_csvfile[:,0:start_dig_signals], axis = 0, dtype=np.float64)
		np.savetxt('stats_avg.csv', avg, delimiter=',', fmt = '%f')
		np.savetxt('stats_std.csv', stdev, delimiter=',',fmt = '%f')

	csvfile = np.subtract(orig_csvfile[:,0:start_dig_signals],avg)
	csvfile = np.divide(csvfile,stdev)
	return np.column_stack((csvfile, orig_csvfile[:,start_dig_signals:])), avg, stdev

def create_datasets(HX, WX, channelsX=1):
	x = torch.randn((HX, WX))
#    target = torch.zeros(HX//2,1)
#    target = torch.cat((target,torch.ones(HX-(HX//2),1)),0)
#    target = target.view(*target.size())
	#CLASSIFICATION TARGET
#    x = torch.randn((channelsX, HX, WX))
	target = torch.zeros(HX//4,1, dtype=torch.float)
	target = torch.cat((target,torch.ones(HX-(HX//4),1)),0)
#    target = target.view(1, *target.size())
	return x, target

def magnitude(x):
	return int(math.log10(x)+1.0)

def cal_shape(prev, ker, pad=0, dil=1, stride=1):
	return math.floor(1 + ( (prev + 2*pad - dil*(ker - 1) - 1) / stride) )

def write_progress_to_file(filename, item, statement):
	with(open(filename,'a')) as f:
		f.write(item+ ': '+ statement+'\n')

def probability_label(prob):
#	prob = nn.Sigmoid()(prob)
	labels = torch.zeros(prob.size())
	for i, item in enumerate(prob):
		if item < 0.5:
			continue
		if 0.5 <= item:
			labels[i] = 1
	return labels

def convert_labels(labels):
	new_labels = np.ones((labels.shape[0],1), dtype=float)
	count = 0
	for i, label in enumerate(labels):
		if label == 1.:
			new_labels[i] = 0.
			count+=1
	print(count)
	input()
	return new_labels

def one_hot_vector(Y):
    pass
#    new_labels = np.zeros((labels.shape[0],1), dtype=float)


#def create_datasets(HX, WX, channelsX=1):
#    x = torch.randn((channelsX, HX, WX))
#    target = torch.zeros(channelsX//4,1, dtype=torch.float)
#    target = torch.cat((target,torch.ones(channelsX-(channelsX//4),1)),0)
##    target = target.view(*target.size(), 1)
#    return x, target
###############################################################
	
if __name__ == '__main__':
	
	###############################################################
	
	#FAKE DATA PARAMETERS
	SAMPLES = 10000
	FEATURES = 51
	FILENAME = 'run_progress.txt'

	#GENERAL PARAMETERS
	TRAIN_SCRATCH = True        # whether to train a model from scratch
	EPOCHS = 25 #500
	SHUFFLE = False
	BATCH_SIZE = 200
	LOG_INTERVAL = 200
	BEST_VAL = float('inf')     # record the best val loss
	LOSS_WEIGHT = [0.1, 1000, 1]
	EVENT = 'REDO'

	#AE PARAMETERS
	LEARNING_RATE = 1e-4##8e-4
	WEIGHT_DECAY = 1e-5
	SPARSITY = 0.1
	SPARSE_REG =  0.0008
	INPUT_SIZES = [51, 40]
	OUTPUT_SIZES= [60, 51]
	#CNN PARAMETERS
#	CHANNELS=[1,20,14,10]
#	KERNEL_CONV=[7]#some sizes may not work due to integer casting in CNN.py
#	STRIDE_CONV = [1,1]
#	PADDING_CONV = [0,0]
#	FC= [6, 1]
	
	################################################################
	#INPUT DATA (PRE-NORMALIZED)
	X_train = np.loadtxt('normalized_train_file.csv', delimiter=',', dtype=float)
	X_test = np.loadtxt('pseudonormalized_test_file.csv', delimiter=',', dtype=float)
	Y_train = np.loadtxt('Y_train_file.csv', delimiter=',', dtype=float)
	Y_test = np.loadtxt('Y_test_file.csv', delimiter=',', dtype=float)
	# Y_train = convert_labels(Y_train)
	# Y_test = convert_labels(Y_test)
	X_train = torch.from_numpy(X_train).float()
	X_test = torch.from_numpy(X_test).float()
	Y_train = torch.from_numpy(Y_train.reshape(Y_train.shape[0],1)).float()
	Y_test = torch.from_numpy(Y_test.reshape(Y_test.shape[0],1)).float()
	write_progress_to_file(FILENAME, 'X_train size', str(X_train.size()))
	write_progress_to_file(FILENAME, 'Y_train size', str(Y_train.size()))
	write_progress_to_file(FILENAME, 'X_test size', str(X_test.size()))
	write_progress_to_file(FILENAME, 'Y_test size', str(Y_test.size())+'\n')
	
	#DATA GENERATORS
	train_loader = data_utils.DataLoader(
		data_utils.TensorDataset(X_train, Y_train),
		shuffle=SHUFFLE, 
		batch_size = BATCH_SIZE)
	test_loader = data_utils.DataLoader(
		data_utils.TensorDataset(X_test, Y_test),
		shuffle=SHUFFLE, 
		batch_size = BATCH_SIZE)
	
	################################################################
	
	#INITIALIZE MODELS AND VISDOM
	# Plots
	global plotter
	plotter = utils.VisdomLinePlotter(env_name=EVENT)
#	convnet = CNN.CNN(CHANNELS,OUTPUT_SIZES[-1],
#					  KERNEL_CONV, STRIDE_CONV, 
#					  PADDING_CONV,FC,
#					  FILENAME)
#	if cuda: convnet.to(device) #run on GPU if available
	autoencoder = SAE.AutoEncoder(INPUT_SIZES, OUTPUT_SIZES, SPARSITY)
	if cuda: autoencoder.to(device) #run on GPU if available
	
	neuralnet = CNN.NeuralNet(INPUT_SIZES[-1], FILENAME)
	if cuda: neuralnet.to(device) #run on GPU if available

	################################################################
	# write_progress_to_file(FILENAME, 'Weights (MSE, BCE, Sparse)', str(LOSS_WEIGHT)+'\n')

	#TRAIN THEN EVALUATE MODEL
	if TRAIN_SCRATCH: # Train autoencoder from scratch?
		for epoch in range(EPOCHS):
			write_progress_to_file(FILENAME, 'epoch', str(epoch+1))
	 
			#MEASURE TIME TO TRAIN MODEL PER EPOCH
			starttime = time.time()
			BEST_VAL = model_training(autoencoder, neuralnet, 
									  train_loader, epoch, BEST_VAL, FILENAME,
									  SPARSE_REG, LEARNING_RATE, WEIGHT_DECAY,
									  LOG_INTERVAL, BATCH_SIZE, LOSS_WEIGHT)
			endtime = time.time()
			print('Epoch {} completed in {} seconds'.format(
					str(epoch+1), str((endtime - starttime))))
			write_progress_to_file(FILENAME, 
								   'elapsed time', 
								   str((endtime - starttime))+'\n')
		
			#EVALUATE MODEL
			write_progress_to_file(FILENAME,'Test-Epoch',str(epoch+1))
			evaluation(autoencoder, neuralnet, test_loader, epoch, 
						FILENAME, LOSS_WEIGHT)
		
		write_progress_to_file(FILENAME, 
							   'best avg batch loss', 
							   str((BEST_VAL.item())))
		print('Training Complete! Best average batch loss: {:.4f}'.format(
				BEST_VAL.item()))
		
	################################################################

	#LOAD AND EVALUATE MODEL
	else:

#		#LOAD MODEL
		autoencoder.load_state_dict(torch.load('sparse_autoencoder_KL.pt'))
##		convnet.load_state_dict(torch.load('CNN.pt'))
#		
#		#EVALUATE MODEL
#		write_progress_to_file(FILENAME,'TESTING','STARTS HERE')
#		evaluation(autoencoder, test_loader, FILENAME, LOSS_WEIGHT)