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

def model_training(autoencoder, cnn, 
                   train_loader, epoch, best_loss, 
                   filename, sparse, lr, weight_decay,
                   log, size_batch):
    
    #VISDOM LOSS INITIALIZER
    #objects to store & plot the losses
    Totallosses = utils.AverageMeter()
    
    #INITIALIZE LOSS AND OPTIMIZERS
    total_loss=0
    AE_loss, CNN_loss = nn.MSELoss(), nn.BCEWithLogitsLoss()
    optimizer_AE = torch.optim.Adam(autoencoder.parameters(), 
        lr=lr, weight_decay=weight_decay)
    optimizer_CNN = torch.optim.Adam(cnn.parameters(), 
        lr=lr, weight_decay=weight_decay)
    cnn.train()
    autoencoder.train()
    
    # TRAIN IN MINI-BATCHES
    for i, data in enumerate(train_loader):
        #CLEAR GRADIENT BUFFERS
        optimizer_AE.zero_grad()
        optimizer_CNN.zero_grad()
        
        #GET BATCH DATA FROM LOADER
        X, Y = data
        write_progress_to_file(filename, 'batch', str(i+1))
        
        #PREDICTION
        kl_loss, latent, outputs = autoencoder(X)
        class_pred = cnn(latent)
        
        #CALCULATE LOSSES
        kl_loss.requires_grad = False
        mse_loss = AE_loss(outputs, X)
        bce_loss = CNN_loss(class_pred, Y)
        loss = mse_loss + kl_loss * sparse + bce_loss
        total_loss += loss #accumulating loss from each batch
                
        #UPDATE
        loss.backward()
        optimizer_AE.step()
        optimizer_CNN.step()
        if (i + 1) % log == 0:
            print('Epoch [{}/{}] - Batch[{}/{}],\n\
                  Total loss:{:.4f},\n\
                  MSE loss:{:.4f},\n\
                  Sparse loss:{:.4f},\n\
                  Classification loss:{:.4f}'.format(
                epoch + 1, 
                EPOCHS, i + 1, 
                len(train_loader.dataset) // size_batch, 
                loss.item(), 
                mse_loss.item(), 
                kl_loss.item()* sparse,
                bce_loss.item()))
            
            #UPDATE PROGRESS REPORT
            write_progress_to_file(filename, 'Accumulated Loss', 
                                   str(total_loss.item()))
            write_progress_to_file(filename, 'Loss', 
                                   str(loss.item()))
            write_progress_to_file(filename, 'MSE Loss', 
                                   str(mse_loss.item()))
            write_progress_to_file(filename, 'Sparse Loss', 
                                   str(kl_loss.item()*sparse))
            write_progress_to_file(filename, 'Classification Loss', 
                                   str(bce_loss.item())+'\n')

    #UPDATE LOSS OBJECTS WITH TOTAL_LOSS/NUM_BATCHES = avg batch loss for this epoch
    # PLOT LOSSES AFTER ALL MINI-BATCHES HAVE FINISHED TRAINING
    Totallosses.update(total_loss.data.numpy(), len(train_loader))
    plotter.plot('total loss', 'train', 'total loss', epoch, Totallosses.avg)       
    
    #IF TRAINING AND AVG BATCH LOSS BECOMES NEW BEST, SAVE MODELS
    avg_loss = Totallosses.avg
    if avg_loss < best_loss:
        best_loss = avg_loss

        #SAVE MODELS
        torch.save(autoencoder.state_dict(), 'sparse_autoencoder_KL.pt')
        torch.save(cnn.state_dict(), 'CNN.pt')
        print('Saved Best Models at epoch {} in folder\n'.format(str(epoch+1))) 

    return best_loss

def evaluation(autoencoder, cnn, test_loader, epoch, filename):
    
    #VISDOM LOSS INITIALIZER
    # objects to store & plot the losses
    testTotallosses = utils.AverageMeter()

    autoencoder.eval()
    cnn.eval()
    with torch.no_grad():     
        
        #INITIALIZE LOSS
        total_loss = 0
        AE_loss, CNN_loss = nn.MSELoss(), nn.BCEWithLogitsLoss()
        
        #MINI-BATCHES
        for i, data in enumerate(test_loader):
            
            #GET BATCH DATA FROM LOADER
            X, Y = data
            write_progress_to_file(filename, 'Test-Batch', str(i+1))
            
            #PREDICTION
            _, latent, outputs = autoencoder(X)
            class_pred = cnn(latent)
            predicted_labels = probability_label(class_pred)
            
            #SAVE PREDICTED TO COMPUTE ACCURACY
            if i==0:
                out = predicted_labels.numpy()
                label = Y.numpy()
            else:
                out = np.concatenate((out,predicted_labels.numpy()),axis=0)
                label = np.concatenate((label, Y.numpy()),axis=0)

            #CALCULATE LOSSES
            mse_loss = AE_loss(outputs, X)
            bce_loss = CNN_loss(class_pred, Y)
            loss = mse_loss + bce_loss
            total_loss += loss
        
            #UPDATE PROGRESS REPORT
            write_progress_to_file(filename, 'Test-Total Loss', 
                                   str(loss.item()))
            write_progress_to_file(filename, 'Test-MSE Loss', 
                                   str(mse_loss.item()))
            write_progress_to_file(filename, 'Test-Classification Loss', 
                                   str(bce_loss.item())+'\n')
        
        #UPDATE THE LOSS OBJECT
        testTotallosses.update(total_loss.data.numpy(), len(test_loader))
        
        # ACCURACY
        acc = np.sum(out == label)/len(out)
        write_progress_to_file(filename, 'Test-Accuracy', str(acc)+'\n')
        
        print('Average total loss in testing: {:.4f}\n'
              'Accuracy in testing: {acc}'.format(testTotallosses.avg, acc=acc))
        write_progress_to_file(filename, 'Test-Average total loss', 
                                   str(testTotallosses.avg)+'\n')
        
        #PLOT TEST RESULTS
        plotter.plot('total loss', 'test', 'total loss', epoch, testTotallosses.avg)
        plotter.plot('accuracy', 'test', 'Class Accuracy', epoch, acc)
        
###############################################################
    
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

def cal_shape(prev, ker, pad=0, dil=1, stride=1):
    return math.floor(1 + ( (prev + 2*pad - dil*(ker - 1) - 1) / stride) )

def write_progress_to_file(filename, item, statement):
    with(open(filename,'a')) as f:
        f.write(item+ ': '+ statement+'\n')

def probability_label(prob):
    labels = torch.zeros(prob.size())
    for i, item in enumerate(prob):
        if item < 0.5:
            continue
        if 0.5 <= item <= 1.0:
            labels[i] =1
    return labels

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
    SAMPLES = 3000
    FEATURES = 51
    FILENAME = 'run_progress.txt'

    #AE PARAMETERS
    EPOCHS = 10
    SHUFFLE = True
    BATCH_SIZE = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    LOG_INTERVAL = 1
    DISTRIBUTION_VAL = 0.03
    SPARSE_REG = 0.01
    TRAIN_SCRATCH = True        # whether to train a model from scratch
    BEST_VAL = float('inf')     # record the best val loss
    INPUT_SIZES = [51, 60, 40, 30]
    OUTPUT_SIZES = [60, 40, 30, 20]
    
    #CNN PARAMETERS
    CHANNELS=[1,20,14,10]
    KERNEL_CONV=[7]#some sizes may not work due to integer casting in CNN.py
    STRIDE_CONV = [1,1]
    PADDING_CONV = [0,0]
    FC= [6, 1]
    
    ################################################################
    
    #INPUT DATA
    X_train, Y_train = create_datasets(SAMPLES, FEATURES)
    write_progress_to_file(FILENAME, 'X_train size', str(X_train.size()))
    write_progress_to_file(FILENAME, 'Y_train size', str(Y_train.size()))
    X_test, Y_test = create_datasets(int(SAMPLES),FEATURES)
    write_progress_to_file(FILENAME, 'X_test size', str(X_test.size()))
    write_progress_to_file(FILENAME, 'Y_test size', str(Y_test.size())+'\n')
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, Y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE)
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_test, Y_test), 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE)
    
    ################################################################
    
    #INITIALIZE MODELS AND VISDOM
    # Plots
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='CASI Plots')
#    first_input, target = create_datasets(1,features, channelsX=samples)
    convnet = CNN.CNN(CHANNELS,OUTPUT_SIZES[-1],
                      KERNEL_CONV, STRIDE_CONV, 
                      PADDING_CONV,FC,
                      FILENAME)
    if cuda: convnet.to(device) #run on GPU if available
    autoencoder = SAE.AutoEncoder(INPUT_SIZES, OUTPUT_SIZES, 
                                  DISTRIBUTION_VAL, FILENAME)
    if cuda: autoencoder.to(device) #run on GPU if available
    
    ################################################################
    
    #TRAIN THEN EVALUATE MODEL
    if TRAIN_SCRATCH: # Train autoencoder from scratch?
        for epoch in range(EPOCHS):
            write_progress_to_file(FILENAME, 'epoch', str(epoch+1))
     
            #MEASURE TIME TO TRAIN MODEL PER EPOCH
            starttime = time.time()
            BEST_VAL = model_training(autoencoder, convnet, 
                                      train_loader, epoch, BEST_VAL, FILENAME,
                                      SPARSE_REG, LEARNING_RATE, WEIGHT_DECAY,
                                      LOG_INTERVAL, BATCH_SIZE)
            endtime = time.time()
            print('Epoch {} completed in {} seconds'.format(
                    str(epoch+1), str((endtime - starttime))))
            write_progress_to_file(FILENAME, 
                                   'elapsed time', 
                                   str((endtime - starttime))+'\n')
        
            #EVALUATE MODEL
            write_progress_to_file(FILENAME,'Test-Epoch',str(epoch+1))
            evaluation(autoencoder, convnet, test_loader, epoch, FILENAME)
        
        write_progress_to_file(FILENAME, 
                               'best avg batch loss', 
                               str((BEST_VAL.item())))
        print('Training Complete! Best average batch loss: {:.4f}'.format(
                BEST_VAL.item()))
        
    ################################################################

    #LOAD AND EVALUATE MODEL
    else:

        #LOAD MODEL
        autoencoder.load_state_dict(torch.load('sparse_autoencoder_KL.pt'))
        convnet.load_state_dict(torch.load('CNN.pt'))
        
        #EVALUATE MODEL
        write_progress_to_file(FILENAME,'TESTING','STARTS HERE')
        evaluation(autoencoder, convnet, test_loader, FILENAME)