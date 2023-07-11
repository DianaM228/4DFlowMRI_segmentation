# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:24:24 2022

@author: Diana_MARIN
"""

from UtilsDL import AortaDataset
from UtilsDL import EarlyStopping, LRScheduler
from train_val_test import train
from train_val_test import val_test
from train_val_test import save_pred
from Networks4D import UNet4DMyronenko, UNet4DMyronenko_short,UNet4DMyronenko_DoubleF
from Networks4D import UNet4DMyronenko_HalfF,UNet4DAntoine2,Residual_UNet4D
from Networks4D import weights_init
import torch
import torch.nn as nn
import numpy as np
from custom_losses import SparseDiceLoss24D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse
import glob
import os
from torch.utils.tensorboard import SummaryWriter
import time
from natsort import natsorted
import random


def main():
    parser = argparse.ArgumentParser(description='main 3D UNet like networks')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')    
    required_named_args = parser.add_argument_group('required named arguments')
    required_named_args.add_argument('-i', '--input_path', help='path to the directory where is the image database',required=True)     
    required_named_args.add_argument('-m', '--model', help='model to train',choices=['UNet4DMyronenko_short','UNet4DMyronenko','UNet4DAntoine2',
                                                                                     'UNet4DMyronenko_DoubleF','UNet4DMyronenko_HalfF','Residual_UNet4D'],required=True)
    required_named_args.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)
    required_named_args.add_argument('-e', '--epoch', type=int, help='number of epoch', required=True)
    required_named_args.add_argument('-l', '--learning_rate', type=float, help='learning rate', required=True)
    required_named_args.add_argument('-c', '--criterion', help='loss function ( SparseDiceLoss24D)', required=True)
    required_named_args.add_argument('-o', '--output_path', help='output path', required=True)
    required_named_args.add_argument('-w', '--Weight_Frames_Loss', help='set the weight to constrain frame regularization in Sparse_loss_function (value between 0 and 1)')
   
    optional_named_args = parser.add_argument_group('optional named arguments')
    optional_named_args.add_argument('-sp', '--save_pred', type=int, help='Number to indicate every how many epochs you want to save the prediction',
                                     required=False, default=None)    
    optional_named_args.add_argument('-seed', '--Seed_pytorch', help='set the same seed for experiments', 
                                     action='store_true',required=False, default=None)
    optional_named_args.add_argument('-cluster', '--Cluster', help='Option to work in gpu, by default CPU', 
                                     action='store_true',required=False, default=None)
    optional_named_args.add_argument('-fun', '--Activation_function', help='set activation function', 
                                     required=False, default='Relu',choices=['Relu','LeakyRelu'])    
    optional_named_args.add_argument('-lr_schedule', '--lr_schedule', dest='lr_schedule',help='Train the model with Plateau lr schedule ',action='store_true', default=None)
    optional_named_args.add_argument('-early', '--early_stop', dest='early_stop', help='Train the model with early stop options',action='store_true', default=None)
    optional_named_args.add_argument('-ev', '--Evaluation_Method', help='Select cross-validation or leave-one-out evaluation, by default 90/10 split', choices=['cv','loo'], default=None)
    optional_named_args.add_argument('-f', '--Folds', help='Select folds number for cross validation',type=int,default=None)
    optional_named_args.add_argument('--fold_id', type=int, help='fold to run',required=False, default=None)
    optional_named_args.add_argument('-st', '--Save_train', help='If you want to save prediction for train images',default=None, action='store_true')
   
    #read inmputs
    args = parser.parse_args()
    verbose = args.verbose    
        # Hyperparameters
    model_to_train = args.model
    batch_size = args.batch_size
    nb_epochs = args.epoch
    learning_rate = args.learning_rate
    loss_name = args.criterion
    SaveEach = args.save_pred
    w_frames_loss = float(args.Weight_Frames_Loss)
    st = args.Save_train
    num_organs = 1
    fold_id = args.fold_id
    activation = args.Activation_function
   
    if args.Evaluation_Method=='cv' and args.Folds is None:
        parser.error("Select the number of folds for cross validation ")
   
    
    print('LR schedule activated? ',args.lr_schedule)
    print('early stopping activated? ',args.early_stop)
    print('Cluster', args.Cluster)
    print('Using Model: ', model_to_train)
    print('Activation Function selected: ',activation)
    
    ## Some information to save check point
    Parameters={'Path_dataset':args.input_path,
                'Path_results':args.output_path,
                'Model':args.model}
    
    if verbose:
        print('verbosity is turned on')
        
        
    #seed for reproducibility
    if args.Seed_pytorch:        
        seed = 2
        torch.manual_seed(seed)  #seed PyTorch random number generator
        torch.cuda.manual_seed_all(seed)
        random.seed(0)  ### Python seed
        np.random.seed(0) ## Numpy seed
        torch.backends.cudnn.benchmark = False ##  (causes cuDNN to deterministically select an algorithm or cuDNN convolution)
        torch.backends.cudnn.deterministic = True ##  Avoiding nondeterministic algorithms >> CUDA convolution determinism
    
        
    Path_images=natsorted(glob.glob(os.path.join(args.input_path,'Images','*.nii')))   
    
    if args.fold_id == None:   ###### To run all folds in a for loop
    
        if args.Evaluation_Method != None:         
            if args.Evaluation_Method == 'cv':
                print('Cross-validation')
                kf = KFold(n_splits=args.Folds,shuffle=True,random_state=42)
            elif args.Evaluation_Method == 'loo':
                print('Leve-one-out Validation')            
                kf = KFold(n_splits=len(Path_images))
       
        
            ## list to save results for each fold to give average folds results at the end
            train_dices = []
            val_dices = []
            
            counter = -1        
            for train_id, val_id in kf.split(Path_images):
                counter += 1
                
                # Net
                model = -1 
                   
                if model_to_train == 'UNet4DMyronenko':
                    model = UNet4DMyronenko(in_channel=1, out_channel=num_organs+1,activation=activation)
                    print('UNet4DMyronenko')
                elif model_to_train == 'UNet4DMyronenko_short':
                    model = UNet4DMyronenko_short(in_channel=1, out_channel=num_organs+1,activation=activation)
                elif model_to_train == 'UNet4DMyronenko_DoubleF':
                    model = UNet4DMyronenko_DoubleF(in_channel=1, out_channel=num_organs+1,activation=activation)
                elif model_to_train == 'UNet4DMyronenko_HalfF':
                    model = UNet4DMyronenko_HalfF(in_channel=1, out_channel=num_organs+1,activation=activation) 
                elif model_to_train == 'UNet4DAntoine2':
                    model = UNet4DAntoine2(in_channel=1, out_channel=num_organs+1)
                elif model_to_train == 'Residual_UNet4D':
                    model = Residual_UNet4D(in_channel=1, out_channel=num_organs+1)
    
                ## Set device
                if args.Cluster:        
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # To GPU if possible
                    if torch.cuda.device_count() > 1:
                        if verbose:
                            print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model = nn.DataParallel(model)
                else:
                    device = "cpu"
                    
                if verbose:
                    print(device)
                    
                model.to(device)
                
                # Defining the loss function
                assert loss_name in ['SparseDiceLoss24D'], \
                    "The criterion does not exist / is not implemented yet"
                    
                if loss_name == 'SparseDiceLoss24D':
                    criterion = SparseDiceLoss24D()
                
                    
                # Defining the optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                ### Set lr_scheduler and early_stopping if you wanna use it
                if args.lr_schedule:    
                    lr_scheduler = LRScheduler(optimizer)
                if args.early_stop:
                    early_stopping = EarlyStopping()
                    
                # Folder to save fold results
                Folddir = os.path.join(args.output_path,'Fold'+str(counter))
                path_save_Best = os.path.join(args.output_path,'Fold'+str(counter),'Best_Model_Pred')
                if not os.path.isdir(path_save_Best):
                    os.makedirs(path_save_Best)
                    
                # default `log_dir` is "runs" - we'll be more specific here
                writer = SummaryWriter(log_dir=os.path.join(Folddir, 'runs'+"BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)),
                                       comment="BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)+'_wt'+str(w_frames_loss))
    
                if verbose:
                    print("train+val:",'\n', Path_images)
                    
                train_dataset = AortaDataset(database_path=args.input_path, list_ids=train_id,path_id=Path_images,num_organs=num_organs)   
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                
                val_dataset = AortaDataset(database_path=args.input_path, list_ids=val_id, path_id=Path_images,num_organs=num_organs)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                # train
                model.apply(weights_init)
                
                best_Metric = 999999
                trainLoss = []
                ValLoss = []
                
                for epoch in range(nb_epochs):
                    if verbose:
                        print("current epoch:", str(epoch))
                    # train
                    current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device, w_frames_loss,
                                                                    verbose=verbose)
                    trainLoss.append(current_train_loss)
                    # val
                    current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, w_frames_loss, verbose=verbose)
                    
                    ValLoss.append(current_val_loss)
                    # log the loss + dice
                    writer.add_scalars('loss_cv/training-validation_loss_' + str(counter), {'training': current_train_loss,
                                                                                            'validation': current_val_loss}, epoch)
                    writer.add_scalars('dice_cv/training-validation_dice_' + str(counter), {'training': current_train_dice,
                                                                                            'validation': current_val_dice}, epoch)
                    
                    if args.lr_schedule:
                       lr_scheduler(current_val_loss)
                    if args.early_stop:
                        early_stopping(current_val_dice) ### To avoid include the frames term in the criterion
                        if early_stopping.early_stop: 
                            break
                        
                    ## Save best model    
                     
                    if best_Metric>current_val_loss:
                        best_Metric=current_val_loss
                        
                        checkpoint = {'Parameters': Parameters,# Experiment parameters
                                      'state_dict': model.state_dict(),# Model layers and weights
                                      'optimizer': optimizer.state_dict(),# Optimizer
                                      'loss_train': current_train_loss,# Average loss history in training set each epoch
                                      'loss_val': current_val_loss,# Average loss history in validation set each epoch
                                      'Metric_train':current_train_dice,
                                      'Metric_val':current_val_dice,
                                      'Batch': args.batch_size,
                                      'epoch':epoch,
                                      'Metric':loss_name,
                                      'TrainLoss': trainLoss,
                                      'ValLoss': ValLoss
                                      }
                        torch.save(checkpoint,os.path.join(Folddir,'BestcheckPoint.pth.tar'))
                                           
                        val_test(val_loader, model, criterion, device,w_frames_loss, save_path=path_save_Best, verbose=verbose)
                    
                    
                writer.close()
                # Let's save the pred test images
                train_dices.append(current_train_dice)
                val_dices.append(current_val_dice)
                val_test(val_loader, model, criterion, device,w_frames_loss, save_path=Folddir, verbose=verbose)
                
                if st:
                    # os.path.join(args.output_path,'Fold'+str(counter)+'Pred_train')
                    path_save_train = os.path.join(args.output_path,'Fold'+str(counter),'Pred_train')
                    if not os.path.isdir(path_save_train):
                       os.makedirs(path_save_train)
                    val_test(train_loader, model, criterion, device,w_frames_loss, save_path=path_save_train, verbose=verbose)
                       
                # Let's save the model on the HDD
                    
                # Where to save the model
                model_save_path = os.path.join(Folddir, 'Last_State_dict_model_' + model_to_train + '.pth.tar')
                # Save
                            
                checkpoint = {'Parameters': Parameters,# Experiment parameters
                              'state_dict': model.state_dict(),# Model layers and weights
                              'optimizer': optimizer.state_dict(),# Optimizer
                              'loss_train': current_train_loss,# Average loss history in training set each epoch
                              'loss_val': current_val_loss,# Average loss history in validation set each epoch
                              'Metric_train':current_train_dice,
                              'Metric_val':current_val_dice,
                              'Batch': args.batch_size,
                              'epoch':epoch,
                              'Metric':loss_name,
                              'TrainLoss': trainLoss,
                              'ValLoss': ValLoss
                                      }
                torch.save(checkpoint,model_save_path)
            # Print results
            print("RESULTS")
            print("train_dice", str(round(np.mean(np.array(train_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(train_dices)),2)))
            print("val_dice", str(round(np.mean(np.array(val_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(val_dices)),2)))
    
        else:  ##### Validation by default if neither CV nor loo is selected
                
             # Net
            model = -1  ### ?
            if model_to_train == 'UNet4DMyronenko':
                model = UNet4DMyronenko(in_channel=1, out_channel=num_organs+1,activation=activation)
                print('UNet4DMyronenko')
            elif model_to_train == 'UNet4DMyronenko_short':
                model = UNet4DMyronenko_short(in_channel=1, out_channel=num_organs+1,activation=activation)
            elif model_to_train == 'UNet4DMyronenko_DoubleF':
                model = UNet4DMyronenko_DoubleF(in_channel=1, out_channel=num_organs+1,activation=activation)                
            elif model_to_train == 'UNet4DMyronenko_HalfF':
                model = UNet4DMyronenko_HalfF(in_channel=1, out_channel=num_organs+1,activation=activation)
            elif model_to_train == 'UNet4DAntoine2':
                    model = UNet4DAntoine2(in_channel=1, out_channel=num_organs+1)
            elif model_to_train == 'Residual_UNet4D':
                    model = Residual_UNet4D(in_channel=1, out_channel=num_organs+1)
                
            ## Set device
            if args.Cluster:        
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # To GPU if possible
                if torch.cuda.device_count() > 1:
                    if verbose:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
            else:
                device = "cpu"
                
            if verbose:
                print(device)
                
            model.to(device)
            
            # Defining the loss function
            assert loss_name in ['SparseDiceLoss24D'], \
                "The criterion does not exist / is not implemented yet"
                
            if loss_name == 'SparseDiceLoss24D':
                criterion = SparseDiceLoss24D()
                
            # Defining the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            ### Set lr_scheduler and early_stopping if you wanna use it
            if args.lr_schedule:    
                lr_scheduler = LRScheduler(optimizer)
            if args.early_stop:
                early_stopping = EarlyStopping()
                
            # Folder to save fold results
            Folddir = os.path.join(args.output_path)
            path_save_Best = os.path.join(args.output_path,'Best_Model_Pred')
            if not os.path.isdir(path_save_Best):
                os.makedirs(path_save_Best)
                
            writer = SummaryWriter(log_dir=os.path.join(args.output_path, 'runs'+"BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)),
                               comment="BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)+'_wt'+str(w_frames_loss))
            
            print('Default evaluation 90/10')
           
            train_id, val_id = train_test_split(np.arange(len(Path_images)), test_size=0.1, random_state=42)
    
            train_dataset = AortaDataset(database_path=args.input_path, list_ids=train_id,path_id=Path_images,num_organs=num_organs)   
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            val_dataset = AortaDataset(database_path=args.input_path, list_ids=val_id, path_id=Path_images,num_organs=num_organs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                    
            # let's retrain the model
            model.apply(weights_init)    
            
            best_Metric = 999999
            trainLoss = []
            ValLoss = []
            for epoch in range(nb_epochs):
                if verbose:
                    print("current epoch:", str(epoch))
                # train
                current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device, w_frames_loss,
                                                                verbose=verbose)
                trainLoss.append(current_train_loss)
                # val
                current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, w_frames_loss, verbose=verbose)
                
                ValLoss.append(current_val_loss)
                # log the loss + dice
                writer.add_scalars('loss_cv/training-validation_loss_', {'training': current_train_loss,
                                                                                        'validation': current_val_loss}, epoch)
                writer.add_scalars('dice_cv/training-validation_dice_', {'training': current_train_dice,
                                                                                        'validation': current_val_dice}, epoch)
                
                if args.lr_schedule:
                   lr_scheduler(current_val_loss)
                if args.early_stop:
                    early_stopping(current_val_dice) ### To avoid include the frames term in the criterion
                    if early_stopping.early_stop: 
                        break
                    
                ## Save best model    
                 
                if best_Metric>current_val_loss:
                    best_Metric=current_val_loss
                    
                    checkpoint = {'Parameters': Parameters,# Experiment parameters
                                  'state_dict': model.state_dict(),# Model layers and weights
                                  'optimizer': optimizer.state_dict(),# Optimizer
                                  'loss_train': current_train_loss,# Average loss history in training set each epoch
                                  'loss_val': current_val_loss,# Average loss history in validation set each epoch
                                  'Metric_train':current_train_dice,
                                  'Metric_val':current_val_dice,
                                  'Batch': args.batch_size,
                                  'epoch':epoch,
                                  'Metric':loss_name,
                                  'TrainLoss': trainLoss,
                                  'ValLoss': ValLoss
                                  }
                    torch.save(checkpoint,os.path.join(Folddir,'BestcheckPoint.pth.tar'))
                                       
                    val_test(val_loader, model, criterion, device,w_frames_loss, save_path=path_save_Best, verbose=verbose)
                    
            writer.close()
            # Let's save the pred test images        
            val_test(val_loader, model, criterion, device,w_frames_loss, save_path=Folddir, verbose=verbose)
            
            if st:                
                path_save_train = os.path.join(args.output_path,'Pred_train')
                if not os.path.isdir(path_save_train):
                   os.makedirs(path_save_train)
                val_test(train_loader, model, criterion, device,w_frames_loss, save_path=path_save_train, verbose=verbose)
                   
            # Let's save the model on the HDD
                
            # Where to save the model
            model_save_path = os.path.join(Folddir, 'Last_State_dict_model_' + model_to_train + '.pth.tar')
            # Save
                        
            checkpoint = {'Parameters': Parameters,# Experiment parameters
                          'state_dict': model.state_dict(),# Model layers and weights
                          'optimizer': optimizer.state_dict(),# Optimizer
                          'loss_train': current_train_loss,# Average loss history in training set each epoch
                          'loss_val': current_val_loss,# Average loss history in validation set each epoch
                          'Metric_train':current_train_dice,
                          'Metric_val':current_val_dice,
                          'Batch': args.batch_size,
                          'epoch':epoch,
                          'Metric':loss_name,
                          'TrainLoss': trainLoss,
                          'ValLoss': ValLoss
                                  }
            torch.save(checkpoint,model_save_path)
                # Print results
            print("RESULTS")
            print("train_dice", str(round(current_train_dice,2)))
            print("val_dice", str(round(current_val_dice,2)))
            
        

    else:   ###### To run each fold independently considering the Fold id provided by the user
        print('Runing Fold ',fold_id)
        if args.Evaluation_Method != None:         
            if args.Evaluation_Method == 'cv':
                print('Cross-validation')
                kf = KFold(n_splits=args.Folds,shuffle=True,random_state=42)
            elif args.Evaluation_Method == 'loo':
                print('Leve-one-out Validation')            
                kf = KFold(n_splits=len(Path_images))
       
        
            ## list to save results for each fold to give average folds results at the end
            train_dices = []
            val_dices = []
            
            counter = -1        
            for train_id, val_id in kf.split(Path_images):
                counter += 1
                if counter < fold_id:
                    continue
                else:
                    break
                
            # Net
            model = -1  ### ?
            if model_to_train == 'UNet4DMyronenko':
                model = UNet4DMyronenko(in_channel=1, out_channel=num_organs+1,activation=activation)
                print('UNet4DMyronenko')
            elif model_to_train == 'UNet4DMyronenko_short':
                model = UNet4DMyronenko_short(in_channel=1, out_channel=num_organs+1,activation=activation)
            elif model_to_train == 'UNet4DMyronenko_DoubleF':
                model = UNet4DMyronenko_DoubleF(in_channel=1, out_channel=num_organs+1,activation=activation)
            elif model_to_train == 'UNet4DMyronenko_HalfF':
                model = UNet4DMyronenko_HalfF(in_channel=1, out_channel=num_organs+1,activation=activation)
            elif model_to_train == 'UNet4DAntoine2':
                    model = UNet4DAntoine2(in_channel=1, out_channel=num_organs+1)
            elif model_to_train == 'Residual_UNet4D':
                    model = Residual_UNet4D(in_channel=1, out_channel=num_organs+1)
                
            ## Set device
            if args.Cluster:        
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # To GPU if possible
                if torch.cuda.device_count() > 1:
                    if verbose:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
            else:
                device = "cpu"
                
            if verbose:
                print(device)
                
            model.to(device)
            
            # Defining the loss function
            assert loss_name in ['SparseDiceLoss24D'], \
                "The criterion does not exist / is not implemented yet"
                
            if loss_name == 'SparseDiceLoss24D':
                criterion = SparseDiceLoss24D()
                
            # Defining the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            ### Set lr_scheduler and early_stopping if you wanna use it
            if args.lr_schedule:    
                lr_scheduler = LRScheduler(optimizer)
            if args.early_stop:
                early_stopping = EarlyStopping()
                
            # Folder to save fold results
            Folddir = os.path.join(args.output_path,'Fold'+str(counter))
            path_save_Best = os.path.join(args.output_path,'Fold'+str(counter),'Best_Model_Pred')
            if not os.path.isdir(path_save_Best):
                os.makedirs(path_save_Best)
                
            # default `log_dir` is "runs" - we'll be more specific here
            writer = SummaryWriter(log_dir=os.path.join(Folddir, 'runs'+"BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)),
                                   comment="BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)+'_wt'+str(w_frames_loss))

            if verbose:
                print("train+val:",'\n', Path_images)
                
            train_dataset = AortaDataset(database_path=args.input_path, list_ids=train_id,path_id=Path_images,num_organs=num_organs)   
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            val_dataset = AortaDataset(database_path=args.input_path, list_ids=val_id, path_id=Path_images,num_organs=num_organs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # train
            model.apply(weights_init)
            
            best_Metric = 999999
            trainLoss = []
            ValLoss = []
            for epoch in range(nb_epochs):
                if verbose:
                    print("current epoch:", str(epoch))
                # train
                current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device, w_frames_loss,
                                                                verbose=verbose)
                trainLoss.append(current_train_loss)
                # val
                current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, w_frames_loss, verbose=verbose)
                
                ValLoss.append(current_val_loss)
                # log the loss + dice
                writer.add_scalars('loss_cv/training-validation_loss_' + str(counter), {'training': current_train_loss,
                                                                                        'validation': current_val_loss}, epoch)
                writer.add_scalars('dice_cv/training-validation_dice_' + str(counter), {'training': current_train_dice,
                                                                                        'validation': current_val_dice}, epoch)
                
                if args.lr_schedule:
                   lr_scheduler(current_val_loss)
                if args.early_stop:
                    early_stopping(current_val_dice) ### To avoid include the frames term in the criterion
                    if early_stopping.early_stop: 
                        break
                    
                ## Save best model    
                 
                if best_Metric>current_val_loss:
                    best_Metric=current_val_loss
                    
                    checkpoint = {'Parameters': Parameters,# Experiment parameters
                                  'state_dict': model.state_dict(),# Model layers and weights
                                  'optimizer': optimizer.state_dict(),# Optimizer
                                  'loss_train': current_train_loss,# Average loss history in training set each epoch
                                  'loss_val': current_val_loss,# Average loss history in validation set each epoch
                                  'Metric_train':current_train_dice,
                                  'Metric_val':current_val_dice,
                                  'Batch': args.batch_size,
                                  'epoch':epoch,
                                  'Metric':loss_name,
                                  'TrainLoss': trainLoss,
                                  'ValLoss': ValLoss
                                  }
                    torch.save(checkpoint,os.path.join(Folddir,'BestcheckPoint.pth.tar'))
                                       
                    val_test(val_loader, model, criterion, device,w_frames_loss, save_path=path_save_Best, verbose=verbose)
                if SaveEach!=None and epoch%SaveEach ==0:
                        val_test(val_loader, model, criterion, device,w_frames_loss, verbose=verbose,save_pred_path=Folddir,epochid=epoch)
                        
                        
            writer.close()
            # Let's save the pred test images
            train_dices.append(current_train_dice)
            val_dices.append(current_val_dice)
            val_test(val_loader, model, criterion, device,w_frames_loss, save_path=Folddir, verbose=verbose)
            
            if st:
                # os.path.join(args.output_path,'Fold'+str(counter)+'Pred_train')
                path_save_train = os.path.join(args.output_path,'Fold'+str(counter),'Pred_train')
                if not os.path.isdir(path_save_train):
                   os.makedirs(path_save_train)
                val_test(train_loader, model, criterion, device,w_frames_loss, save_path=path_save_train, verbose=verbose)
                   
            # Let's save the model on the HDD
                
            # Where to save the model
            model_save_path = os.path.join(Folddir, 'Last_State_dict_model_' + model_to_train + '.pth.tar')
            # Save
                        
            checkpoint = {'Parameters': Parameters,# Experiment parameters
                          'state_dict': model.state_dict(),# Model layers and weights
                          'optimizer': optimizer.state_dict(),# Optimizer
                          'loss_train': current_train_loss,# Average loss history in training set each epoch
                          'loss_val': current_val_loss,# Average loss history in validation set each epoch
                          'Metric_train':current_train_dice,
                          'Metric_val':current_val_dice,
                          'Batch': args.batch_size,
                          'epoch':epoch,
                          'Metric':loss_name,
                          'TrainLoss': trainLoss,
                          'ValLoss': ValLoss
                                  }
            torch.save(checkpoint,model_save_path)
        # Print results
        print("RESULTS")
        print("train_dice", str(round(np.mean(np.array(train_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(train_dices)),2)))
        print("val_dice", str(round(np.mean(np.array(val_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(val_dices)),2)))


if __name__ == '__main__':
    t = time.time()
    main()
    elapsed = time.time() - t  
print('Time All = ',(elapsed/60)/60,' Hours')  




