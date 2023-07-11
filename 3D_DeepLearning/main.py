# -*- coding:utf-8 -*-
from UtilsDL import RandomTranslationZ
from UtilsDL import RandomTranslationY
from UtilsDL import RandomTranslationX
from UtilsDL import RandomUniformScaling
from UtilsDL import AortaDataset
from UtilsDL import EarlyStopping, LRScheduler
from UtilsDL import init_tri_complex3D
from train_val_test import train
from train_val_test import val_test
from UNetNetworks import UNet3DAntoine
from UNetNetworks import UNet3DAntoine2
from UNetNetworks import weights_init
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from custom_losses import DiceLoss
from custom_losses import DiceLoss2
from custom_losses import DiceBCELoss2
from custom_losses import JaccardLoss
from custom_losses import JaccardLoss2
from custom_losses import GeneralizedDiceLoss2
from custom_losses import Dice2TopoLoss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse
import glob
import os
from torch.utils.tensorboard import SummaryWriter
import sys
import time
import random
from natsort import natsorted
from operator import itemgetter 
import nibabel as nib
from topologylayer.nn import LevelSetLayer
from torchsummary import summary


def main():
    parser = argparse.ArgumentParser(description='main 3D UNet like networks')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('--scaling', help='random uniform scaling', action='store_true')
    required_named_args = parser.add_argument_group('required named arguments')
    required_named_args.add_argument('-i', '--input_path', help='path to the directory where is the image database',required=True)
    required_named_args.add_argument('--num_organs', type=int, help='number of organs to segment (1 = tumour, 2 = tumour + liver)',required=True)    
    required_named_args.add_argument('-n', '--norm_value', type=float, help='normalization value', required=True)
    required_named_args.add_argument('-m', '--model', help='model to train (UNet3DAntoine, UNet3DAntoine2)',required=True)
    required_named_args.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)
    required_named_args.add_argument('-e', '--epoch', type=int, help='number of epoch', required=True)
    required_named_args.add_argument('-l', '--learning_rate', type=float, help='learning rate', required=True)
    required_named_args.add_argument('-c', '--criterion', help='loss function (Dice, Dice2, DiceBCE2, Jaccard, Jaccard2, GeneralizedDice2,Dice2Topo)', required=True)
    required_named_args.add_argument('-o', '--output_path', help='output path', required=True)
   
    optional_named_args = parser.add_argument_group('optional named arguments')
    optional_named_args.add_argument('-s', '--smooth', type=float, help='smooth parameter for Dice type losses',required=False, default=0)
    optional_named_args.add_argument('-da', '--data_augmentation', help='increase dataset using multiple data augmentation strategies', 
                                     action='store_true',required=False, default=None)
    optional_named_args.add_argument('-seed', '--Seed_pytorch', help='set the same seed for experiments', 
                                     action='store_true',required=False, default=None)
    optional_named_args.add_argument('-cluster', '--Cluster', help='Option to work in gpu, by default CPU', 
                                     action='store_true',required=False, default=None)
    
    optional_named_args.add_argument('-lr_schedule', '--lr_schedule', dest='lr_schedule',help='Train the model with Plateau lr schedule ',action='store_true', default=None)
    optional_named_args.add_argument('-early', '--early_stop', dest='early_stop', help='Train the model with early stop options',action='store_true', default=None)
    optional_named_args.add_argument('-ev', '--Evaluation_Method', help='Select cross-validation or leave-one-out evaluation, by default 90/10 split', choices=['cv','loo','personalized'], default=None)
    optional_named_args.add_argument('-f', '--Folds', help='Select folds number for cross validation',type=int,default=None)
    optional_named_args.add_argument('-p', '--Path_indices', help='If select personalized evaluation, give path to dictionary with indices (data number influence test)',default=None)
    optional_named_args.add_argument('-t', '--test', help='Option to save data for test before split data in Train-val ',action='store_true',default=None)
    optional_named_args.add_argument('--fold_id', type=int, help='fold to run',required=False, default=0)

       
    #read inmputs
    args = parser.parse_args()
    verbose = args.verbose
    scaling = args.scaling
    num_organs = args.num_organs
    # Hyperparameters
    model_to_train = args.model
    batch_size = args.batch_size
    nb_epochs = args.epoch
    learning_rate = args.learning_rate
    loss_name = args.criterion
    smooth_value = args.smooth
    Data_augm = args.data_augmentation
    norm_value = args.norm_value
    fold_id = args.fold_id
    
    if args.Evaluation_Method=='cv' and args.Folds is None:
        parser.error("Select the number of folds for cross validation ")
        
    if args.Evaluation_Method=='personalized' and args.Path_indices is None:
        parser.error("Enter the path to the dictionary with the personalized index for validation")
    
    
    print('LR schedule activated? ',args.lr_schedule)
    print('early stopping activated? ',args.early_stop)
    print('Cluster', args.Cluster)
    
    ## Some information to save check point
    Parameters={'Path_dataset':args.input_path,
                'Path_results':args.output_path,
                'Model':args.model}
     
    if verbose:
        print('verbosity is turned on')
    
    #seed for reproducibility
    if args.Seed_pytorch:        
        seed = 10
        torch.manual_seed(seed)  #seed PyTorch random number generator
        torch.cuda.manual_seed_all(seed)
                 
    ## Split data considering the validation selected    
    Path_images=natsorted(glob.glob(os.path.join(args.input_path,'Images','*.nii')))
    
    if args.test:
        path_id_test = Path_images
        trainVal_id, test_id = train_test_split(np.arange(len(Path_images)), test_size=0.166, random_state=42)  
        print('TEST IMAGES','\n',list(itemgetter(*test_id)(path_id_test)))
        Path_images=list(itemgetter(*trainVal_id)(path_id_test))
              
              
    print('TRAIN-VAL IMAGES','\n',Path_images)
    ### personalized    
    if args.Evaluation_Method != None:
        if args.Evaluation_Method == 'personalized':
            print('Personalized Validation') 
            Index = np.load(args.Path_indices,allow_pickle='TRUE').item()
            val_id = Index['Val'][0]
            
            trainI=[]
            train_dices = []
            val_dices = []
            for nf,f in enumerate(Index['Train']): ##### Fold in Train
                trainI.append(f)                
                train_id = np.concatenate(trainI)
                print('TEST ',nf)
                print('Train inxed ',train_id)
                print('Valinxed ',val_id)
                # Net
                model = -1 
                if model_to_train == 'UNet3DAntoine':
                    model = UNet3DAntoine(in_channel=1, out_channel=num_organs+1) 
                    print('UNet3DAntoine')
                elif model_to_train == 'UNet3DAntoine2':
                    model = UNet3DAntoine2(in_channel=1, out_channel=num_organs+1)
                    print('UNet3DAntoine2')
                else:
                    print('The model does not exist')
                    sys.exit(1)
                
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
                assert loss_name in ['Dice' , 'Dice2', 'DiceBCE2', 'Jaccard', 'Jaccard2', 'GeneralizedDice2'], \
                    "The criterion does not exist / is not implemented yet"
                criterion = DiceLoss()
                if loss_name == 'Dice':
                    criterion = DiceLoss(smooth_value)
                elif loss_name == 'Dice2':
                    criterion = DiceLoss2(smooth_value)
                elif loss_name == 'DiceBCE2':
                    criterion = DiceBCELoss2(smooth_value)
                elif loss_name == 'Jaccard':
                    criterion = JaccardLoss()
                elif loss_name == 'Jaccard2':
                    criterion = JaccardLoss2()
                elif loss_name == 'GeneralizedDice2':
                    criterion = GeneralizedDiceLoss2(smooth_value)
                # Defining the optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
                ### Set lr_scheduler and early_stopping if you wanna use it
                if args.lr_schedule:    
                    lr_scheduler = LRScheduler(optimizer)
                if args.early_stop:
                    early_stopping = EarlyStopping()
                    
                #### Define data augmentation
                if Data_augm:
                    if scaling:            
                        data_augmentation = transforms.Compose([RandomUniformScaling(1, 1.5),  # >= 1                                                
                                                                RandomTranslationZ(-4, 4),  # pixels
                                                                RandomTranslationY(-4, 4),  # pixels
                                                                RandomTranslationX(-4, 4),  # pixels                       
                                                                ])
                    else:
                        data_augmentation = transforms.Compose([RandomTranslationZ(-4, 4), # pixels
                                                            RandomTranslationY(-4, 4), # pixels
                                                            RandomTranslationX(-4, 4), # pixels                                            
                                                            ])        
                else:
                    data_augmentation = None
             

                # Folder to save fold results
                Folddir = os.path.join(args.output_path,'Test'+str(nf)+'_'+str(len(train_id))+'P')
                if not os.path.isdir(Folddir):
                    os.makedirs(Folddir)
                    
                # default `log_dir` is "runs" - we'll be more specific here
                writer = SummaryWriter(log_dir=os.path.join(Folddir, 'runs'+"BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)),
                                       comment="BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate))
    
        
                    
                if verbose:
                    print("train+val:",'\n', Path_images)
                    
                train_dataset = AortaDataset(database_path=args.input_path, list_ids=train_id,path_id=Path_images, norm_value=norm_value,
                                                    transform=data_augmentation, num_organs=num_organs)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_dataset = AortaDataset(database_path=args.input_path, list_ids=val_id,path_id=Path_images, norm_value=norm_value,
                                                  transform=None, num_organs=num_organs)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                # train
                model.apply(weights_init)
                
                best_Metric = 0
                for epoch in range(nb_epochs):
                    if verbose:
                        print("current epoch:", str(epoch))
                    # train
                    current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device,verbose=verbose)
                    # val
                    current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, verbose=verbose)
                    # log the loss + dice
                    writer.add_scalars('loss/training-validation_loss_' + str(nf), {'training': current_train_loss,
                                                                                            'validation': current_val_loss}, epoch)
                    writer.add_scalars('dice/training-validation_dice_' + str(nf), {'training': current_train_dice,
                                                                                            'validation': current_val_dice}, epoch)
                    
                    if args.lr_schedule:
                       lr_scheduler(current_val_loss)
                    if args.early_stop:
                        early_stopping(current_val_loss)
                        if early_stopping.early_stop:                        
                            break
                    ## Save best model    
                    if current_val_dice>best_Metric:
                        best_Metric=current_val_dice
                        
                        checkpoint = {'Parameters': Parameters,# Experiment parameters
                                      'state_dict': model.state_dict(),# Model layers and weights
                                      'optimizer': optimizer.state_dict(),# Optimizer
                                      'loss_train': current_train_loss,# Average loss history in training set each epoch
                                      'loss_val': current_val_loss,# Average loss history in validation set each epoch
                                      'Metric_train':current_train_dice,
                                      'Metric_val':current_val_dice,
                                      'Batch': args.batch_size,
                                      'epoch':epoch,
                                      'Metric':loss_name
                                      }
                        torch.save(checkpoint,os.path.join(Folddir,'BestcheckPoint.pth.tar'))
                       
                
                writer.close()
                # Let's save the pred test images
                train_dices.append(current_train_dice)
                val_dices.append(current_val_dice)
                val_test(val_loader, model, criterion, device, save_path=Folddir, verbose=verbose)
                
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
                              'Metric':loss_name
                                      }
                torch.save(checkpoint,model_save_path)
            # Print results
            print("RESULTS")
            print("train_dice", str(round(np.mean(np.array(train_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(train_dices)),2)))
            print("val_dice", str(round(np.mean(np.array(val_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(val_dices)),2)))
            
 
        ### Validación cruzada o Leave-one-out    
        else:
            
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

  
            #######Model, optimizer and data augmentation
                
            # Net
            model = -1 
            if model_to_train == 'UNet3DAntoine':
                model = UNet3DAntoine(in_channel=1, out_channel=num_organs+1)  
                print('UNet3DAntoine')
            elif model_to_train == 'UNet3DAntoine2':
                model = UNet3DAntoine2(in_channel=1, out_channel=num_organs+1)
                print('UNet3DAntoine2')
            else:
                print('The model does not exist')
                sys.exit(1)
            
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
            assert loss_name in ['Dice' , 'Dice2', 'DiceBCE2', 'Jaccard', 'Jaccard2', 'GeneralizedDice2','Dice2Topo'], \
                "The criterion does not exist / is not implemented yet"
            criterion = DiceLoss()
            if loss_name == 'Dice':
                criterion = DiceLoss(smooth_value)
            elif loss_name == 'Dice2':
                criterion = DiceLoss2(smooth_value)
            elif loss_name == 'DiceBCE2':
                criterion = DiceBCELoss2(smooth_value)
            elif loss_name == 'Jaccard':
                criterion = JaccardLoss()
            elif loss_name == 'Jaccard2':
                criterion = JaccardLoss2()
            elif loss_name == 'GeneralizedDice2':
                criterion = GeneralizedDiceLoss2(smooth_value)
            elif loss_name == 'Dice2Topo':
                criterion1 = DiceLoss2(smooth_value)
                criterion = Dice2TopoLoss(smooth_value)
                print('Computing complex 3D')
                tcpx=time.time()
                #### load and image from the dataset to know the complex size 
                I = nib.load(Path_images[0])
                I_array=I.get_fdata()
                width, height, deep=np.shape(I_array)
                cpx = init_tri_complex3D(deep+2,height+2,width+2) #### Padding each side with zero line to avoid errors on homology computation 
                elapsedcxp = time.time() - tcpx 
                print('Time  complex= ',(elapsedcxp/60),' minutes')                    
                dgminfo = LevelSetLayer(cpx, maxdim=2, sublevel=False)
            # Defining the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
            ### Set lr_scheduler and early_stopping if you wanna use it
            if args.lr_schedule:    
                lr_scheduler = LRScheduler(optimizer)
            if args.early_stop:
                early_stopping = EarlyStopping()
                
            #### Define data augmentation
            if Data_augm:
                if scaling:            
                    data_augmentation = transforms.Compose([RandomUniformScaling(1, 1.5),  # >= 1                                                
                                                            RandomTranslationZ(-4, 4),  # pixels
                                                            RandomTranslationY(-4, 4),  # pixels
                                                            RandomTranslationX(-4, 4),  # pixels                                                
                                                            ])
                else:
                    data_augmentation = transforms.Compose([RandomTranslationZ(-4, 4), # pixels
                                                        RandomTranslationY(-4, 4), # pixels
                                                        RandomTranslationX(-4, 4), # pixels                                            
                                                        ])        
            else:
                data_augmentation = None
         

            # Folder to save fold results
            Folddir = os.path.join(args.output_path,'Fold'+str(counter))
            if not os.path.isdir(Folddir):
                os.makedirs(Folddir)
                
            # default `log_dir` is "runs" - we'll be more specific here
            writer = SummaryWriter(log_dir=os.path.join(Folddir, 'runs'+"BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate)),
                                   comment="BATCH_"+str(batch_size)+"_EPOCHS_"+str(nb_epochs)+"_LR_"+str(learning_rate))

    
                
            if verbose:
                print("train+val:",'\n', Path_images)
                #print("train:",'\n', np.array(Path_images)[train_id],'\n', "val:", np.array(Path_images)[val_id])
                
            train_dataset = AortaDataset(database_path=args.input_path, list_ids=train_id,path_id=Path_images, norm_value=norm_value,
                                                transform=data_augmentation, num_organs=num_organs)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_dataset = AortaDataset(database_path=args.input_path, list_ids=val_id,path_id=Path_images, norm_value=norm_value,
                                              transform=None, num_organs=num_organs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            if args.test:
                test_dataset = AortaDataset(database_path=args.input_path, list_ids=test_id,path_id=path_id_test, norm_value=norm_value,
                                              transform=None, num_organs=num_organs)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
                
            # train
            model.apply(weights_init)
            
            best_Metric = 0
            for epoch in range(nb_epochs):
                if verbose:
                    print("current epoch:", str(epoch))
                # train
                if loss_name == 'Dice2Topo':                        
                    if epoch+1 > nb_epochs-round(nb_epochs*0.12):                            
                        current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device,verbose=verbose,dgminfo=dgminfo)
                    else:
                        current_train_loss, current_train_dice = train(train_loader, model, criterion1, optimizer, epoch, device,verbose=verbose)
                                               
                
                else:                        
                    current_train_loss, current_train_dice = train(train_loader, model, criterion, optimizer, epoch, device,verbose=verbose)
                
                # val
                if loss_name == 'Dice2Topo':
                    if epoch+1 >  nb_epochs-round(nb_epochs*0.12):
                        current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, verbose=verbose,dgminfo=dgminfo)                    
                    else:
                        current_val_loss, current_val_dice = val_test(val_loader, model, criterion1, device, verbose=verbose)                    
                                        
                else: 
                    current_val_loss, current_val_dice = val_test(val_loader, model, criterion, device, verbose=verbose)
                # log the loss + dice                    
                if args.test:
                    current_test_loss, current_test_dice = val_test(test_loader, model, criterion, device, verbose=verbose)
                    writer.add_scalars('loss_cv/training-validation-test_loss_' + str(counter), {'training': current_train_loss,
                                                                                                 'validation': current_val_loss,
                                                                                                  'test': current_test_loss}, epoch)
                    writer.add_scalars('dice_cv/training-validation-test_dice_' + str(counter), {'training': current_train_dice,
                                                                                                 'validation': current_val_dice,
                                                                                                 'test': current_test_dice}, epoch)
                else:
                    writer.add_scalars('loss_cv/training-validation_loss_' + str(counter), {'training': current_train_loss,
                                                                                        'validation': current_val_loss}, epoch)
                    writer.add_scalars('dice_cv/training-validation_dice_' + str(counter), {'training': current_train_dice,
                                                                                        'validation': current_val_dice}, epoch)
                
                
                
                if args.lr_schedule:
                   lr_scheduler(current_val_loss)
                if args.early_stop:
                    early_stopping(current_val_loss)
                    if early_stopping.early_stop:                        
                        #val_test(val_loader, model, criterion, device, save_path=Folddir, verbose=verbose)
                        break
                ## Save best model    
                 
                if current_val_dice>best_Metric:
                    best_Metric=current_val_dice
                    
                    checkpoint = {'Parameters': Parameters,# Experiment parameters
                                  'state_dict': model.state_dict(),# Model layers and weights
                                  'optimizer': optimizer.state_dict(),# Optimizer
                                  'loss_train': current_train_loss,# Average loss history in training set each epoch
                                  'loss_val': current_val_loss,# Average loss history in validation set each epoch
                                  'Metric_train':current_train_dice,
                                  'Metric_val':current_val_dice,
                                  'Batch': args.batch_size,
                                  'epoch':epoch,
                                  'Metric':loss_name
                                  }
                    torch.save(checkpoint,os.path.join(Folddir,'BestcheckPoint.pth.tar'))
                   
            
            writer.close()
            # Let's save the pred test images
            train_dices.append(current_train_dice)
            val_dices.append(current_val_dice)
            if loss_name == 'Dice2Topo':                    
                val_test(val_loader, model, criterion, device, save_path=Folddir, verbose=verbose,dgminfo=dgminfo)                    
            else:
                val_test(val_loader, model, criterion, device, save_path=Folddir, verbose=verbose)                    
       
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
                          'Metric':loss_name
                                  }
            torch.save(checkpoint,model_save_path)
        # Print results
        print("RESULTS")
        print("train_dice", str(round(np.mean(np.array(train_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(train_dices)),2)))
        print("val_dice", str(round(np.mean(np.array(val_dices)),2)) + u"\u00B1" + str(round(np.std(np.array(val_dices)),2)))
 

if __name__ == '__main__':
    main()
