# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:21:21 2022

@author: Diana_MARIN
"""

import os
import time
import torch
import shutil
import nibabel as nib
from custom_losses import soft_dice_coefficient2
import numpy as np


def patien_index_finder(name):
    
    temp_string=name.split("Patient")
    temp_string=temp_string[1].split(".nii")
    
    patient_index=int(temp_string[0])

    return patient_index



#https://github.com/pytorch/examples/blob/b9f3b2ebb9464959bdbf0c3ac77124a704954828/imagenet/main.py#L359
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


#https://github.com/pytorch/examples/blob/b9f3b2ebb9464959bdbf0c3ac77124a704954828/imagenet/main.py#L359
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_images(input, target, pred, affine, count, batch_size, save_path,names):
    
    for i in range(batch_size):
        current_affine = affine[i, :, :]
        current_affine = current_affine.cpu().detach().numpy()
        current_input = input[i, 0, :, :, :, :]
        current_input = current_input.permute(3, 2, 1, 0)
        current_input = torch.squeeze(current_input)
        current_input = current_input.cpu().detach().numpy()
        current_input = nib.Nifti1Image(current_input, affine=current_affine)
        nib.save(current_input, os.path.join(save_path, 'InputTestNorm_' + str(count)+'_' + names[i] ))
        # Assume that 0 is the background
        current_target = torch.argmax(target[i, :, :, :, :, :], dim=0).float()
        current_target = current_target.permute(3, 2, 1, 0)
        current_target = torch.squeeze(current_target)
        current_target = current_target.cpu().detach().numpy()
        current_target = nib.Nifti1Image(current_target, affine=current_affine)
        nib.save(current_target, os.path.join(save_path, 'TargetTest_' + str(count)+'_' + names[i]))
        # Assume that 0 is the background
        current_pred = torch.argmax(pred[i, :, :, :, :, :], dim=0).float()
        current_pred = current_pred.permute(3, 2, 1, 0)
        current_pred = torch.squeeze(current_pred)
        current_pred = current_pred.cpu().detach().numpy()
        current_pred = nib.Nifti1Image(current_pred, affine=current_affine)
        nib.save(current_pred, os.path.join(save_path, 'PredTest_' + str(count)+'_' + names[i] ))
        
        ####### Save probability Map
        probability_map = pred[i, 1, :, :, :, :]
        probability_map = probability_map.permute(3, 2, 1, 0)
        probability_map = torch.squeeze(probability_map)
        probability_map = probability_map.cpu().detach().numpy()
        probability_map = nib.Nifti1Image(probability_map, affine=current_affine)
        nib.save(probability_map, os.path.join(save_path,'Map_PredTest_' + str(count)+'_' + names[i] ))

def save_pred(input, pred, affine, batch_size, save_path,epoch):
    save_path = os.path.join(save_path, 'PredictionsEpochs')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for i in range(batch_size):
        current_affine = affine[i, :, :]
        current_affine = current_affine.cpu().detach().numpy()    
        probability_map = pred[i, 1, :, :, :, :]
        probability_map = probability_map.permute(3, 2, 1, 0)
        probability_map = torch.squeeze(probability_map)
        probability_map = probability_map.cpu().detach().numpy()
        probability_map = nib.Nifti1Image(probability_map, affine=current_affine)
        nib.save(probability_map, os.path.join(save_path,'Map'+str(epoch)+'.nii'))
        
        current_pred = torch.argmax(pred[i, :, :, :, :, :], dim=0).float()
        current_pred = current_pred.permute(3, 2, 1, 0)
        current_pred = torch.squeeze(current_pred)
        current_pred = current_pred.cpu().detach().numpy()
        current_pred = nib.Nifti1Image(current_pred, affine=current_affine)
        nib.save(current_pred, os.path.join(save_path,'Pred'+str(epoch)+'.nii'))


def train(train_loader, model, criterion, optimizer, epoch, device, w_frames_loss=None,verbose=False):
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Dice', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dices],
        prefix="Train / Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()
    
   
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target, _, _, time_id = data['image'], data['mask'], data['affine'],data['name'], data['time_id']
         
     
        # send data to device
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)

        # measure dice
        # dices.update(soft_dice_coefficient2(output, target, time_id).item(), input.size(0))
        dices.update(soft_dice_coefficient2(output, target, time_id).item())


        # update loss/opti
        loss = criterion(output, target, time_id,w_frames_loss)
        #losses.update(loss.item(), input.size(0)) ##########
        losses.update(loss.item())

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       
        if verbose:
            progress.display(i)
            
    return losses.avg, dices.avg


def val_test(val_test_loader, model, criterion, device,w_frames_loss=None, save_path=None, verbose=False,save_pred_path=None, epochid=None):
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Dice', ':.4e')
    progress = ProgressMeter(
        len(val_test_loader),
        [batch_time, data_time, losses, dices],
        prefix="Val / Test:")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
               
        for i, data in enumerate(val_test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input, target, affine,name, time_id = data['image'], data['mask'], data['affine'],data['name'], data['time_id']
                        
            # send data to device
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)

            # Save the images
            if save_path:
                #names=val_test_loader.dataset.Images_Names                                
                save_images(input, target, output, affine, i, input.size(0), save_path,name)
                
            if save_pred_path:
                save_pred(input, output, affine, input.size(0), save_pred_path,epoch=epochid)
                                

            # measure dice
            dices.update(soft_dice_coefficient2(output, target, time_id).item(), input.size(0))

            # update loss
            loss = criterion(output, target, time_id,w_frames_loss)
            losses.update(loss.item())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics
            if verbose:
                progress.display(i)
    #
    return losses.avg, dices.avg
