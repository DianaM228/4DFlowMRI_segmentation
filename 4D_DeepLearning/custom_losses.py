# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:57:47 2022

@author: Diana_MARIN
"""

import torch.nn as nn
import torch
import torch.nn.functional as F



##############################################################################
#                                                                            #
#                            4D Deep learning                                #
#                                                                            #
##############################################################################
#

def soft_dice_coefficient2(pred, target,time_id, smooth=0):

    B,C,t,d,y,x = pred.shape    
    
    TwoTimes4DTensor_Pred = torch.empty((B,C,time_id.shape[1],d,y,x))
    TwoTimes4DTensor_Targ = torch.empty((B,C,time_id.shape[1],d,y,x))    
    
    for b,lab_times in enumerate(time_id): # batch index and labeled frames info (patient)
        for phase,f in enumerate(lab_times.squeeze().T): ## positions to fill in new pred and target tensors
            
            TwoTimes4DTensor_Pred[b,:,phase,:,:,:] = pred[b,:,f.item(),:,:,:]
            TwoTimes4DTensor_Targ[b,:,phase,:,:,:] = target[b,:,f.item(),:,:,:]
            
    pflat = TwoTimes4DTensor_Pred.view(TwoTimes4DTensor_Pred.shape[0],TwoTimes4DTensor_Pred.shape[1],-1)
    tflat = TwoTimes4DTensor_Targ.view(TwoTimes4DTensor_Targ.shape[0],TwoTimes4DTensor_Targ.shape[1],-1)
    num = torch.sum(torch.mul(pflat, tflat), dim=2) + smooth
            
    tflat=torch.pow(tflat,2 )
    pflat=torch.pow(pflat,2 )
        
    den = torch.sum( pflat , dim=2) + torch.sum( tflat , dim=2) + smooth + 1e-6 
    dices = (2 * num) / den
    dices = dices[:, 1:]
    dice = dices.mean() # mean batch
    return dice




def norm2_diff_images(img1, img2, reduction='mean'):
    img1 = img1.view(img1.shape[0], img1.shape[1], -1)
    img2 = img2.view(img2.shape[0], img2.shape[1], -1)
    diff=torch.sub(img1, img2)
    norm2_diff = torch.sum(torch.pow(diff, 2), dim=2)/img1.shape[2]  # img2 has the same size as img1
    # We do not want to take into account the background
    norm2_diff = norm2_diff[:, 1:]
    norm2_diff = norm2_diff.mean(dim=1)

    if reduction == 'mean':
        return norm2_diff.mean()
    elif reduction == 'sum':
        return norm2_diff.sum()
    elif reduction == 'none':
        return norm2_diff
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


class SparseDiceLoss24D(nn.Module):
    def __init__(self, smooth=0, reduction='mean'):
        super(SparseDiceLoss24D, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target, time_id,w_frames_loss):
        nb_time_points = target.shape[2]
        first_term = 0
        second_term = 0
        for i in range(nb_time_points):
            current_pred3D = pred[:, :, i, :, :, :]            
            first_term += 1 - soft_dice_coefficient2(pred, target,time_id)
            if i < (nb_time_points - 1):
                current_pred3D_2 = pred[:, :, i+1, :, :, :]
                second_term += norm2_diff_images(current_pred3D_2, current_pred3D)
       
        w_metric = 1-w_frames_loss
        loss = (first_term/time_id.shape[1])*w_metric + (second_term/(nb_time_points - 1))*w_frames_loss

        return loss
