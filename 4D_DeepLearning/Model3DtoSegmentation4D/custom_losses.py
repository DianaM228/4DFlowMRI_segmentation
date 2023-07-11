# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import time
from scipy.spatial import Delaunay
from itertools import combinations
from topologylayer.functional.persistence import SimplicialComplex
from topologylayer.nn import LevelSetLayer, TopKBarcodeLengths, SumBarcodeLengths, PartialSumBarcodeLengths

#
# Jaccard index
#
def jaccard_index(pred, target):

    pflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (pflat * tflat).sum()
    union = pflat.sum() + tflat.sum() - intersection + 1e-6 # to be sure
    jaccard = intersection / union

    return jaccard


#
# Jaccard loss function
#
class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        jaccard_loss:1-jaccard
        """
        jaccard_loss = 1 - jaccard_index(pred, target)

        return jaccard_loss


#################################################################################
#                                                                               #
# With the version 2 – a Jaccard is calculated for each image and then averaged #
#                                                                               #
#################################################################################
#
# Jaccard index 2
#
def jaccard_index2(pred, target, reduction='mean'):
    ## DEBUG
    #for i in range(pred.shape[0]):
    #    current_input = pred[i, 0, :, :, :]
    #    current_input = torch.squeeze(current_input)
    #    current_input = current_input.cpu().detach().numpy()
    #    current_input = nib.Nifti1Image(current_input, np.eye(4))
    #    nib.save(current_input, os.path.join("out", 'InputTest_' + str(i) + str(i) + '.nii.gz'))
    #    current_target = target[i, 0, :, :, :]
    #    current_target = torch.squeeze(current_target)
    #    current_target = current_target.cpu().detach().numpy()
    #    current_target = nib.Nifti1Image(current_target, np.eye(4))
    #    nib.save(current_target, os.path.join("out", 'TargetTest_' + str(i) + str(i) + '.nii.gz'))
    ## DEBUG
    pflat = pred.view(pred.shape[0], -1)
    tflat = target.view(target.shape[0], -1)
    intersection = torch.sum(torch.mul(pflat, tflat), dim=1)
    union = torch.sum(pflat, dim=1) + torch.sum(tflat, dim=1) - intersection + 1e-6

    jaccards = intersection / union

    if reduction == 'mean':
        return jaccards.mean()
    elif reduction == 'sum':
        return jaccards.sum()
    elif reduction == 'none':
        return jaccards
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


#
# Jaccard loss function 2
#
class JaccardLoss2(nn.Module):
    def __init__(self, reduction='mean'):
        super(JaccardLoss2, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        jaccard_loss2: 1 - jaccard
        """
        assert pred.shape[0] == target.shape[0], "predict & target batch size do not match"
        jaccard_loss2 = 1 - jaccard_index2(pred, target, reduction=self.reduction)

        return jaccard_loss2


#
# Dice coefficient
#
def dice_coefficient(pred, target, smooth=0):

    pflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (pflat * tflat).sum()
    dice = (2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth + 1e-6)

    return dice


#
# Dice loss function
#
class DiceLoss(nn.Module):
    def __init__(self, smooth=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        dice_loss:1-dice
        """
        dice_loss = 1 - dice_coefficient(pred, target, smooth=self.smooth)

        return dice_loss


##############################################################################
#                                                                            #
# With the version 2 – a dice is calculated for each image and then averaged #
#                                                                            #
##############################################################################
#
# Dice coefficient 2
#
def dice_coefficient2(pred, target, smooth=0, reduction='mean'):
    """
        Computes dice coefficient given  a multi channel input and a multi channel target
        Assumes the input is a normalized probability, a result of a Softmax function.
        Assumes that the channel 0 in the input is the background
        Args:
             pred (torch.Tensor): NxCxSpatial pred tensor
             target (torch.Tensor): NxCxSpatial target tensor
        """
    ## DEBUG
    #for i in range(pred.shape[0]):
    #    current_input = pred[i, 0, :, :, :]
    #    current_input = torch.squeeze(current_input)
    #    current_input = current_input.cpu().detach().numpy()
    #    current_input = nib.Nifti1Image(current_input, np.eye(4))
    #    nib.save(current_input, os.path.join("out", 'InputTest_' + str(i) + str(i) + '.nii.gz'))
    #    current_target = target[i, 0, :, :, :]
    #    current_target = torch.squeeze(current_target)
    #    current_target = current_target.cpu().detach().numpy()
    #    current_target = nib.Nifti1Image(current_target, np.eye(4))
    #    nib.save(current_target, os.path.join("out", 'TargetTest_' + str(i) + str(i) + '.nii.gz'))
    ## DEBUG
    pflat = pred.view(pred.shape[0], pred.shape[1], -1)
    tflat = target.view(target.shape[0], target.shape[1], -1)
    num = torch.sum(torch.mul(pflat, tflat), dim=2) + smooth
    den = torch.sum(pflat, dim=2) + torch.sum(tflat, dim=2) + smooth + 1e-6

    dices = 2 * num / den
    # We do not want to take into account the background
    dices = dices[:, 1:]
    dices = dices.mean(dim=1)

    if reduction == 'mean':
        return dices.mean()
    elif reduction == 'sum':
        return dices.sum()
    elif reduction == 'none':
        return dices
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


#
# Dice loss function 2
#
class DiceLoss2(nn.Module):
    def __init__(self, smooth=0, reduction='mean'):
        super(DiceLoss2, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        dice_loss2:1-dice
        """
        assert pred.shape[0] == target.shape[0], "predict & target batch size do not match"
        dice_loss2 = 1 - dice_coefficient2(pred, target, smooth=self.smooth, reduction=self.reduction)

        return dice_loss2


class DiceBCELoss2(nn.Module):
    def __init__(self, smooth=0, reduction='mean'):
        super(DiceBCELoss2, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        dice_loss2 = 1 - dice_coefficient2(pred, target, smooth=self.smooth, reduction=self.reduction)
        bce_loss = F.binary_cross_entropy(pred, target, reduction=self.reduction)
        dice_bce = bce_loss + dice_loss2

        return dice_bce


#
# generalized_dice (see https://arxiv.org/pdf/1707.03237.pdf)
#
def generalized_dice2(pred, target, smooth=0, reduction='mean'):
    """
        Computes dice coefficient given  a multi channel input and a multi channel target
        Assumes the input is a normalized probability, a result of a Softmax function.
        Assumes that the channel 0 in the input is the background
        Args:
             pred (torch.Tensor): NxCxSpatial pred tensor
             target (torch.Tensor): NxCxSpatial target tensor
        """
    pflat = pred.view(pred.shape[0], pred.shape[1], -1)
    tflat = target.view(target.shape[0], target.shape[1], -1)
    tflat_sum = tflat.sum(dim=2)
    w_tflat = 1/(torch.mul(tflat_sum,tflat_sum)).clamp(min=1e-6)
    w_tflat.requires_grad = False
    num = w_tflat*torch.sum(torch.mul(pflat, tflat), dim=2) + smooth
    den = w_tflat*(torch.sum(pflat+tflat, dim=2)) + smooth + 1e-6
    # We do not want to take into account the background
    num = num[:, 1:]
    den = den[:, 1:]
    dices = 2 * num.sum(dim=1) / den.sum(dim=1)

    if reduction == 'mean':
        return dices.mean()
    elif reduction == 'sum':
        return dices.sum()
    elif reduction == 'none':
        return dices
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


#
# Generalized Dice loss function
#
class GeneralizedDiceLoss2(nn.Module):
    def __init__(self, smooth=0, reduction='mean'):
        super(GeneralizedDiceLoss2, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        generalized_dice_loss:1-generalized_dice
        """
        assert pred.shape[0] == target.shape[0], "predict & target batch size do not match"
        generalized_dice_loss2 = 1 - generalized_dice2(pred, target, smooth=self.smooth, reduction=self.reduction)
        return generalized_dice_loss2

##############################################################################
#                                                                            #
# Compute Topology on data to include it in the loss function                #
#                                                                            #
##############################################################################
#

def Homology3D(pred, dgminfo=None, reduction='mean'):
    """
        Computes the homology for each image in the batch and gives the average value 
        (H0 = Conected components, H1 = Holes or loops, H2 = hollow voids)
        Paper: A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology (James R. Clough)
        """
    H_ao = {0:1, 1:0, 2:0}  ## Aorta topology
    H0 =0
    H1 = 0
    H2 = 0    
    for im in range(len(pred)):
        imagen = pred[im,1,:,:,:] ## [z,y,x ]Take image im from batch and only probability to be 1 = aorta
        imagenH = F.pad(imagen,(1,1,1,1,1,1))
        
        print('Computing Homology in batch image ',im)
        thomo=time.time()
        a = dgminfo(imagenH)
        elapsedhomo = time.time() - thomo 
        print('Time  homology= ',(elapsedhomo/60),' minutes')
        
        H0 += (SumBarcodeLengths(dim=0)(a))
        # 1-dimensional topological features           
        H1 += PartialSumBarcodeLengths(dim=1, skip=H_ao[1])(a) ## sum bars lengths skipping the first skip finite bars
        
        # 2-dimensional topological features           
        H2 += PartialSumBarcodeLengths(dim=2, skip=H_ao[2])(a)


    if reduction == 'mean':
        return (H0+H1+H2)/len(pred)
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


#
# Generalized Dice loss function
#
class Dice2TopoLoss(nn.Module):
    def __init__(self, smooth=0, reduction='mean'):
        super(Dice2TopoLoss, self).__init__()
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target, dgminfo):        
     
        assert pred.shape[0] == target.shape[0], "predict & target batch size do not match"
        dice_loss2 = 1 - dice_coefficient2(pred, target, smooth=self.smooth, reduction=self.reduction)
        print('Metric loss: ',dice_loss2)
        topo_loss = Homology3D(pred,dgminfo,reduction=self.reduction)
        print('Topo loss: ',topo_loss)
        return dice_loss2+(topo_loss*0.0008)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
