# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:49:31 2022

@author: Diana_MARIN
"""

import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import scipy.ndimage
import os
import nibabel as nib
from natsort import natsorted
import pandas as pd
import numpy as np
from skimage.measure import label


class ToTensor(object):
    """Convert nifty ndarrays to Tensors."""

    def __call__(self, sample):
        image, mask, affine = sample['image'], sample['mask'], sample['affine']
        shape_image = image.get_data().shape
        shape_mask = mask.get_data().shape
        np_img = image.get_data().reshape(1,shape_image[0],shape_image[1],shape_image[2],shape_image[3])
        np_mask = mask.get_data().reshape(1,shape_mask[0],shape_mask[1],shape_mask[2],shape_mask[3])
        torch_image = torch.from_numpy(np_img).float()
        torch_image = torch_image.permute(0, 4, 3, 2, 1)
        torch_mask = torch.from_numpy(np_mask).float()
        torch_mask = torch_mask.permute(0, 4, 3, 2, 1)
        return {'image': torch_image,
                'mask':  torch_mask,
                'affine': torch.from_numpy(affine).float()}


class Normalize(object):
    """Normalize the Tensors – mean-std nomalization."""

    def __init__(self, standard_deviation, mean):
        self.standard_deviation = standard_deviation
        self.mean = mean

    def __call__(self, sample):
        image, mask, affine = sample['image'], sample['mask'], sample['affine']
        #current_tensor = torch.div(current_tensor, self.max_value)
        image = torch.add(image, -self.mean)
        image = torch.div(image, self.standard_deviation)
        return {'image': image,
                'mask': mask,
                'affine': affine}


class NormToTensor(object):
    """Normalize + convert nifty ndarrays to Tensors."""

    def __init__(self, div_value, num_organs):
        self.div_value = div_value
        self.num_organs = num_organs

    def __call__(self, sample):
        image, mask, affine, name, time_id = sample['image'], sample['mask'], sample['affine'], sample['name'],sample['time_id']
        np_img = image.get_fdata()
        np_mask = mask.get_data()
        shape_np_img = np_img.shape
        shape_np_mask = np_mask.shape
        np_img = np.divide(np_img, self.div_value, dtype=np.float32)        
        np_img = np_img.reshape(1,shape_np_img[0],shape_np_img[1],shape_np_img[2], shape_np_img[3]) # Add axis
        torch_image = torch.from_numpy(np_img).float() # convert numoy array to tensor
        torch_image = torch_image.permute(0, 4, 3, 2, 1) # invert shape, free axis, slices, ...
        num_classes = self.num_organs + 1
        new_shape_np_mask = shape_np_mask + (num_classes,)
        new_np_mask = np.zeros(shape=new_shape_np_mask, dtype=np.uint8)
        for i in range(0, num_classes):
            new_np_mask[:, :, :, :, i] = np.where(np_mask[:, :, :, :] == i, 1, 0)
        torch_mask = torch.from_numpy(new_np_mask).float()
        torch_mask = torch_mask.permute(4, 3, 2, 1, 0)
        return {'image': torch_image,
                'mask':  torch_mask,
                'affine': torch.from_numpy(affine).float(),
                'name':name,
                'time_id': torch.from_numpy(time_id)}


class Resampling(object):

    def __call__(self, sample, resampling_size):
        image, mask, affine = sample['image'], sample['mask'], sample['affine']

        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)
        image = image.permute(0,1,5,4,3,2)
        mask = mask.permute(0,1,5,4,3,2)
        resample_image = F.interpolate(image, resampling_size, mode='trilinear', align_corners=True)
        resample_mask = F.interpolate(mask, resampling_size, mode='nearest')

        image = torch.squeeze(resample_image, 1)
        mask = torch.squeeze(resample_mask, 1)

        return {'image': image, 'mask': mask, 'affine': affine}

def NumInString(string):
    List = [ i for i,x in enumerate(string) if x.isdigit() == bool('True')]    
    name = string[List[0]:List[-1]+1]
    return name, List[0]


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(
        self, optimizer, patience=50, min_lr=1e-4, factor=0.1):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):        
        self.lr_scheduler.step(val_loss)
        
        
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs considering a porcentage of the best previous value.
    """
    def __init__(self, patience=110, min_delta=0.01):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        # elif (self.best_loss - val_loss) > self.min_delta:
        elif val_loss >= ( self.best_loss + self.best_loss*self.min_delta) :
            self.best_loss = val_loss
            self.counter = 0
        # elif (self.best_loss - val_loss) <= self.min_delta:
        elif val_loss < ( self.best_loss + self.best_loss*self.min_delta) :
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class AortaDataset(Dataset):
    """Tumour liver dataset."""

    def __init__(self, database_path, list_ids,path_id, transform=None, num_organs=1, norm_value=1):
        """
        Args:
            database_path (string): Directory where is the image database
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.database_path = database_path
        self.list_ids = list_ids
        self.norm_value = norm_value
        self.transform = transform
        self.num_organs = num_organs
        self.img_files = []
        self.mask_files = []
        self.Images_Names = []
        
        
        for i in self.list_ids:
            self.img_files.append(path_id[i])
            basname = os.path.splitext(os.path.basename(path_id[i]))[0]
            keyname,namestr = NumInString(basname)            
            self.mask_files.append(os.path.join(self.database_path,'Masks', basname[0:namestr]+keyname+'.nii.gz'))
            self.Images_Names.append(basname[0:namestr]+keyname+'.nii.gz')

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        Image_Name = self.Images_Names[index]

        # load images - image + mask
        image = nib.load(img_path)
        mask = nib.load(mask_path)
        # Let's retrieve the labeled_time_points
        time_id = []
        for i in range(mask.shape[3]): ##frames
            if(np.sum(mask.get_data()[:, :, :, i]) > 0):
                time_id.append(i)
        time_id = np.array(time_id, dtype=np.uint8)

        if self.transform:
            augmentation_struct = {'transformation_matrix': np.identity(4, dtype=np.float32),
                                   'noise_image': np.zeros(image.shape, dtype=np.float32),
                                   'shape_image': image.shape}
            # BE CAREFULL : we need to compute the inverse transfom!
            augmentation_struct = self.transform(augmentation_struct) ## get Tansformation matrix (after all tansformations)
            
            
            # apply compose transformations to image and mask            
            image_data = scipy.ndimage.affine_transform(image.get_data(),
                                                        np.linalg.inv(augmentation_struct['transformation_matrix']))
            mask_data = scipy.ndimage.affine_transform(mask.get_data(),
                                                       np.linalg.inv(augmentation_struct['transformation_matrix']),
                                                       order=0)
           
            image = nib.Nifti1Image(image_data, affine=image.affine)
            mask = nib.Nifti1Image(mask_data, affine=mask.affine)
          
        sample = {'image': image, 'mask': mask, 'affine': image.affine,'name':Image_Name, 'time_id': time_id}
        # Normalise the sample
        norm = NormToTensor(self.norm_value, self.num_organs)
        sample = norm(sample)
        # Return the sample
        return sample

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    try:
        largest=max(list_seg, key=lambda x:x[1])[0]
        out=(labels == largest).astype(int)
    except:
        out = segmentation        
    return out
    
    
    
    
    
    