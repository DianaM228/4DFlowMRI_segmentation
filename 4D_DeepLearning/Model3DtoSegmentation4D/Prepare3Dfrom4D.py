# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:25:34 2022

@author: Diana_MARIN
"""

import argparse
import os
from os import listdir
import nibabel as nib
import shutil as sh
import glob
from natsort import natsorted
import subprocess
import shutil as sh
import numpy as np
import torch


parser = argparse.ArgumentParser(description='Function to convert nifti volumes to 4D, resize the images and generate the 4D mask ')
parser.add_argument('-i','--Path_4D_Data', help='Path to the Folder with dataset to run 4D deep learning')
parser.add_argument('-o','--Path_out', help='Path to the base folder save the results')


args = parser.parse_args()

datset = args.Path_4D_Data
out = args.Path_out

subjects = natsorted(glob.glob(os.path.join(datset,'Masks')+'/*.nii.gz'))

### create folders to save results 

    
tmp = os.path.join(out,'TemporalFor3D')
if not os.path.isdir(tmp):
    os.makedirs(tmp)


for p in subjects:
    p = os.path.normpath(p)
    image = nib.load(p)
    MT = nib.load(p).affine
    
    ### read labeled time frames
    time_frame = []
    for i in range(image.shape[3]): ##frames
        if(np.sum(image.get_data()[:, :, :, i]) > 0):
            time_frame.append(i)
            
    ##### split 4D images and mask into 3D
    patient = os.path.basename(p)
    patient = patient.split('.')[0]
    patient = patient.split('Patient')[1]
    
    ###############
    outIm = os.path.join(out, 'For3D_'+os.path.basename(datset),'Images','Patient'+str(patient))
    outMk = os.path.join(out, 'For3D_'+os.path.basename(datset),'Masks','Patient'+str(patient))
    
    if not os.path.isdir(outIm):
        os.makedirs(outIm)
    if not os.path.isdir(outMk):
        os.makedirs(outMk)
        
    clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmp,'Image'),'-d', '3', '--nii','-i',os.path.join(datset,'Images','Patient'+str(patient)+'.nii')]
    subprocess.run(clitkSplit)
    
    clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmp,'Mask'),'-d', '3', '--nii','-i',p]
    subprocess.run(clitkSplit)
    
    ##### copy only labeled images to the folder 
    for i in time_frame:
        if i < 10:
            InameIm = '0'+str(i)
        else:
            InameIm = str(i)
                        
        sh.copy(os.path.join(tmp,'Mask_'+InameIm+'.nii'), os.path.join(outMk,'P'+patient+'MASKT'+str(i)+'.nii'))
        sh.copy(os.path.join(tmp,'Image_'+InameIm+'.nii'), os.path.join(outIm,'P'+patient+'GRAYT'+str(i)+'.nii'))
            
    
sh.rmtree(tmp, ignore_errors=True)            
    
            
            
            
            
            
            
            
            
            
            
            
            