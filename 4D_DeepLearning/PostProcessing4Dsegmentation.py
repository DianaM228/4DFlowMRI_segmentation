# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:39:07 2022

This code save in a folder the postporcessed 4D segmentation (25 frames) for each patient. It takes the outputs from best model and it does resizing, 
main object identification frame by frame, and opening operation frame by frame.

-i 4D_DeepLearning\Results\Final\25Fram_lr0.0001_5lab_500e_wt0.8
-j 4D_DeepLearning\DataV4_Invert_shifted_timens_5LabeledFrames\Dataset4D_5LabTimes3Parts_146x176x44x25 
-k E:\4D_DeepLearning\ManualSegmentationsDiana\Diana_5_labels (needed for compliance)
-r 3
-t 0.1
@author: Diana_MARIN
"""

import argparse
import torch
import os
import shutil as sh
from os import listdir
import subprocess
import glob
import nibabel as nib
import numpy as np
from natsort import natsorted
import pandas as pd
from UtilsDL import getLargestCC
from importlib.machinery import SourceFileLoader
foo = SourceFileLoader("NumInString", r"C:\Users\Diana_MARIN\Documents\aortadiag4d\Utils.py").load_module()

parser = argparse.ArgumentParser(description='Function to format the 4D image size to the original one, apply appening over the 25 frames')
parser.add_argument('-i' ,'--input1', help='Path to Folder with the results of the experiment from 4D deep learning')
parser.add_argument('-j', '--input2',help='Path to the dataset with original size 4D to reformat predictions')
parser.add_argument('-k', '--input3',help='Path to manual segmentation')
parser.add_argument('-r','--radius', help='radius of the ball element for the opening operation', required=True)
parser.add_argument('-o','--output', help='Path to save folder with results', required=True)
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-t', '--threshold', help='If you want to work with the results from a desired threshod',default=None)

 
args = parser.parse_args()

path_results = args.input1
path_original = args.input2
path_orig3D = args.input3
radius = args.radius
out = args.output

if args.threshold:
    OutTest = os.path.join(out, 'PostprocessData_Test_'+os.path.basename(path_results)+'Threshols'+args.threshold+'R'+radius)
else:    
    OutTest = os.path.join(out, 'PostprocessData_Test_'+os.path.basename(path_results)+'_R'+radius)
    
if not os.path.isdir(OutTest):
        os.makedirs(OutTest)

###### folder to save temporal results

tmp = os.path.join(out,'tmp'+os.path.basename(path_results))
if not os.path.isdir(tmp):
        os.makedirs(tmp)


for ind,fold in enumerate(natsorted(listdir(path_results))): # fold or patient
    if os.path.isdir(os.path.join(path_results,fold)):        
        p_last = os.path.join(path_results,fold)
        
        if args.threshold:
            path_pred = os.path.join(path_results,fold,'Best_Model_Pred','Threshold_'+args.threshold+'_Post-Process_MainO')
            prediction =  glob.glob(path_pred+os.path.sep+"Pred_Threshold_*.nii.gz")[0]                        
                                    
        else:
            path_pred = os.path.join(path_results,fold,'Best_Model_Pred')
            prediction =  glob.glob(path_pred+os.path.sep+'PredTest*.nii.gz')[0]
        
        ##### Patient name 
        name = os.path.basename(prediction)
        name = name.split('Patient')[1]
        name = name.split('.')[0]
        
        ######## Resize 4D image 
        Orig_like = os.path.join(path_original,'Masks','Patient'+name+'.nii.gz')
        
        ##### Patient output folder
        OutPatient = os.path.join(OutTest,'Patient'+name)
        if not os.path.isdir(OutPatient):
            os.makedirs(OutPatient)
        
        #### Resize segmentation from best model for patient X and save it in tmp folder to input 4D network
        clitk = ['clitkAffineTransform.exe','-i', prediction, '-l', os.path.normpath(Orig_like), '-o', 
                 os.path.normpath(os.path.join(tmp,'Patient'+name+'.nii.gz')), '--interp', '0']
        subprocess.run(clitk) 
        
        
        ###### Main object on each 3D flrame 
        Image = nib.load(os.path.join(tmp,'Patient'+name+'.nii.gz')).get_fdata()        
        MT = nib.load(os.path.join(tmp,'Patient'+name+'.nii.gz')).affine
        
        ######## take only main object in each frame
        print(fold)
        
        #### reshape to 3D original size
        like3d = natsorted(glob.glob(path_orig3D+os.path.sep+'Patient'+name+os.path.sep+"*.nii.gz"))[0]
            
        
        framesToMerge = []    
        for frame in range(np.size(Image,3)):         
            result= getLargestCC(Image[:,:,:,frame])
            ### save 3D 
            out3dOpenBest = os.path.join(OutPatient,'Patient'+name+'_T'+str(frame)+'.nii.gz')
            Nii_Main3DPredB = nib.Nifti1Image(result,MT) 
            nib.save(Nii_Main3DPredB,out3dOpenBest)
            framesToMerge.append(os.path.normpath(out3dOpenBest))                    
            #### opening 3D 
            op = ['clitkMorphoMath.exe','-i',out3dOpenBest,'-o',out3dOpenBest,'-t','3','-r',radius]
            subprocess.run(op)
            
             #### reshape to 3D original size
            clitk = ['clitkAffineTransform.exe','-i', out3dOpenBest, '-l', os.path.normpath(like3d), '-o', 
            out3dOpenBest, '--interp', '0']
            subprocess.run(clitk) 
            
    
        
        ###### Merge to generate 4D segentation
        out4D = os.path.join(OutPatient,'Patient'+name+'_4Dsegmentation.nii')
        clitk_list = ['clitkMergeSequence.exe']+ framesToMerge+['-o',out4D]
        subprocess.run(clitk_list)
    
sh.rmtree(tmp, ignore_errors=True)
    
    
    