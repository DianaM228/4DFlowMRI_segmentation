# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:39:07 2022

This code allows the prediction of the complite cardiac cycle (25 frames) and postporcesses the 4D segmentation for each patient.  
resizing, main object identification frame by frame, and opening operation frame by frame.

-i E:\4D_DeepLearning\Results\Final\Results4Dwith3D_lr0.01_5lab_500e
-j 4D_DeepLearning\DataV4_Invert_shifted_timens_5LabeledFrames\Dataset4D_5LabTimes3Parts_146x176x44x25 
-k E:\4D_DeepLearning\DataV4_Invert_shifted_timens_5LabeledFrames\Dataset4D_5LabTimes_146x176x44x25ToSize73x88x22x25
-r 3

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
from UtilsDL import getLargestCC
import time
from UtilsDL import NumInString
from collections import OrderedDict
from UNetNetworks import UNet3DAntoine2


parser = argparse.ArgumentParser(description='Function to format the 4D image size to the original one, apply appening over the 25 frames')
parser.add_argument('-i' ,'--input1', help='Path to Folder with the results of the experiment from 4D deep learning')
parser.add_argument('-j', '--input2',help='Path to the 4D dataset with original size to reformat predictions')
parser.add_argument('-k', '--input3',help='Path to the 4D dataset to predict 25 frames')
parser.add_argument('-r','--radius', help='radius of the ball element for the opening operation', required=True)
parser.add_argument('-o','--output', help='Path to save folder with results', required=True)
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')

 
args = parser.parse_args()

path_results = args.input1
path_OrgSize = args.input2
path_Train = args.input3
radius = args.radius
out = args.output

   
OutTest = os.path.join(out, 'PostprocessData_Test_'+os.path.basename(path_results)+'_R'+radius)
    
if not os.path.isdir(OutTest):
        os.makedirs(OutTest)

###### folder to save temporal results

tmp = os.path.join(out,'tmp'+radius)
if not os.path.isdir(tmp):
        os.makedirs(tmp)
        

for ind,fold in enumerate(natsorted(listdir(path_results))): # fold or patient
    if not os.path.isdir(fold):
        print('file')
        continue
       
    path_pred = os.path.join(path_results,fold,'Best_Model_Pred')
    prediction =  natsorted(glob.glob(path_pred+os.path.sep+'PredTest*.nii.gz'))[0]    
    ##### Patient name 
    name = os.path.basename(prediction)
    name = name.split('_')[-1]
    name,_ = NumInString(name)
    print('Patient ', name)
    timePatient = 0
    
    tmp = os.path.join(out,'tmp'+radius+'P'+name)
    if not os.path.isdir(tmp):
            os.makedirs(tmp)
    
    ################### Predict 25 frames
    ###### split 4D to feed the model with the each frame
    clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmp,'ImageFrame3D'),'-d', '3', '--nii',
                 '-i',os.path.join(path_Train,'Images','Patient'+name+'.nii')]
    subprocess.run(clitkSplit)
    
    imagesFeed = natsorted(glob.glob(tmp+os.path.sep+'ImageFrame3D*.nii')) 
    
    ####### load the respective model 
    StateDic = os.path.join(path_results,fold,'BestcheckPoint.pth.tar')
    state_dict = torch.load(StateDic, map_location=torch.device('cpu'))["state_dict"]
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            nameL = k[7:]  # remove 'module.'
            new_state_dict[nameL] = v
        else:
            nameL = k
            new_state_dict[nameL] = v  
  
    model = UNet3DAntoine2(in_channel=1, out_channel=2) ### aorta and background
       
    model.load_state_dict(new_state_dict)
    model.eval()
    
    ##### Patient output folder
    OutPatient = os.path.join(OutTest,'Patient'+name)
    if not os.path.isdir(OutPatient):
        os.makedirs(OutPatient)
    
    
    ########### predict and post process each frame
    tomerge4D = []
    for ind,im in enumerate(imagesFeed):
        img=nib.load(im)
        volume_3D=img.get_fdata()
        shape_Orig_img = volume_3D.shape
        
        np_img = volume_3D.reshape(1, 1, shape_Orig_img[0], shape_Orig_img[1], shape_Orig_img[2])
        torch_image = torch.from_numpy(np_img).float()
        torch_image = torch_image.permute(0, 1, 4, 3, 2)

        start = time.time()
        pred = model(torch_image)
        end = time.time()
        #print('Pred time : ',end - start)
        timePatient += (end - start)
        current_pred = torch.argmax(pred[0, :, :, :, :], dim=0).float()
        current_pred = current_pred.permute(2, 1, 0)
        current_pred = torch.squeeze(current_pred)
        current_pred = current_pred.cpu().detach().numpy()
        current_pred = nib.Nifti1Image(current_pred, affine=img.affine)
        pathFrame = os.path.join(OutPatient,'Patient'+name+'_T'+str(ind)+'.nii.gz')
        nib.save(current_pred,pathFrame)
        tomerge4D.append(pathFrame)    
    
    #print('Total Time ', timePatient)
    ########## Merge 3D predictions into 4D
    out4D = os.path.join(OutPatient,'Patient'+name+'_4Dsegmentation.nii')
    clitk_list = ['clitkMergeSequence.exe']+ tomerge4D+['-o',out4D]
    subprocess.run(clitk_list)    
    
    ######## Resize 4D image 
    Orig_like = os.path.join(path_OrgSize,'Masks','Patient'+name+'.nii.gz')    
    
    #### Resize segmentation from best model for patient X and save it in tmp folder
    clitk = ['clitkAffineTransform.exe','-i', out4D, '-l', os.path.normpath(Orig_like), '-o', 
             out4D, '--interp', '0']
    subprocess.run(clitk)     
    
    ###### Main object on each 3D flrame 
    Image = nib.load(out4D).get_fdata()        
    MT = nib.load(out4D).affine
    
    ######## take only main object in each frame
    
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

    
    ###### Merge to generate 4D segentation
    out4D = os.path.join(OutPatient,'Patient'+name+'_4Dsegmentation.nii')
    clitk_list = ['clitkMergeSequence.exe']+ framesToMerge+['-o',out4D]
    subprocess.run(clitk_list)
    sh.rmtree(tmp, ignore_errors=True)
    

    
    