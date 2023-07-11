#This code normalizes 3D nifti volumens and convert them to 4D and resize them to be used in 4D deep learning 
#e.g.
#-i \Users\Diana_MARIN\Documents\Diana\Deep_Learning\4D_Deep_Learning\AllFramesNii_36P 
#-j \Manual_Segmentations_Arnaud_Diana\MergedGTManualSegm_4D_1Lab 
#-l \Users\Diana_MARIN\Documents\Diana\Deep_Learning\4D_Deep_Learning\SystoleDiastoleFrames36.csv 
#-s 146 176 44

import argparse
import os
from os import listdir
import nibabel as nib
import shutil as sh
import glob
from natsort import natsorted
import subprocess
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Function to convert nifti volumes to 4D, resize the images and generate the 4D mask ')
parser.add_argument('-i','--Path_Nifti_Vol', help='Path to the Folder with subforlder for every patient, which has all frames in Nifti format')
parser.add_argument('-j','--Path_Segmentations', help='Path to the folder with subforlder for every patient, which has systole and diastole segmentation')
parser.add_argument('-l','--Patient_Frames',  help='Path to file with information about time in systole and diastole for each patient', required=True)
parser.add_argument('-s','--Size', nargs='+', help='List of size to be cur in x,y,z ', required=True)


args = parser.parse_args()

P_nii = args.Path_Nifti_Vol
P_seg = args.Path_Segmentations
Frames = args.Patient_Frames
size = args.Size

df = pd.read_csv(Frames)
for i in natsorted(listdir(P_nii)):
    Patient = os.path.join(P_nii,i)    
    ### Convert to 4D all the 3D frames from patient i 
    volumes = natsorted(glob.glob(Patient+'/*.nii'))
        
    out4DI = os.path.join(os.path.dirname(P_nii),'Dataset4D_crop_'+size[0]+'x'+size[1]+'x'+size[2]+'x25','Images')
    if not os.path.isdir(out4DI):
        os.makedirs(out4DI)
        
    ### Normalize each volume
    
    for j in natsorted(volumes):
        Norm = ['clitkNormalizeImageFilter.exe','-i',j,'-o',j]
        subprocess.run(Norm)
    
    
    ### Merge 3D 
    clitk_list = ['clitkMergeSequence.exe']
    o = ['-o', os.path.join(out4DI,i+'.nii')]
    args_list4D = clitk_list + volumes + o
    subprocess.run(args_list4D)
    
    
    ## Crop 4D images
    resampling = ['clitkExtractPatch.exe', '-i', o[-1],'-o',o[-1],'-s ' + size[0] + ',' + size[1]+ ',' + size[2]+ ',' +'25']            
    subprocess.run(resampling)
    

    #### Build 4D mask 
    
    
    p_seg = os.path.join(P_seg,i)
    Patient_Number = i.split('Patient')[-1]
    
    ######## Resize Systole and Diastole as gray intensity images
    
    OrigDiastole = os.path.join(p_seg,'D'+Patient_Number+'MASK.nii.gz')
    OrigSystole = os.path.join(p_seg,'S'+Patient_Number+'MASK.nii.gz')
    
    CropDiastole = os.path.join(p_seg,'Crop_D'+Patient_Number+'MASK.nii.gz')
    CropSystole = os.path.join(p_seg,'Crop_S'+Patient_Number+'MASK.nii.gz')
    
    resampling = ['clitkExtractPatch.exe', '-i', OrigDiastole,'-o',CropDiastole,'-s ' + size[0] + ',' + size[1]+ ',' + size[2]]            
    subprocess.run(resampling)
    
    resampling = ['clitkExtractPatch.exe', '-i', OrigSystole,'-o',CropSystole,'-s ' + size[0] + ',' + size[1]+ ',' + size[2]]            
    subprocess.run(resampling)
    
    
    Frame_Diastole = nib.load(CropDiastole).get_fdata()
    Frame_systole = nib.load(CropSystole).get_fdata()
    
    mask4D = np.zeros((int(size[0]),int(size[1]),int(size[2]),25))
    
    ss=df.loc[df['Patient']==int(Patient_Number)]
    
    mask4D[:,:,:,int(ss['Diastole'].values)] = Frame_Diastole
    mask4D[:,:,:,int(ss['Systole'].values)] = Frame_systole
    
    MT = nib.load(CropSystole).affine
    Nii_4DMask = nib.Nifti1Image(mask4D,MT)
    
    out4DM = os.path.join(os.path.dirname(P_nii),'Dataset4D_crop_'+size[0]+'x'+size[1]+'x'+size[2]+'x25','Masks')
    if not os.path.isdir(out4DM):
        os.makedirs(out4DM)
    
    
    nib.save(Nii_4DMask,os.path.join(out4DM,i+'.nii.gz'))

    
  