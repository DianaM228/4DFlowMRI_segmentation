# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:23:03 2022

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

parser = argparse.ArgumentParser(description='Function to format the 4D image size to the original one, apply appening and compute segmentation metrics')
parser.add_argument('-i' ,'--input1', help='Path to Folder with the results of the experiment from 4D deep learning')
parser.add_argument('-f','--folds_number', help='number of folds ', required=True, type=int)
parser.add_argument('-r','--radius', help='radius of the ball element for the opening operation', required=True)
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-t', '--threshold', help='set a different threshold than 0.5 for the probability map',default=None)



## New4DPred_Open_3Lab_PredTest_0_Patient12.nii
## New4D_Targ_3l_Patient12_5Frames.nii

args = parser.parse_args()

path_results = args.input1


# if labels>1: ## Multi-labeled masks
print('Using Multi-labeled Masks')
LisNames=[]
Hau=[]
Dic=[]
  

ColNam=['subject','Global H','Global D'] 
#Create Dataframe
indx=list(range(int(args.folds_number)))
measuresBest = pd.DataFrame(columns=ColNam,index=indx) 


##############
def ReduceLabelNumber(mask,lab_to_be_rep,new_lab,outpath):
    M= nib.load(mask).get_fdata()
    head=nib.load(mask).affine
    for i in lab_to_be_rep:  
        ind = np.where(M == int(i))
        M[ind]=new_lab
  
    New_Mask = nib.Nifti1Image(M,head)
    nib.save(New_Mask,outpath)
    
    l = ['clitkImageConvert.exe','-i',outpath,'-o',outpath,'-t','uchar']
    subprocess.run(l)
    
#############
          
GlobalHD=[]
GlobalDSC=[]  
counter=0 ## counter to manage the positions of the patient in the dataframe
for ind,fold in enumerate(natsorted(listdir(path_results))): # fold or patient
     
     if args.threshold:
         p_best = os.path.join(path_results,fold,'Threshold_'+args.threshold+'_R'+args.radius+'_Post-Process_MainO')
     else:         
         p_best = os.path.join(path_results,fold,'Best_Model_Pred','Post-Processing_MainObject_R'+args.radius)
     
     if fold.endswith('.xlsx'): ## To discard other files generated with results
         continue
     else:
        print(fold)
        imagesBest = natsorted(glob.glob(p_best+'/*.nii.gz'))
        
     OutBest = os.path.join(p_best,'Post-Processing_TwoLabels')
     if not os.path.isdir(OutBest):
        os.makedirs(OutBest)  
        
     for imB in imagesBest: 
         if 'New4DPred_Open_3Lab' in imB:    
             pred = imB
         elif 'New4D_Targ_3l' in imB:
             targ = imB
     keyname = os.path.basename(pred).split('Patient')[-1]
     keyname = keyname.split('.')[0]
     #### reduce labels to keep only thoracic aorta 
     ReduceLabelNumber(pred,[3],0,os.path.join(OutBest,'Pred_2labels.nii'))
     ReduceLabelNumber(targ,[3],0,os.path.join(OutBest,'Targ_2labels.nii'))
     
     #### take images with only thoracic aorta and convert them to 1 label 
     ReduceLabelNumber(os.path.join(OutBest,'Pred_2labels.nii'),[2],1,
                       os.path.join(OutBest,'Pred_ThoracicFull.nii'))
     
     ReduceLabelNumber(os.path.join(OutBest,'Targ_2labels.nii'),[2],1,
                       os.path.join(OutBest,'Targ_ThoracicFull.nii'))
     
     # ######## Compute Global Dice Metric 
     # dice = ['clitkDice.exe','-i',os.path.join(OutBest,'Pred_ThoracicFull.nii'), '-j', os.path.join(OutBest,'Targ_ThoracicFull.nii')]
     # Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
     # Dice_best_local = Dice.stdout.read()  
     # ###
     
         
     ##### Split frames 
     tmp = os.path.join(OutBest,'tmp')
     if not os.path.isdir(tmp):
        os.makedirs(tmp)
     
     clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmp,'Target'),'-d', '3', '--nii','-i',os.path.join(OutBest,'Targ_ThoracicFull.nii')]
     subprocess.run(clitkSplit)
     
     clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmp,'Pred'),'-d', '3', '--nii','-i',os.path.join(OutBest,'Pred_ThoracicFull.nii')]
     subprocess.run(clitkSplit)
    
     indpHD_best = []
     indpDSC_best = []
     for time in range(5):
         ###### do main object to avoid problems with disconected point in the manual segmentation       
         targetPart = nib.load(os.path.join(tmp,'Target_'+str(time)+'.nii')).get_fdata()
         targetPartMain = getLargestCC(targetPart)
         MT = nib.load(os.path.join(tmp,'Target_'+str(time)+'.nii')).affine
         Nii_targetPart = nib.Nifti1Image(targetPartMain,MT)
         nib.save(Nii_targetPart,os.path.join(tmp,'Target_'+str(time)+'.nii'))
         
         ###### do main object to avoid problems with disconected point in the automatic segmentation after opening  
         targetPart = nib.load(os.path.join(tmp,'Pred_'+str(time)+'.nii')).get_fdata()
         targetPartMain = getLargestCC(targetPart)
         MT = nib.load(os.path.join(tmp,'Pred_'+str(time)+'.nii')).affine
         Nii_targetPart = nib.Nifti1Image(targetPartMain,MT)
         nib.save(Nii_targetPart,os.path.join(tmp,'Pred_'+str(time)+'.nii'))
         
         haus = ['clitkHausdorffDistance.exe', '-i',os.path.join(tmp,'Pred_'+str(time)+'.nii'),'-j',os.path.join(tmp,'Target_'+str(time)+'.nii')]
         Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
         Haus_Best = Haus.stdout.read() 
         
         ######## Compute Global Dice Metric 
         l = ['clitkImageConvert.exe','-i',os.path.join(tmp,'Pred_'+str(time)+'.nii'),'-o',os.path.join(tmp,'Pred_'+str(time)+'.nii'),'-t','uchar']
         subprocess.run(l)
         
         dice = ['clitkDice.exe','-i',os.path.join(tmp,'Pred_'+str(time)+'.nii'),'-j',os.path.join(tmp,'Target_'+str(time)+'.nii')]
         Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
         Dice_best_local = Dice.stdout.read() 
         
         try:
            HausDec=Haus_Best.decode()
            indpHD_best.append(float(HausDec))
            GlobalHD.append(float(HausDec))  
            
            diceDec=Dice_best_local.decode()
            indpDSC_best.append(float(diceDec))
            GlobalDSC.append(float(diceDec))
            
         except:
            print('Error Best Model >> Patient '+str(keyname)+'  Frame '+str(time))
            
            
     avgHDBestlocal = round(np.mean(np.array(indpHD_best)),2)
     stdHDBestlocal = round(np.std(np.array(indpHD_best)),2)
     
     avgDSCBestlocal = round(np.mean(np.array(indpDSC_best)),2)
     stdDSCBestlocal = round(np.std(np.array(indpDSC_best)),2)
                    
     
     try:
         measuresBest.iat[counter,1] = str(avgHDBestlocal)+u"\u00B1"+str(stdHDBestlocal)
         #Dice_Dec = Dice_best_local.decode()
         measuresBest.iat[counter,2] = str(avgDSCBestlocal)+u"\u00B1"+str(stdDSCBestlocal)
         #GlobalDSC.append(float(Dice_Dec))
        
     except:                    
         print('Error computing measures in Patient ',keyname)
                
         measuresBest.iat[counter,1] = 'error'
         measuresBest.iat[counter,2] = 'error'
         
     
     measuresBest.iat[counter,0] = 'Patient'+keyname
     counter += 1
     
     sh.rmtree(tmp, ignore_errors=True) 

lismeans=['Mean']
meanGlobalAllP = np.mean(GlobalHD)
stdGlobalAllP = np.std(GlobalHD)
lismeans.append(str(round(meanGlobalAllP,2))+u"\u00B1"+str(round(stdGlobalAllP,2))) 

meanGlobalAllP = np.mean(GlobalDSC)
stdGlobalAllP = np.std(GlobalDSC)   
lismeans.append(str(round(meanGlobalAllP,2))+u"\u00B1"+str(round(stdGlobalAllP,2))) 

add_row = pd.Series(lismeans,index=ColNam)
measuresBest = measuresBest.append(add_row, ignore_index=True)

if args.threshold:  
    measuresBest.to_excel(os.path.join(path_results,'Threshold_'+args.threshold+'_R'+args.radius+'MainObject_BestModel_PerformanceAllFolds_TwoLabels.xlsx'))  

else:    
    measuresBest.to_excel(os.path.join(path_results,'R'+args.radius+'MainObject_BestModel_PerformanceAllFolds_TwoLabels.xlsx'))        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

