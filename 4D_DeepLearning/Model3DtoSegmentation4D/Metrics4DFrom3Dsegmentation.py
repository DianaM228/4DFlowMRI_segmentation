# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:39:07 2022
-i Users\Diana_MARIN\Documents\Diana\Deep_Learning\4D_Deep_Learning\OutTest4D 
-j 4D_DeepLearning\DataV4_Invert_shifted_timens_5LabeledFrames\Dataset4D_5LabTimes3Parts_146x176x44x25 
-k E:\4D_DeepLearning\DataV4_Invert_shifted_timens_5LabeledFrames\Dataset4D_5LabTimes_146x176x44x25
-f 36 
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
from UtilsDL import NumInString


parser = argparse.ArgumentParser(description='Function to format the 4D image size to the original one, apply appening and compute segmentation metrics')
parser.add_argument('-i' ,'--input1', help='Path to Folder with the results of the experiment from 4D deep learning')
parser.add_argument('-j', '--input2',help='Path to the dataset with original size 3 labels (AAo TDAo ABAo)')
parser.add_argument('-k', '--input3',help='Path to the dataset with original size 1 label (full aorta)')
parser.add_argument('-f','--folds_number', help='number of folds ', required=True, type=int)
parser.add_argument('-r','--radius', help='radius of the ball element for the opening operation', required=True)
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-t', '--threshold', help='set a different threshold than 0.5 for the probability map',default=None)

 
args = parser.parse_args()

path_results = args.input1
path_original = args.input2
path_originalFull = args.input3
labels = 3
radius = args.radius

print('Using Multi-labeled Masks')
LisNames=[]
Hau=[]
Dic=[]
for li in range(labels): # Create lists to save Measures and lists for aorta parts
    n=li+1        
                    
    LisNames.append('H{}'.format(n))
    LisNames.append('D{}'.format(n))
    globals()['H{}'.format(n)] = []
    globals()['D{}'.format(n)] = []
    

    ColNam=['subject']+LisNames+['Global H','Global D'] 
    #Create Dataframe
    indx=list(range(int(args.folds_number)))
    measuresBest = pd.DataFrame(columns=ColNam,index=indx)     
    measuresBestAvg = pd.DataFrame(columns=ColNam,index=indx)
          
GlobalHD=[]
GlobalDSC=[]  
counter=0 ## counter to manage the positions of the patient in the dataframe
for ind,fold in enumerate(natsorted(listdir(path_results))): # fold or patient    
    p_best = os.path.join(path_results,fold,'Best_Model_Pred')
    
    if fold.endswith('.xlsx'): ## To discard other files generated with results
        continue
    else:
        print(fold)                
        predictions =  natsorted(glob.glob(p_best+os.path.sep+"PredTest_*.nii.gz")) 
        targets =  natsorted(glob.glob(p_best+os.path.sep+"TargetTest_*.nii.gz"))      
        
     #### Output path        
    OutBest = os.path.join(p_best,'Post-Processing_MainObject_R'+args.radius)        
    if not os.path.isdir(OutBest):
        os.makedirs(OutBest)
        
    ####### Get labeled time point info
    time_frame = []
    for tname in predictions:
        ### get patient number or key name 
        basname = (os.path.basename(tname)).split('_')[-1]     
        ########## get index for labeled frames of each patient
        base = basname.split('T')[-1]
        base = base.split('.')[0]
        time_frame.append(int(base))
    ########   Get reference to resize the 3D images
    keyname,namestr = NumInString(basname)
        
            
    ####### Convert 3D to 4D 
    out4dPred=os.path.join(OutBest,'Pred4D_'+str(len(time_frame))+'labPatient'+keyname+'.nii.gz')
    clitk_list = ['clitkMergeSequence.exe']+ predictions +['-o',out4dPred]
    subprocess.run(clitk_list)
    
    ########  New 4D images for target     
    origin_1lab = os.path.join(path_originalFull,'Masks','Patient'+keyname+'.nii.gz')
    Orig_like = os.path.join(path_original,'Masks','Patient'+keyname+'.nii.gz')
    target3 = nib.load(Orig_like)
    target1 = nib.load(origin_1lab)   
    tar3 = torch.tensor(target3.get_data())
    tar1 = torch.tensor(target1.get_data())
    MT = nib.load(Orig_like).affine
    
    #### save New 4D 
    x,y,d,t = target3.shape        
    LabeledTimes4DTensor_Targ3 = torch.empty(x,y,d,len(time_frame))
    LabeledTimes4DTensor_Targ1 = torch.empty(x,y,d,len(time_frame))
    
    for phase,f in enumerate(time_frame):                   
        LabeledTimes4DTensor_Targ3[:,:,:,phase] = tar3[:,:,:,f]
        LabeledTimes4DTensor_Targ1[:,:,:,phase] = tar1[:,:,:,f]
    
    Nii_New4DTarg3 = nib.Nifti1Image(LabeledTimes4DTensor_Targ3.numpy(),MT)
    Nii_New4DTarg1 = nib.Nifti1Image(LabeledTimes4DTensor_Targ1.numpy(),MT)
    
    #### output new 4D prediction and target 1 and 3 labels     
    outnew4dTarg3B = os.path.join(OutBest,'New4D_Targ_3l_Patient'+str(keyname)+'_'+str(len(time_frame))+'Frames.nii.gz')
    outnew4dTarg1B = os.path.join(OutBest,'New4D_Targ_1l_Patient'+str(keyname)+'_'+str(len(time_frame))+'Frames.nii.gz')
    nib.save(Nii_New4DTarg3,outnew4dTarg3B)
    nib.save(Nii_New4DTarg1,outnew4dTarg1B)
      
    
    ####### Resize 4D prediction    
    outb = os.path.join(OutBest,'Resized_'+os.path.basename(out4dPred))
    clitk = ['clitkAffineTransform.exe','-i',out4dPred , '-l', os.path.normpath(outnew4dTarg1B), '-o', os.path.normpath(outb), '--interp', '0']
    subprocess.run(clitk)
    
    #### folder to save temporal results
    tmpBest = os.path.join(OutBest,'tmp' )                          
    if not os.path.isdir(tmpBest):
        os.makedirs(tmpBest)
        
    ImT_Best = nib.load(outb).get_fdata()     ### load new 4D prediction
    MT = nib.load(outb).affine
    
    ######## take only main object in each frame
     
    toMergeBest = []    
    for frame in range(np.size(ImT_Best,3)): 
        #### prediction >> new 4D resized
        result_Best= getLargestCC(ImT_Best[:,:,:,frame])        
        ### save 3D 
        out3dOpenBest = os.path.join(tmpBest,'OpenFrame_'+str(frame)+'.nii')
        Nii_Main3DPredB = nib.Nifti1Image(result_Best,MT) 
        nib.save(Nii_Main3DPredB,out3dOpenBest)
        toMergeBest.append(os.path.normpath(out3dOpenBest))                    
        #### opening 3D 
        op = ['clitkMorphoMath.exe','-i',out3dOpenBest,'-o',out3dOpenBest,'-t','3','-r',radius]
        subprocess.run(op)
           
            
    ##### merge the open 3D frames
    op_best = os.path.normpath(os.path.join(OutBest,'M_open_New4D_Pred_Patient'+str(keyname)+'_'+str(len(time_frame))+'_Frames.nii.gz'))
    
    clitk_list = ['clitkMergeSequence.exe']+ toMergeBest+['-o',op_best]
    subprocess.run(clitk_list)   
   
    ##propagate boundaries from the manual segmentation to automatic one
    
    ### Binarize Multi-labeled Manual segmentation                        
    ManualParts=[]
    SmapTarget=[]
    for ao in range(labels): # aorta parts or labels            
        OutBinTar=os.path.join(tmpBest,'Target_Ao_Part'+str(ao+1)+'.nii.gz')
        binOmaskT = ['clitkBinarizeImage.exe','-i',outnew4dTarg3B,'-l',str(ao+1),'-u',str(ao+1),'-o',OutBinTar]
        subprocess.run(binOmaskT)
        ManualParts.append(os.path.join(tmpBest,'Target_Ao_Part'+str(ao+1)+'.nii.gz'))
        
        # Signed Maurer distance Map for all aorta parts
        smMap = ['clitkSignedMaurerDistanceMap.exe','-i',OutBinTar,'-o',os.path.join(tmpBest,'TargMap'+str(ao+1)+'.nii')]
        subprocess.run(smMap)
        SmapTarget.append(os.path.join(tmpBest,'TargMap'+str(ao+1)+'.nii'))

    ### Argmin with SmapTarget
    SmapTarget = ','.join(SmapTarget)
    LArgmin = ['clitkArgminImage.exe','-i',SmapTarget,'-o',os.path.join(tmpBest,'ArgminImage.nii'),'-s']
    subprocess.run(LArgmin)

    ## Convert automatic binary mask (new 4D after resize and open) to mask with N labels keeping the same boundaries as manual segmentation
    AutMaskNlabBest= os.path.join(OutBest,'New4DPred_Open_3Lab_'+'Patient'+keyname+'.nii.gz')    
    LMulArgmin = ['clitkImageArithm.exe','-i',os.path.join(tmpBest,'ArgminImage.nii'),'-j',op_best,'-o',AutMaskNlabBest,'-t','1']
    subprocess.run(LMulArgmin)
    
    ###### split target and pred 1 label  for global HD  (after ope and main object) op_best
    clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmpBest,'Target1lab'),'-d', '3', '--nii','-i',outnew4dTarg1B]
    subprocess.run(clitkSplit)
    
    clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmpBest,'Pred1lab'),'-d', '3', '--nii','-i',op_best]
    subprocess.run(clitkSplit)
    
    ### Binarize Atomatic multi-labeled mask
    AutomaticPartsBest =[]    
    indpHD_bestAll = []
    indpDSC_bestAll = []
    b=1 ### to manage the position of the metric in the dataframe
    for ao in range(labels): # aorta parts or labels            
        OutBinAutBest=os.path.join(tmpBest,'Aut_Ao_Part'+str(ao+1)+'.nii.gz')        
        binOmaskA = ['clitkBinarizeImage.exe','-i',AutMaskNlabBest,'-l',str(ao+1),'-u',str(ao+1),'-o',OutBinAutBest]
        subprocess.run(binOmaskA)
        AutomaticPartsBest.append(OutBinAutBest)
                
        ### Metrics each aorta part
        convManual = ['clitkImageConvert.exe', '-i',ManualParts[ao], '-o', ManualParts[ao], '-t', 'uchar']
        subprocess.run(convManual)
        ###### Dice 4D
        dice = ['clitkDice.exe','-i',ManualParts[ao], '-j', OutBinAutBest] 
        Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
        Dice_best_local = Dice.stdout.read()
                
        #### separate 4D to compute HD 
        clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmpBest,'Pred'+str(ao+1)),'-d', '3', '--nii','-i',OutBinAutBest]
        subprocess.run(clitkSplit)
                 
        ##### split target local 
        clitkSplit =['clitkSplitImage.exe', '-o',os.path.join(tmpBest,'Target'+str(ao+1)),'-d', '3', '--nii','-i',ManualParts[ao]]
        subprocess.run(clitkSplit)
        
        
        ### Compute hausdorff independently per frame for the ao part
        indpHD_best = []
        indpDSC_best_avg = []                        
        for time in range(len(time_frame)):      ##loop over frames to compute HD independently 
        
            ###### do main object to avoid problems with disconected point in the manual segmentation 
          
            targetPart = nib.load(os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii'))).get_fdata()
            targetPartMain = getLargestCC(targetPart)
            Nii_targetPart = nib.Nifti1Image(targetPartMain,MT)
            nib.save(Nii_targetPart,os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii')))
            
             ###### if there are disconections after open                  
            targetPart = nib.load(os.path.normpath(os.path.join(tmpBest,'Pred'+str(ao+1)+'_'+str(time)+'.nii'))).get_fdata()
            targetPartMain = getLargestCC(targetPart)
            Nii_targetPart = nib.Nifti1Image(targetPartMain,MT)
            nib.save(Nii_targetPart,os.path.normpath(os.path.join(tmpBest,'Pred'+str(ao+1)+'_'+str(time)+'.nii')))
            
            ############
            #### compute HD for aorta region in the specific time frame                  
            haus = ['clitkHausdorffDistance.exe', '-i',os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii')) ,'-j',os.path.normpath(os.path.join(tmpBest,'Pred'+str(ao+1)+'_'+str(time)+'.nii'))]
            Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
            Haus_Best = Haus.stdout.read()
            
            convManual = ['clitkImageConvert.exe','-i', os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii')), '-o', os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii')), '-t', 'uchar']
            subprocess.run(convManual)
            #### compute Dice for aorta region in the specific time frame
            dice = ['clitkDice.exe', '-i',os.path.normpath(os.path.join(tmpBest,'Target'+str(ao+1)+'_'+str(time)+'.nii')) ,'-j',os.path.normpath(os.path.join(tmpBest,'Pred'+str(ao+1)+'_'+str(time)+'.nii'))]
            Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
            Dice_best_local_avg = Dice.stdout.read()
            try:
                Haus_BestDec = Haus_Best.decode()
                globals()['H{}'.format(ao+1)].append(float(Haus_BestDec))
                indpHD_best.append(float(Haus_BestDec))
                indpDSC_best_avg.append(float(Dice_best_local_avg.decode()))
            except:
                print('Error Best Model >> Patient '+str(keyname)+'  Frame '+str(time)+'  Aorta part ',str(ao+1))
            
                        
            if ao==0:  ### global HD and DCS computed only once (the first time or when it's computed the metric for label 1)
                ### Main object for prediction full one label 
                # targetPart = nib.load(os.path.join(tmpBest,'Pred1lab_'+str(time)+'.nii')).get_fdata()
                # targetPartMain = getLargestCC(targetPart)
                # Nii_targetPart = nib.Nifti1Image(targetPartMain,MT)
                # nib.save(Nii_targetPart,os.path.normpath(os.path.join(tmpBest,'Pred1lab_'+str(time)+'.nii')))
            
                haus = ['clitkHausdorffDistance.exe', '-i',os.path.join(tmpBest,'Target1lab_'+str(time)+'.nii') ,'-j',os.path.join(tmpBest,'Pred1lab_'+str(time)+'.nii')]
                Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
                Haus_bestAll = Haus.stdout.read()
                #indpHD_bestAll.append(float(Haus_bestAll.decode()))                
                Haus_bestAllDec = Haus_bestAll.decode()
                indpHD_bestAll.append(float(Haus_bestAllDec))
                GlobalHD.append(float(Haus_bestAllDec))
                
                #### AVG dice
                convManual = ['clitkImageConvert.exe','-i',os.path.join(tmpBest,'Target1lab_'+str(time)+'.nii') , '-o',os.path.join(tmpBest,'Target1lab_'+str(time)+'.nii') , '-t', 'uchar']
                subprocess.run(convManual)
                
                dice = ['clitkDice.exe', '-i',os.path.join(tmpBest,'Target1lab_'+str(time)+'.nii') ,'-j',os.path.join(tmpBest,'Pred1lab_'+str(time)+'.nii')]
                Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
                Dice_best_all_avg = Dice.stdout.read()
                indpDSC_bestAll.append(float(Dice_best_all_avg.decode()))
                
           
            
        avgHDBestlocal = round(np.mean(np.array(indpHD_best)),2)
        stdHDBestlocal = round(np.std(np.array(indpHD_best)),2)
        
        avgDSCBestlocal = round(np.mean(np.array(indpDSC_best_avg)),2)
        stdDSCBestlocal = round(np.std(np.array(indpDSC_best_avg)),2)
        
                       
        try:
                                    
            measuresBest.iat[counter,b] = str(avgHDBestlocal)+u"\u00B1"+str(stdHDBestlocal)
            measuresBestAvg.iat[counter,b] = str(avgHDBestlocal)+u"\u00B1"+str(stdHDBestlocal)            
            b=b+1
            Dice_best_localDec =Dice_best_local.decode()
            globals()['D{}'.format(ao+1)].append(float(Dice_best_localDec))
            measuresBest.iat[counter,b] = float(Dice_best_localDec)
            measuresBestAvg.iat[counter,b] = str(avgDSCBestlocal)+u"\u00B1"+str(stdDSCBestlocal)            
            b=b+1
        except:                    
            print('Error computing measures in Patient ',keyname,' label  ',str(ao+1))
                    
            measuresBest.iat[counter,b] = 'error'
            measuresBestAvg.iat[counter,b] = 'error'            
            b=b+1
            measuresBest.iat[counter,b] = 'error'
            measuresBestAvg.iat[counter,b] = 'error'            
            b=b+1

    ### Global metrics 
    avgHDBestAll = round(np.mean(np.array(indpHD_bestAll)),2)
    stdHDBestAll = round(np.std(np.array(indpHD_bestAll)),2)
    
    ## AVG Dice
    avgDSCBestAll = round(np.mean(np.array(indpDSC_bestAll)),2)
    stdDSCBestAll = round(np.std(np.array(indpDSC_bestAll)),2)
    
    convManual = ['clitkImageConvert.exe','-i', op_best, '-o', op_best, '-t', 'uchar']
    subprocess.run(convManual)

   
    gdice = ['clitkDice.exe','-i', op_best, '-j', outnew4dTarg1B]
    GDice = subprocess.Popen(gdice, stdout=subprocess.PIPE)
    GDice_M_best = GDice.stdout.read()
    
    try:
        
        measuresBest.iat[counter,b] = str(avgHDBestAll)+u"\u00B1"+str(stdHDBestAll)
        measuresBestAvg.iat[counter,b] = str(avgHDBestAll)+u"\u00B1"+str(stdHDBestAll)
        GDice_M_bestDec = GDice_M_best.decode()
        GlobalDSC.append(float(GDice_M_bestDec))
        measuresBest.iat[counter,b+1] = float(GDice_M_bestDec)
        #measuresBest.iat[counter,b+1] = float(GDice_M_best.decode())
        measuresBestAvg.iat[counter,b+1] = str(avgDSCBestAll)+u"\u00B1"+str(stdDSCBestAll)
       
    except:
        print('Error computing global measures in Patient ',keyname)
        measuresBest.iat[counter,b] = 'error'    
        measuresBestAvg.iat[counter,b] = 'error' 
        measuresBest.iat[counter,b+1] = 'error'
        measuresBestAvg.iat[counter,b+1] = 'error'
      
    
    sh.rmtree(tmpBest, ignore_errors=True) # Delet temporal results      
    measuresBest.iat[counter,0] = 'Patient'+keyname
    measuresBestAvg.iat[counter,0] = 'Patient'+keyname
    counter+=1
    
    
lismeans=['Mean']
for li in range(labels):
    n=li+1      
    meanH=np.mean(np.array(globals()['H{}'.format(n)]))
    stdH=np.std(np.array(globals()['H{}'.format(n)]))
    

    meanD=np.mean(np.array(globals()['D{}'.format(n)]))
    stdD=np.std(np.array(globals()['D{}'.format(n)]))  
    
    lismeans.append(str(round(meanH,2))+u"\u00B1"+str(round(stdH,2)))
    lismeans.append(str(round(meanD*100,2))+u"\u00B1"+str(round(stdD*100,2)))

meanGlobalAllP = np.mean(GlobalHD)
stdGlobalAllP = np.std(GlobalHD)
lismeans.append(str(round(meanGlobalAllP,2))+u"\u00B1"+str(round(stdGlobalAllP,2))) 

meanGlobalAllP = np.mean(GlobalDSC)
stdGlobalAllP = np.std(GlobalDSC)   
lismeans.append(str(round(meanGlobalAllP,2))+u"\u00B1"+str(round(stdGlobalAllP,2))) 

add_row = pd.Series(lismeans,index=ColNam)
measuresBest = measuresBest.append(add_row, ignore_index=True)
   
    
if args.threshold:
    measuresBest.to_excel(os.path.join(path_results,'Threshold_'+args.threshold+'_R'+args.radius+'_MainObject_BestModel_PerformanceAllFolds.xlsx'))
    measuresBestAvg.to_excel(os.path.join(path_results,'Threshold_'+args.threshold+'_R'+args.radius+'_MainObject_BestModel_PerformanceAllFolds_Avg_HD_DSC.xlsx'))
else:
    measuresBest.to_excel(os.path.join(path_results,'R'+args.radius+'MainObject_BestModel_PerformanceAllFolds.xlsx'))
    measuresBestAvg.to_excel(os.path.join(path_results,'R'+args.radius+'MainObject_BestModel_PerformanceAllFolds_Avg_HD_DSC.xlsx'))



