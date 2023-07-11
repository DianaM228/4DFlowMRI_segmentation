"""
This code does post processing for 3D UNet results (convert images to original size)
"""

import argparse
import os
from os import listdir
import glob
from natsort import natsorted
import subprocess
import shutil as sh
from importlib.machinery import SourceFileLoader
foo = SourceFileLoader("NumInString", r"C:\Users\Diana_MARIN\Documents\aortadiag4d\Utils.py").load_module()
import pandas as pd
import numpy as np
from skimage.measure import label
import nibabel as nib


parser = argparse.ArgumentParser(description='Function to propagate aorta boungaries')
parser.add_argument('-i','--Path_Results', help='Path to folder with results (one folder by fold or test)')
parser.add_argument('-j','--Path_Masks_labels', help='Path to folder with manual segmentations (multi-label)')
parser.add_argument('-l','--Labels', help='number of boundaries', required=True, type=int)
parser.add_argument('-t','--Number_test', help='number of test patients (if loo = folds)', required=True, type=int)
parser.add_argument('-r','--results', help='Compute average results between all folds (folders) (for leave-one-out) or per each fold (Influence of number data)', choices=['all','test'], required=True)
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-m', '--Main_object', help='activate if you want to keep just biggest object in segentation', action='store_true')


args = parser.parse_args()
results = args.Path_Results
Manual_maks = args.Path_Masks_labels
labels = args.Labels
Number_test =args.Number_test


if args.Main_object:
    def getLargestCC(segmentation):
        labels = label(segmentation)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max=(labels == largest).astype(int)
        return labels_max
    
    
## Create dataframe to save results in csv file 
if labels>1: ## Multi-labeled masks
        print('Using Multi-labeled Masks')
        LisNames=[]
        Hau=[]
        Dic=[]
        for li in range(labels): # Create lists to save Measures and lists for aorta parts
            n=li+1        
            globals()['H{}'.format(n)] = []
            globals()['D{}'.format(n)] = []
                        
            LisNames.append('H{}'.format(n))
            LisNames.append('D{}'.format(n))
            
        
            ColNam=['subject']+LisNames+['Global H','Global D'] 
            #Create Dataframe
            indx=list(range(Number_test))
            measures = pd.DataFrame(columns=ColNam,index=indx)         
          
else: ## binary masks
    Hau=[]
    Dic=[]
    indx=list(range(Number_test))
           
    ColNam=['subject','Hausdorff','Dice']
    measures = pd.DataFrame(columns=ColNam,index=indx)
    

###### Resize, open, label propagation and metrics by patient 
if args.results=='all':
    counter=0
for pat, i in enumerate(natsorted(listdir(results))): #folds
    fold=os.path.join(results,i)    
    if fold.endswith('.xlsx'):
        continue
    else:
        print('Test ',i)
        images = natsorted(glob.glob(fold+'/*.nii.gz'))
        
        ###### keep predictions
        if args.results=='test':
            counter = 0
        for p in images:
            if 'PredTest' in p: 
                
                ### Folder to save result of postprocessing
                if args.Main_object:
                    Out = os.path.join(fold,'Post-Processing_MainObject')
                else:
                    Out = os.path.join(fold,'Post-Processing')
                
                if not os.path.isdir(Out):
                    os.makedirs(Out)
        
                ### get patient number or key name 
                basname = (os.path.basename(p)).split('_')[-1]    
                keyname,namestr = foo.NumInString(basname)
                
                print('Patient ', keyname)   
                #### If Main_object is activated, keep only biggest object in prediction
                if args.Main_object:
                    ImT = nib.load(p).get_fdata()
                    MT = nib.load(p).affine
                    result= getLargestCC(ImT)
            
                    Nii_PSn = nib.Nifti1Image(result,MT)
                    
                    Result = os.path.join(Out,'M'+os.path.basename(p))
                    nib.save(Nii_PSn,Result)
                    p = Result
                    
                # Convert Mask (prediction) to original size
                Orig_like = os.path.join(Manual_maks,'Patient'+keyname,basname)
                
                print('resizing ', os.path.basename(p), ' like ', os.path.basename(Orig_like))
                
                out =  os.path.join(Out,'Resized_'+os.path.basename(p))   
                clitk = ['clitkAffineTransform.exe','-i', p, '-l', Orig_like, '-o', out, '--interp', '0']
                subprocess.run(clitk)
                
                ###  Apply opening to automatic mask (after sizing)
                Omaskp = os.path.join(Out,'Resized_Open_'+os.path.basename(p))
                op = ['clitkMorphoMath.exe','-i',out,'-o',Omaskp,'-t','3','-r','3']
                subprocess.run(op)
                
                measures.iat[counter,0] = 'Patient'+keyname
                
                if labels>1:
                    ##propagate boundaries from the manual segmentation to automatic one
                    ### Binarize Multi-labeled Manual segmentation        
                    #Path to save Temporal information       
                    outTemp = os.path.join(Out,'Temporal_Res_Patient'+keyname)               
                    if not os.path.isdir(outTemp):
                        os.makedirs(outTemp) 
                       
                    ManualParts=[]
                    SmapTarget=[]
                    for ao in range(labels): # aorta parts or labels            
                        OutBinTar=os.path.join(outTemp,'Target_Ao_Part'+str(ao+1)+'.nii.gz')
                        binOmaskT = ['clitkBinarizeImage.exe','-i',Orig_like,'-l',str(ao+1),'-u',str(ao+1),'-o',OutBinTar]
                        subprocess.run(binOmaskT)
                        ManualParts.append(os.path.join(outTemp,'Target_Ao_Part'+str(ao+1)+'.nii.gz'))
                        
                        # Signed Maurer distance Map for all aorta parts
                        smMap = ['clitkSignedMaurerDistanceMap.exe','-i',OutBinTar,'-o',os.path.join(outTemp,'TargMap'+str(ao+1)+'.nii')]
                        subprocess.run(smMap)
                        SmapTarget.append(os.path.join(outTemp,'TargMap'+str(ao+1)+'.nii'))
                
                    ### Argmin with SmapTarget
                    SmapTarget = ','.join(SmapTarget)
                    LArgmin = ['clitkArgMin.exe','-i',SmapTarget,'-o',os.path.join(outTemp,'ArgminImage.nii'),'-s']
                    subprocess.run(LArgmin)
            
                    ## Convert automatic binary mask (the resized one) to mask with N labels keeping the same boundaries as manual segmentation
                    AutMaskNlab= os.path.join(Out,'Resized_Open_3Lab_'+os.path.basename(p))
                    LMulArgmin = ['clitkImageArithm.exe','-i',os.path.join(outTemp,'ArgminImage.nii'),'-j',Omaskp,'-o',AutMaskNlab,'-t','1']
                    subprocess.run(LMulArgmin)
                    
                    ### 
                    ## Binarize Atomatic multi-labeled mask 
                    
                    AutomaticParts =[]
                    b=1
                    for ao in range(labels): # aorta parts or labels            
                        OutBinAut=os.path.join(outTemp,'Aut_Ao_Part'+str(ao+1)+'.nii.gz')
                        binOmaskA = ['clitkBinarizeImage.exe','-i',AutMaskNlab,'-l',str(ao+1),'-u',str(ao+1),'-o',OutBinAut]
                        subprocess.run(binOmaskA)
                        AutomaticParts.append(OutBinAut)
                        
                        ### Metrics each aorta part
                        convManual = ['clitkImageConvert.exe', '-i',ManualParts[ao], '-o', ManualParts[ao], '-t', 'uchar']
                        subprocess.run(convManual)
                        
                        haus = ['clitkHausdorffDistance.exe', '-i',ManualParts[ao] ,'-j',OutBinAut]
                        Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
                        Haus_M = Haus.stdout.read()
                        
                        dice = ['clitkDice.exe','-i',ManualParts[ao], '-j', OutBinAut]
                        Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
                        Dice_M = Dice.stdout.read()
                        
                        try:
                            globals()['H{}'.format(ao+1)].append(float(Haus_M.decode()))
                            globals()['D{}'.format(ao+1)].append(float(Dice_M.decode())) 
                                
                            measures.iat[counter,b] = float(Haus_M.decode())
                            b=b+1
                            measures.iat[counter,b] = float(Dice_M.decode())*100
                            b=b+1
                          
                                
                            print('Hausdorff Distance Label ',str(ao+1)+': ', Haus_M.decode(),'Dice Distance label ',str(ao+1)+': ',str(float(Dice_M.decode())*100),'\n')
                                
                                
                        except:
                            print('Error computing measures in Patient ',keyname,' label  ',str(ao+1))
                            
                            measures.iat[counter,b] = 'error'
                            b=b+1
                            measures.iat[counter,b] = 'error'
                            b=b+1
                    
                    
                    ## Global measures 
                    convManual = ['clitkImageConvert.exe','-i', Orig_like, '-o', Orig_like, '-t', 'uchar']
                    subprocess.run(convManual)
                    
                    HMap = os.path.join(Out,'HausMAP_Patient'+keyname+'.nii')
                    ghaus = ['clitkHausdorffDistance.exe', '-i', Orig_like, '-j', AutMaskNlab,'-o',HMap]
                    GHaus= subprocess.Popen(ghaus, stdout=subprocess.PIPE)
                    GHaus_M = GHaus.stdout.read()
                    
                            
                    
                    gdice = ['clitkDice.exe','-i', Orig_like, '-j', AutMaskNlab]
                    GDice = subprocess.Popen(gdice, stdout=subprocess.PIPE)
                    GDice_M = GDice.stdout.read()
                    
                    try:
                        Hau.append(float(GHaus_M.decode()))
                        Dic.append(float(GDice_M.decode())*100) 
                        
                        
                        measures.iat[counter,b] = float(GHaus_M.decode())        
                        measures.iat[counter,b+1] = float(GDice_M.decode())*100
                        
                        print('Global Hausdorff Distance: ',str(float(GHaus_M.decode())), 
                          'Global Dice Score: ',str(float(GDice_M.decode())*100))
                        
                    except:
                        
                        measures.iat[counter,b] = 'error'        
                        measures.iat[counter,b+1] ='error'
                         
                        print('Segmentation error')
                    
                    sh.rmtree(outTemp, ignore_errors=True) # Delet temporal results   
                        
                else: ## One label in Atlas Mask  
                
                    convManual = ['clitkImageConvert.exe','-i', Orig_like, '-o', Orig_like, '-t', 'uchar']
                    subprocess.run(convManual)
                   
                    #Measures
                    HMap = os.path.join(Out,'HausMAP_Patient'+keyname+'.nii')
                    haus = ['clitkHausdorffDistance.exe', '-i', Orig_like, '-j', Omaskp,'-o',HMap]
                    Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
                    Haus_M = Haus.stdout.read()               
                    
                    dice = ['clitkDice.exe','-i', Orig_like, '-j', Omaskp]
                    Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
                    Dice_M = Dice.stdout.read()                       
                    
                    try:
                        Hau.append(float(Haus_M.decode()))
                        Dic.append(float(Dice_M.decode()))            
                        
                        measures.iat[counter,1] = float(Haus_M.decode())
                        measures.iat[counter,2] = float(Dice_M.decode())*100
             
                        print('Patient'+keyname,'\n','Measure between manual and automatic segmentation','\n','Hausdorff Distance: ', Haus_M.decode(),
                          'Dice Distance: ',str(float(Dice_M.decode())*100),'\n')
                
                    except:           
                        measures.iat[counter,1] = 'Segmentation error'
                        measures.iat[counter,2] = 'Segmentation error'
                            
                        print('Segmentation error')
                #measures.drop(measures.tail(len(measures)-1).index,inplace=True)
                counter +=1
        if args.results=='test':            
            if labels>1:
                lismeans=['Mean']
                for li in range(labels):
                    n=li+1      
                    meanH=np.mean(np.array(globals()['H{}'.format(n)]))
                    stdH=np.std(np.array(globals()['H{}'.format(n)]))
                    
            
                    meanD=np.mean(np.array(globals()['D{}'.format(n)]))
                    stdD=np.std(np.array(globals()['D{}'.format(n)]))  
                    
                    print('Mean Hausdorff Label ',str(n),': ',meanH,u"\u00B1",stdH) 
                    print('Mean Dice label ',str(n),': ',meanD*100,u"\u00B1",stdD*100)
                    lismeans.append(str(round(meanH,2))+u"\u00B1"+str(round(stdH,2)))
                    lismeans.append(str(round(meanD*100,2))+u"\u00B1"+str(round(stdD*100,2)))
                
                
                # Global mean
                print('Mean Global Hausdorff: ',np.mean(np.array(Hau)),u"\u00B1",np.std(np.array(Hau))) 
                print('Mean Global Dice: ',(np.mean(np.array(Dic))),u"\u00B1",(np.std(np.array(Dic))))
                hh= str(round(np.mean(np.array(Hau)),2)) + u"\u00B1" + str(round(np.std(np.array(Hau)),2))
                lismeans.append(hh)
                dd= str(round(np.mean(np.array(Dic)),2)) + u"\u00B1" + str(round(np.std(np.array(Dic)),2))
                lismeans.append(dd)    
                   
                add_row = pd.Series(lismeans,index=ColNam)
                measures = measures.append(add_row, ignore_index=True)
                
                    
            else:            
                print('Mean Hausdorff: ',np.mean(np.array(Hau)),u"\u00B1",np.std(np.array(Hau))) 
                print('Mean Dice: ',(np.mean(np.array(Dic)))*100,u"\u00B1",(np.std(np.array(Dic)))*100)
                hh= str(round(np.mean(np.array(Hau)),2)) + u"\u00B1" + str(round(np.std(np.array(Hau)),2))
                dd= str(round(np.mean(np.array(Dic))*100,2)) + u"\u00B1" + str(round(np.std(np.array(Dic))*100,2))
                    
               
                add_row = pd.Series(['Mean',hh,dd],index=ColNam)
                measures = measures.append(add_row, ignore_index=True)
              
            if args.Main_object:
                measures.to_excel(os.path.join(fold,'MainObjectPerformanceAllFolds.xlsx'))
            else:
                measures.to_excel(os.path.join(fold,'PerformanceAllFolds.xlsx'))
            
            measures = pd.DataFrame(columns=ColNam,index=indx) ### Declarate it again for new test
                    
# mean measures 
if args.results=='all':
    if labels>1:
        lismeans=['Mean']
        for li in range(labels):
            n=li+1      
            meanH=np.mean(np.array(globals()['H{}'.format(n)]))
            stdH=np.std(np.array(globals()['H{}'.format(n)]))
            
    
            meanD=np.mean(np.array(globals()['D{}'.format(n)]))
            stdD=np.std(np.array(globals()['D{}'.format(n)]))  
            
            print('Mean Hausdorff Label ',str(n),': ',meanH,u"\u00B1",stdH) 
            print('Mean Dice label ',str(n),': ',meanD*100,u"\u00B1",stdD*100)
            lismeans.append(str(round(meanH,2))+u"\u00B1"+str(round(stdH,2)))
            lismeans.append(str(round(meanD*100,2))+u"\u00B1"+str(round(stdD*100,2)))
        
        
        # Global mean
        print('Mean Global Hausdorff: ',np.mean(np.array(Hau)),u"\u00B1",np.std(np.array(Hau))) 
        print('Mean Global Dice: ',(np.mean(np.array(Dic))),u"\u00B1",(np.std(np.array(Dic))))
        hh= str(round(np.mean(np.array(Hau)),2)) + u"\u00B1" + str(round(np.std(np.array(Hau)),2))
        lismeans.append(hh)
        dd= str(round(np.mean(np.array(Dic)),2)) + u"\u00B1" + str(round(np.std(np.array(Dic)),2))
        lismeans.append(dd)    
           
        add_row = pd.Series(lismeans,index=ColNam)
        measures = measures.append(add_row, ignore_index=True)
        
            
    else:            
        print('Mean Hausdorff: ',np.mean(np.array(Hau)),u"\u00B1",np.std(np.array(Hau))) 
        print('Mean Dice: ',(np.mean(np.array(Dic)))*100,u"\u00B1",(np.std(np.array(Dic)))*100)
        hh= str(round(np.mean(np.array(Hau)),2)) + u"\u00B1" + str(round(np.std(np.array(Hau)),2))
        dd= str(round(np.mean(np.array(Dic))*100,2)) + u"\u00B1" + str(round(np.std(np.array(Dic))*100,2))
            
       
        add_row = pd.Series(['Mean',hh,dd],index=ColNam)
        measures = measures.append(add_row, ignore_index=True)
      
    if args.Main_object:
        measures.to_excel(os.path.join(results,'MainObjectPerformanceAllFolds.xlsx'))
    else:
        measures.to_excel(os.path.join(results,'PerformanceAllFolds.xlsx'))
        
        
        
        
