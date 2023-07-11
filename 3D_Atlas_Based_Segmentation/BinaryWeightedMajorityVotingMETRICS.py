# -*- coding: utf-8 -*-
"""
Console inputs:
    input_Segmentation: Path to the segmentations main folder, which has sub-folder per patient with atlas deformed masks
    Path_Resgistrations: Path to the deformed gray intensity images main folder, which has sub-folder per patient with atlas deformed gray intensity images
    input_Atlas: Path to leave one out atlas, to find manual segmentation 
    phi: Threshold to select masks, if phi = 0 is equal to don't do label selection
    
"""

import argparse
import os
from os import listdir
import subprocess
import glob
import nibabel as nib
import numpy as np
from natsort import natsorted
from Utils import *


parser = argparse.ArgumentParser(description='Function to apply simple Majority Voting and measure Hausdorff and Dice distances between automatinc and manual segmentations')
parser.add_argument('input_Segmentation', help='Path to deformed atlas masks')
parser.add_argument('Path_Resgistrations', help='Path to the main folder with deformed gray intensity images per patient')
parser.add_argument('input_Atlas', help='Path to Atlas Folder')
parser.add_argument('phi', help='Threshold of Mutual correlation for label selection ')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')


args = parser.parse_args()

    
segmentations = args.input_Segmentation
Atlas = args.input_Atlas
phi = float(args.phi)
regist = args.Path_Resgistrations
    
    
Hau=[]
Dic=[]

for LA,LS in zip(natsorted(listdir(Atlas)),natsorted(listdir(segmentations))):
    if LA == LS:
        Manual_S = glob.glob(os.path.join(Atlas,LA)+'/*.nii.gz')[0]
        target = glob.glob(os.path.join(Atlas,LA)+'/*.nii')[0]
        print ('Processing ', LA)
           
        outF = os.path.join(os.path.dirname(segmentations),'Seg_WeightedMajoriyVoting_'+args.phi,LS)
        
        if not os.path.isdir(outF):
            os.makedirs(outF)
                
        
        print('Weighted Majority Voting applying label selection with threshold ',phi)
            
        path_reg = os.path.join(regist,LA)
        PathSel,ris,MInf = LabelSelection(path_reg,target,phi)   # Label selection
        
        pathSel_Seg=[]
        for paths in natsorted(PathSel): # Convert paths to gray images in path to segmentations 
            spt=splitall(paths)
            spt1=[r if r!=os.path.basename(regist) else os.path.basename(segmentations) for r in spt]
            listToStr = os.path.sep.join(map(str, spt1))
            pathSel_Seg.append(listToStr)

              
        WMVMask = BinaryWeightedMajorityVoting(pathSel_Seg,ris,outF)
        
        Omaskp = os.path.join(outF,'OPEN_'+os.path.basename(WMVMask))
        op = ['clitkMorphoMath.exe','-i',WMVMask,'-o',Omaskp,'-t','3']
        subprocess.run(op)

                    
        #Measures
        haus = ['clitkHausdorffDistance.exe', '-i', Manual_S, '-j', Omaskp]
        Haus= subprocess.Popen(haus, stdout=subprocess.PIPE)
        Haus_M = Haus.stdout.read()
        
        convManual = ['clitkImageConvert.exe', Manual_S, '-o', Manual_S, '-t', 'uchar']
        subprocess.run(convManual)
        
        
        dice = ['clitkDice.exe','-i', Manual_S, '-j', Omaskp]
        Dice = subprocess.Popen(dice, stdout=subprocess.PIPE)
        Dice_M = Dice.stdout.read()
        
        Hau.append(float(Haus_M.decode()))
        Dic.append(float(Dice_M.decode()))
        
        print('Measure between manual and automatic segmentation',
              '\n','Hausdorff Distance: ', Haus_M.decode(), 'Dice Distance: ',str(float(Dice_M.decode())*100),'\n')
                   
print('Mean Hausdorff: ',np.mean(np.array(Hau)),u"\u00B1",np.std(np.array(Hau))) 
print('Mean Dice: ',(np.mean(np.array(Dic)))*100,u"\u00B1",(np.std(np.array(Dic)))*100)


