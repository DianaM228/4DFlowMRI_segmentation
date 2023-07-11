# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:43:18 2021

@author: Diana_MARIN
"""

import argparse
import os
from os import listdir
import nibabel as nib
import glob
import statistics as st
from natsort import natsorted
import pandas as pd
import numpy as np
import subprocess


parser = argparse.ArgumentParser(description='Function to resize data considering mean or median dataset sizes')
required_named_args = parser.add_argument_group('required named arguments')
required_named_args.add_argument('-i', '--input_path', help='path to the directory where is the image database',required=True)
required_named_args.add_argument('-o', '--output_path', help='path to the directory where you want to save the resized dataset',required=True)
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-s','--Size', nargs='+', help='List of new size for x,y,z dimensions', default=None)
optional_named_args.add_argument('-m', '--mean_size', help='Resize the images considering the mean size of x and y cordinates (default = median)', default=None,action='store_true')



args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
new_size = args.Size
metric = args.mean_size

directory = set([os.path.dirname(p) for p in glob.glob(input_path+"/*/*")]) #### It's to know if all the images are in folders by patient or together

#### Get patients axis sizes 
ColNam=['Patient','x','y','z']
measures = pd.DataFrame(columns=ColNam,index=(np.arange(len(listdir(input_path)))))
X = []
Y = []
Z = []

for pat,i in enumerate(natsorted(listdir(input_path))):
    patient = os.path.join(input_path,i)
    if len(directory) != 0:         ## if folder per patient
        image = glob.glob(os.path.join(patient)+'/*.nii')[0]
    else:                           ## if one folder for all images
        image = os.path.join(patient)
    Array = nib.load(image).get_fdata()
    x,y,z = Array.shape
    X.append(x)
    Y.append(y)
    Z.append(z)
    measures.iat[pat, 0] = os.path.basename(image)
    measures.iat[pat, 1] = x
    measures.iat[pat, 2] = y
    measures.iat[pat, 3] = z
        

if metric==None:    
    print('median size x: ',st.median(X) , '\n', 'median size y: ',st.median(Y), '\n','maximum Z: ', max(Z))
    if new_size==None:
        size_x = st.median(X)
        size_y = st.median(Y)
        size_z = max(Z)
    else:
        size_x = new_size[0]
        size_y = new_size[1]
        size_z = new_size[2]       
        
else:
    print('mean size x: ',st.mean(X) , '\n', 'mean size y: ',st.mean(Y), '\n','maximum Z: ', max(Z))
    if new_size==None:
        size_x = st.mean(X)
        size_y = st.mean(Y)
        size_z = max(Z)
    else:
        size_x = new_size[0]
        size_y = new_size[1]
        size_z = new_size[2]
    

measures.to_csv(os.path.join(os.path.dirname(output_path),'Patients_Original_Axis_Size.csv'), index=False)


##### Resize images 
generalOut = os.path.join(output_path,'ResizedData_'+str(size_x)+'x'+str(size_y)+'x'+str(size_z))

if len(directory) != 0:    
    OutGray = os.path.join(generalOut,'Images')
    if not os.path.isdir(OutGray):
        os.makedirs(OutGray)
    OutMask = os.path.join(generalOut,'Masks')
    if not os.path.isdir(OutMask):
        os.makedirs(OutMask)
else:
    if not os.path.isdir(generalOut):
        os.makedirs(generalOut)
    
if len(directory) == 0:  #### If all images in a folder
    for p in natsorted(os.listdir(input_path)):
        patient=os.path.join(input_path,p)
        save = os.path.join(generalOut,p)
        resampling = ['clitkExtractPatch.exe', '-i', patient,'-o',save,'-s ' + str(size_x) + ',' + str(size_y)+ ',' + str(size_z)]            
        subprocess.run(resampling)
        
else:
    for p in natsorted(os.listdir(input_path)): # Patient folder
        patient=os.path.join(input_path,p)
        
        for v in natsorted(os.listdir(patient)): # folder with Gray and Mask Images
            im = os.path.normpath(os.path.join(patient,v))
            
            if 'GRAY' in v:
                save = os.path.normpath(os.path.join(OutGray,os.path.basename(im)))
            elif 'MASK' in v:
                save = (os.path.join(OutMask,os.path.basename(im)))
            else:
                'The name of the image must have the word GRAY or MASK to indicate where to save it'            
                  
            resampling = ['clitkExtractPatch.exe', '-i', im,'-o',save,'-s ' + str(size_x) + ',' + str(size_y)+ ',' + str(size_z)]            
            subprocess.run(resampling)
            
        