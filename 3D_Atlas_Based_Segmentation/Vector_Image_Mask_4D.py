# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:10:27 2020

@author: Diana_MARIN
"""

import argparse
import os
import subprocess
from natsort import natsorted
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser(description='Convert Phase images to vector image using a mask')
parser.add_argument('input_path_clitk', help='path to the directory where the clitk tools are')
parser.add_argument('input_path_nifti', help='path to main directory with patients in nifti format')
parser.add_argument('input_path_dicom', help='path to main directory with patients in dicom format')




args = parser.parse_args()


Path_clitk = args.input_path_clitk
Path_nii = args.input_path_nifti
Path_dcm = args.input_path_dicom

for pn in natsorted(os.listdir(Path_nii)): # Patient nifti
    
    # Create 2 lists with pahts to 3 Phases images on Dicom and Nii format 
        p_input_nii = os.path.join(Path_nii, pn)
        p_input_dcm = os.path.join(Path_dcm, pn)
        phases_nii=[]
        phases_dcm=[]
        
        for pnii,pdcm in zip(natsorted(os.listdir(p_input_nii)),natsorted(os.listdir(p_input_dcm))):
            if '_P_' in pnii:
                phases_nii.append(os.path.join(p_input_nii, pnii)) # list with phat to phase sequences in nii                
                
            if '_P_' in pdcm:
                phases_dcm.append(os.path.join(p_input_dcm, pdcm)) # list with phat to phase sequences in dcm
                
        # Look for mask phat in patient path
            
        L_masks=[]
        for pack in os.walk(p_input_nii):
                for f in pack[2]:
                    if f.endswith('.gz'):
                        fullpath = pack[0] + "\\" + f
                        L_masks.append(fullpath)
                                
        
        # Start iteration over Phases 
                
        for ph in range(len(phases_dcm)):            
            NII_Phase=phases_nii[ph]
            DCM_Phase=phases_dcm[ph]
            
            sliceDCM=os.path.join(DCM_Phase,(natsorted(os.listdir(DCM_Phase))[0]))
            
            # Get dicom tags
            p0 = subprocess.Popen([os.path.join(Path_clitk,"clitkDicomInfo"), sliceDCM], stdout=subprocess.PIPE)
            output = p0.communicate()[0]
            output = output.decode()
            
            Tags=[] # Has Bits, RescaleIntercept, RescaleSlope and Velocity
            
            for tag in output.splitlines():                
                    if "(0051,1014)" in tag: #(velocity)
                        instanceRow_splited = tag.split("_")
                        instanceRow_splited = instanceRow_splited[0].split("v")
                        instance_number = instanceRow_splited[1].strip()
                        Tags.append(instance_number)
                        
                    elif "Bits Stored" in tag: #(28,0101)
                        instanceRow_splited = tag.split("#")
                        instanceRow_splited = instanceRow_splited[0].split("S")
                        instance_number = instanceRow_splited[1].strip()
                        Tags.append(instance_number)
                        
                    
                    elif "Rescale Intercept" in tag: #(28,1052)
                        instanceRow_splited = tag.split("]")
                        instanceRow_splited = instanceRow_splited[0].split("[")
                        instance_number = instanceRow_splited[1].strip()
                        Tags.append(instance_number)
                    
                    elif "Rescale Slope" in tag: #(28,1053)
                        instanceRow_splited = tag.split("]")
                        instanceRow_splited = instanceRow_splited[0].split("[")
                        instance_number = instanceRow_splited[1].strip()
                        Tags.append(instance_number)
             
            pen= ( (int(Tags[3])*-1 ) - (int(Tags[3])) ) /( (int(Tags[1]))-(int(Tags[2])*(2**int(Tags[0])-1) + int(Tags[1])) )# -> Velmin-Velmax / Intmin-Intmax
            
            # Generate folder to save velocity images
            path_save_vel=os.path.abspath((os.path.join(os.path.join(p_input_nii,NII_Phase,'Velocity_Images_'+os.path.basename(NII_Phase)))))
            if not os.path.isdir(path_save_vel):
                os.mkdir(path_save_vel)
                
            # Generate folder to save velocity images multiplied by mask
            path_save_velMASK=os.path.abspath((os.path.join(os.path.join(p_input_nii,NII_Phase,'Velocity_Images_Mask_'+os.path.basename(NII_Phase)))))
            if not os.path.isdir(path_save_velMASK):
                os.mkdir(path_save_velMASK)
                
              
            for vol in (natsorted(os.listdir(NII_Phase))): 
                if '.nii' in vol:
                    # Multiply image by slope >> velIma
                    path_vol=os.path.join(NII_Phase,vol)
                    clitk_list = [os.path.join(Path_clitk, 'clitkImageArithm.exe')]
                    outputV = os.path.join(path_save_vel,vol)
                    params_list = ['-i', path_vol, '-s', str(pen), '-t', '1', '-o', outputV,'-f']
                    lista_Aritm = clitk_list + params_list
                    subprocess.run(lista_Aritm)
                                        
                    # Multiply velIma by mask                    
                    outputVM = os.path.join(path_save_velMASK,vol)
                    params_listM = ['-i', outputV, '-j', L_masks[0], '-t', '1', '-o', outputVM,'-f' ]
                    lista_AritmM = clitk_list + params_listM
                    subprocess.run(lista_AritmM)
                    
                            
        # Generate vector image     
        
        # make folder to save Vector images 
        if not os.path.isdir(os.path.join(p_input_nii,'Vector_Images')):
            os.mkdir(os.path.join(p_input_nii,'Vector_Images'))
                    
                    
        # Look for paths where the Vel*mask images were saved for each direction
        L_phases_VxM=[]
        for pack in os.walk(p_input_nii):
                for f in pack[1]:
                    if 'Velocity_Images_Mask' in f:
                        fullpath = pack[0] + "\\" + f
                        L_phases_VxM.append(fullpath)
                        
        L_phases_VxM=natsorted(L_phases_VxM)
        
        
        # cycle through the 3 phase folders to load the times and convert them to a vector image
        LX = natsorted(os.listdir(L_phases_VxM[0]))
        LY = natsorted(os.listdir(L_phases_VxM[1]))
        LZ = natsorted(os.listdir(L_phases_VxM[2]))
        for k,(px,py,pz) in enumerate(zip(LX,LY,LZ)):
            PLX=os.path.join(L_phases_VxM[0],px)
            PLY=os.path.join(L_phases_VxM[1],py)
            PLZ=os.path.join(L_phases_VxM[2],pz)
            clitk_list = [os.path.join(Path_clitk, 'clitkImageToVectorImage.exe')]
            outputVIXYZ = os.path.join(p_input_nii,'Vector_Images','T_'+str(k)+'.nii')
            Param_Vect=[PLX,PLY,PLZ,'-o',outputVIXYZ]
            Lis_Vec=clitk_list+Param_Vect
            subprocess.run(Lis_Vec)
            

        # # Join 3D images to make 4D vector image
        clitk_list2 = [os.path.join(Path_clitk, 'clitkMergeSequence.exe')]
        L_tiemposVI = natsorted(os.listdir(os.path.join(p_input_nii,'Vector_Images')))
        
        for i in range(len(L_tiemposVI)): 
                            L_tiemposVI[i]=((os.path.join(p_input_nii,'Vector_Images')+'/'+L_tiemposVI[i]))
                            
        outP=os.path.join(p_input_nii,'Vector_Images','4D_Vector_Image.nii')
        param_M = ['-o',outP]
        L = clitk_list2 + L_tiemposVI + param_M
        subprocess.run(L)
          
            
             
                        
                        
        
                        
            




        
        
        
        
        
        
        
        
        
        