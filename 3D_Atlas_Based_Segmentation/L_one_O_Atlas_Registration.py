"""
Note: By default, gray intensities images are saved in nii format and masks in nii.gz

Console inputs:
    input_Atlas:  Atlas folder with one sub-folder for each patient, which has masks and gray images
    input_Parameters: Folder with all the txt files to do registration
    
"""


import argparse
import os
from os import listdir
import subprocess
import time
import shutil as sh
import glob
import sys
from natsort import natsorted
from pathlib import Path




parser = argparse.ArgumentParser(description='Function to apply leave one out for inter-patient segmentation using registration')
parser.add_argument('-at','--input_Atlas', help='Path to the Atlas')
parser.add_argument('-op_seg','--input_Atlas2', help='Path to second Atlas, if you want to deforme it with the same transformations (ej: 2 labels data)',default=None)
parser.add_argument('-af','--input_Affine', help='Path to the folder with registration parameters')
parser.add_argument('-bs','--input_Bspline', help='Path to the folder with registration parameters',default=None)
parser.add_argument('-bsm','--Multi_Images',help='Path to Bspline parameters for Multi-Images registration (second iteration)')
parser.add_argument('-c', '--Path_Clitk_Cluster',help='Path to clitk tools in the cluster')
parser.add_argument('-e ', '--Path_elastix',help='Path to elastix tools')
parser.add_argument('-t', '--Path_Folder_Target',help='Path to target patient folder with gray and mask images. Option for Cluster')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
parser.add_argument('-n', '--Path_indices', help='Path to dictionary with indices. Option to activate registration increasing number of data',default=None)


args = parser.parse_args()

t = time.time()

if args.Path_Clitk_Cluster:
    sys.path.append(os.path.normpath('/work/le2i/di4567ma/aortadiag4d/'))
    from C_Utils import *
else:
    from Utils import *


Atlas = args.input_Atlas
paramsAF = args.input_Affine
paramsBS = args.input_Bspline
Atla2 = args.input_Atlas2
indName = args.Path_indices

# key name to folder
if paramsBS:  
    kname = os.path.basename(paramsBS).split('.')[0]
else:
    kname = os.path.basename(paramsAF).split('.')[0]

if args.Path_Clitk_Cluster and args.Path_Folder_Target is None:
    if args.Path_indices:
        pass
    else:
        parser.error("--Path_Clitk_Cluster  requieres --Path_Folder_Target parameter")
    
if args.Path_Clitk_Cluster and args.Path_elastix is None:
    parser.error("--Path_Clitk_Cluster  requieres --Path_elastix parameter")


'''Run code in cluster'''

if args.Path_Clitk_Cluster:
    if args.Path_indices:
        print('analysis of number data influence activated')
        #################################################################################### Number Data analysis
        transformix =os.path.join(args.Path_elastix,'transformix')
        elastix = os.path.join(args.Path_elastix,'elastix')
        
        ###### Load dictionary with index of target and atlases 
        Index = np.load(args.Path_indices,allow_pickle='TRUE').item()
        
        ##### read paths to all patients nii images
        list_reg =[]
        for path in Path(Atlas).rglob('*.nii'):
            list_reg.append(os.path.join(Atlas,path.parts[-2],path.name))        
        list_reg =natsorted(list_reg)
        ####### Define target patient (for loop to do the process with all targets)
        for t in Index['Val'][0]: ### All target patient (Validation)
            pathTP = os.path.dirname(list_reg[t])      
        
            for file in os.listdir(pathTP):
                if file.endswith('.nii'):
                    target = os.path.normpath(os.path.join(pathTP, file))                
                elif file.endswith('.gz'):
                    OrigMask = os.path.normpath(os.path.join(pathTP, file))
                    
        
            i=os.path.basename(pathTP) # i =Patient name#
        
            GrayA = []
            masks = []
            
            for nf,f in enumerate(Index['Train']): ##### Fold in Train               
                for j in f: # Lists gray and masks with patients different to target (Atlas to be deformed)                    
                    pathAP = os.path.dirname(list_reg[j])
                    for file in os.listdir(os.path.normpath(pathAP)):
                        if file.endswith('.nii'):
                            ImAtlas = os.path.normpath(os.path.join(pathAP, file))
                            GrayA.append(ImAtlas)  
                        elif file.endswith('.gz'):
                            MaskA = os.path.normpath(os.path.join(pathAP, file))
                            masks.append(MaskA)

            ### Registration Target and each image
                print('######################################################################','\n')
                print('############ REGISTERING #######',i,'\n',GrayA)
                for k in GrayA: # Start registrations between target and image k in Atlas-i
                    PAname = splitall(k)
                    O1=os.path.normpath(Atlas).split(os.path.sep)[:-1] 
                    O1=os.path.sep.join(map(str, O1))
                    valname = (indName.split('_')[-1]).split('.')[0]
                    outf = os.path.normpath(os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P',
                                                         'OutputAll'+kname,i,'R_'+PAname[len(PAname)-2]))
                    if not os.path.isdir(outf):
                        os.makedirs(outf)
                    
                    outAffine = os.path.join(outf,'Affine')
                    ##outAffine = os.path.join(outf,'Translation')
                    if not os.path.isdir(outAffine):
                        os.makedirs(outAffine)
                    
                    if paramsBS:
                        outNoRigid = os.path.join(outf,'Bspline')
                        if not os.path.isdir(outNoRigid):
                            os.makedirs(outNoRigid)
            
                    ## Image registration            
    
                    ## Affine registration
                    
                    Affine = [elastix,'-f', target ,'-m',k,'-out',outAffine ,'-p',paramsAF]
                    subprocess.run(Affine)
                    
                    ## Bspline registration 
                    if paramsBS:
                        NoRigid = [elastix,'-f', target ,'-m',k,'-out',outNoRigid ,'-p',paramsBS,'-t0',os.path.join(outAffine,'TransformParameters.0.txt')]
                        subprocess.run(NoRigid)
                    
            ### Apply transformation to the mask from this Patient   
                            
                    Parameters = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                                  ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
                    
                    # Change parameters in TransformParameters.0.txt files
            
            
            ## If registration process was successful        
                    try:
                        ChangeParameters(os.path.join(outAffine,'TransformParameters.0.txt'),os.path.join(outAffine,'TransformParameters.0BIN.txt'),Parameters)
                        flagA = 1
                        if paramsBS:
                            ChangeParameters(os.path.join(outNoRigid,'TransformParameters.0.txt'),os.path.join(outNoRigid,'TransformParameters.0BIN.txt'),Parameters)
                            flagBs = 1
                    except:
                        FileE = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','OutputAll'+kname,'FailedRegistration'+os.path.splitext(os.path.basename(target))[0]+'.txt')
                        if not os.path.isfile(FileE):
                            txtErr = open(FileE,'w+')
                        else:
                                txtErr = open(FileE,'a')
                        txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                        txtErr.write ('Error registering -f '+ os.path.basename(target)+' -m '+ os.path.basename(k) + '\n')
                        txtErr.close ()
                        continue
                    
                                       
                    if paramsBS:
                        # Folder to save segmentations Affine + Bspline
                        SegFolder = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','Output'+kname,'Segmentations_Af_Bs_R0',i)
                        if not os.path.isdir(SegFolder):
                            os.makedirs(SegFolder)
                            
                        # Folder to save warped images Affine + Bspline
                        DefGray = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','Output'+kname,'DeformedGray_Af_Bs_R0',i)
                        if not os.path.isdir(DefGray):
                            os.makedirs(DefGray)
                            
                        # Path Output Transformix Affine + Bspline
                        pseg = os.path.join(outNoRigid,'pseg')
                        if not os.path.isdir(pseg):
                            os.makedirs(pseg)
                        
                    
                        
                        
                    basname = os.path.splitext(os.path.basename(k))[0]
                    keyname,namestr = NumInString(basname)
                    inima = os.path.join(os.path.dirname(k),basname[0:namestr]+keyname+'MASK.nii.gz')
                    
                  
                    # apply transform  Affine + Bspline to mask
                    if paramsBS:
                        Seg = [transformix,'-in',inima,'-out',pseg,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                        subprocess.run(Seg)
                        
                        
                    namR=splitall(os.path.dirname(k))[-1][0] # Name patient from Atlas
        
                    if paramsBS:
                        sh.copy(os.path.join(outNoRigid,'result.0.nii.gz'), os.path.join(DefGray,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline
                        sh.copy(os.path.join(pseg,'result.nii.gz'), os.path.join(SegFolder,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
         
                    ## If you wanna deformate another mask based in the same deformation transform (ej: 2 labels mask, removing abdominal aorta )
                    if Atla2:
                        
                        ## Folder to save all from transformix
                        out2all = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','OutputAll2'+kname,i,'R_'+PAname[len(PAname)-2])
                        if not os.path.isdir(out2all):
                                os.makedirs(out2all)
                        # Mask to warp
                        if flagA ==1:
                            outAffine2 = os.path.join(out2all,'Affine')
                            #outAffine = os.path.join(outf,'Translation')
                            if not os.path.isdir(outAffine2):
                                os.makedirs(outAffine2)
                            Apat = k.split(os.path.sep)[-2]
                            inima2 = os.path.join(Atla2,Apat,basname[0:namestr]+keyname+'MASK.nii.gz')
                            
                            
                            # Folders to save affine OUTPUT
                            out2sA = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','Output'+kname,'Segmentations_Af_R0_2',i)
                            if not os.path.isdir(out2sA):
                                os.makedirs(out2sA)
                                
                            
                            # apply transform  Affine to mask
                            SegA2 = [transformix,'-in',inima2,'-out',outAffine2,'-tp',os.path.join(outAffine,'TransformParameters.0BIN.txt')]
                            subprocess.run(SegA2)
                            
                            sh.copy(os.path.join(outAffine2,'result.nii.gz'), os.path.join(out2sA,i+'_'+namR+keyname+'.nii.gz'))
            
                        if paramsBS and flagBs==1:
                            outBsp = os.path.join(out2all,'BS')
                            #outAffine = os.path.join(outf,'Translation')
                            if not os.path.isdir(outBsp):
                                os.makedirs(outBsp)
                            out2sAB = os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','Output'+kname,'Segmentations_Af_Bs_R0_2',i)
                            if not os.path.isdir(out2sAB):
                                os.makedirs(out2sAB)
                            
                            Seg2 = [transformix,'-in',inima2,'-out',outBsp,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                            subprocess.run(Seg2)
                            
                            sh.copy(os.path.join(outBsp,'result.nii.gz'), os.path.join(out2sAB,i+'_'+namR+keyname+'.nii.gz'))
                      
                    
                    flagA = 0
                    flagBs = 0
            ################### Multi Imagenes
              
            
                    if args.Multi_Images:                
                        print('Iteration 2: Multiple images Bspline registration')
                        # Folder to save all deformed gray images after second iteration (OUT)
                        DefGray1 = os.path.join(os.path.dirname(Atlas),'Output','DeformedGray_Af_MBs_R1',i) 
                        if not os.path.isdir(DefGray1):
                            os.makedirs(DefGray1)
                        
                        # Folder to save all deformed masks for the patient after second iteration (OUT)
                        SegFolder1 = os.path.join(os.path.dirname(Atlas),'Output','Segmentations_Af_MBs_R1',i)
                        if not os.path.isdir(SegFolder1):
                            os.makedirs(SegFolder1)
                        
                        # folder to save registration result between target and patien i (OUTPUT_ALL)
                        outNoRigid1 = os.path.join(outf,'Bspline1')
                        if not os.path.isdir(outNoRigid1):
                            os.makedirs(outNoRigid1)
                        
                        # First automatic mask dilatation 
                        DilM1 = os.path.join(outNoRigid1,'DilMaks1.nii.gz')
                        ld=[os.path.join(args.Path_Clitk_Cluster,'clitkMorphoMath'),'-i',os.path.join(pseg,'result.nii.gz'),'-t','1','-o',DilM1]
                        subprocess.run(ld)
                        
                        # Second iteration with multi-images Bspline registration
                        m1 = glob.glob(os.path.dirname(k)+'/*.nii.gz')[0]
                        r2=[elastix,'-f0', target,'-f1',os.path.join(pseg,'result.nii.gz'),'-fMask',DilM1,'-m0',k,
                            '-m1',m1,'-t0',os.path.join(outAffine,'TransformParameters.0.txt'),'-p',args.Multi_Images, '-out',outNoRigid1]
                        subprocess.run(r2)
                        
                        
                        # Apply transformation to get the mask from this Patient (second iteration)   
                        ParametersR2 = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                                      ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
                                    
                        # Change parameters in TransformParameters.0.txt files
                        
                        try:
                            ChangeParameters(os.path.join(outNoRigid1,'TransformParameters.0.txt'),os.path.join(outNoRigid1,'TransformParameters.0BIN.txt'),ParametersR2)
                        
                        except:
                            FileE = os.path.join(os.path.dirname(Atlas),'IT2_FailedRegistration'+os.path.splitext(os.path.basename(target))[0]+'.txt')
                            if not os.path.isfile(FileE):
                                txtErr = open(FileE,'w+')
                            else:
                                txtErr = open(FileE,'a')
                                
                            txtErr.write ( '######  Error In second iteration  ###### \n')
                            txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                            txtErr.write ('Error registering -f0 '+ target + '\n' +'  -f1 '+os.path.join(pseg,'result.nii.gz')+ '\n'+'  -fMask '+ DilM1+ '\n'+ '  -m0 '+ k+ '\n' +
                            '-m1  '+ m1 + '\n')
                            txtErr.close ()
                            continue
                        
                        # Path Output Transformix iteration 2
                        psegR2 = os.path.join(outNoRigid1,'pseg')
                        if not os.path.isdir(psegR2):
                            os.makedirs(psegR2)
                            
                        SegR2 = [transformix,'-in',inima,'-out',psegR2,'-tp',os.path.join(outNoRigid1,'TransformParameters.0BIN.txt')]
                        subprocess.run(SegR2)
                        
                        sh.copy(os.path.join(outNoRigid1,'result.0.nii.gz'), os.path.join(DefGray1,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline multi images
                        sh.copy(os.path.join(psegR2,'result.nii.gz'), os.path.join(SegFolder1,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
                        
                oAllP =os.path.normpath(os.path.join(O1,'Result_Registro'+valname,'Test'+str(nf)+'_'+str(len(Index['Train'][0])*(nf+1))+'P','OutputAll'+kname,i))
                sh.rmtree(oAllP, ignore_errors=True)


    else: #################################################################################Leave one out evaluation
    
        transformix =os.path.join(args.Path_elastix,'transformix')
        elastix = os.path.join(args.Path_elastix,'elastix')        
        pathTP = args.Path_Folder_Target
        
        for file in os.listdir(pathTP):
            if file.endswith('.nii'):
                target = os.path.normpath(os.path.join(pathTP, file))                
            elif file.endswith('.gz'):
                OrigMask = os.path.normpath(os.path.join(pathTP, file))
                
        
        i=os.path.basename(pathTP) # i =Patient#
        
        GrayA = []
        masks = []
       
        for j in natsorted(listdir(Atlas)): # Lists gray and masks with patients different to target
            pathAP = os.path.normpath(os.path.join(Atlas,j))
            
            for file in os.listdir(os.path.normpath(pathAP)):
                if file.endswith('.nii'):
                    ImAtlas = os.path.normpath(os.path.join(pathAP, file))
                elif file.endswith('.gz'):
                    MaskA = os.path.normpath(os.path.join(pathAP, file))
                    
                    
            if os.path.basename(ImAtlas) != os.path.basename(target):          
                GrayA.append(ImAtlas)            
                masks.append(MaskA)
                
        
    ### Registration Target and each image        
        for k in GrayA: # Start registrations between target and image k in Atlas-i
            PAname = splitall(k)
            O1=os.path.normpath(Atlas).split(os.path.sep)[:-1] 
            O1=os.path.sep.join(map(str, O1))
            outf = os.path.normpath(os.path.join(O1,'Result_Registro','OutputAll'+kname,i,'R_'+PAname[len(PAname)-2]))
            if not os.path.isdir(outf):
                os.makedirs(outf)
            
            outAffine = os.path.join(outf,'Affine')
            ##outAffine = os.path.join(outf,'Translation')
            if not os.path.isdir(outAffine):
                os.makedirs(outAffine)
            
            if paramsBS:
                outNoRigid = os.path.join(outf,'Bspline')
                if not os.path.isdir(outNoRigid):
                    os.makedirs(outNoRigid)
    
    ## Image registration            
                
            
            ## Affine registration
            
            Affine = [elastix,'-f', target ,'-m',k,'-out',outAffine ,'-p',paramsAF]
            subprocess.run(Affine)
            
            ## Bspline registration 
            if paramsBS:
                NoRigid = [elastix,'-f', target ,'-m',k,'-out',outNoRigid ,'-p',paramsBS,'-t0',os.path.join(outAffine,'TransformParameters.0.txt')]
                subprocess.run(NoRigid)
            
    ### Apply transformation to the mask from this Patient   
                    
            Parameters = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                          ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
            
            # Change parameters in TransformParameters.0.txt files
    
    
    ## If registration process was successful        
            try:
                ChangeParameters(os.path.join(outAffine,'TransformParameters.0.txt'),os.path.join(outAffine,'TransformParameters.0BIN.txt'),Parameters)
                flagA = 1
                if paramsBS:
                    ChangeParameters(os.path.join(outNoRigid,'TransformParameters.0.txt'),os.path.join(outNoRigid,'TransformParameters.0BIN.txt'),Parameters)
                    flagBs = 1
            except:
                FileE = os.path.join(O1,'Result_Registro','OutputAll'+kname,'FailedRegistration'+os.path.splitext(os.path.basename(target))[0]+'.txt')
                if not os.path.isfile(FileE):
                    txtErr = open(FileE,'w+')
                else:
                        txtErr = open(FileE,'a')
                txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                txtErr.write ('Error registering -f '+ os.path.basename(target)+' -m '+ os.path.basename(k) + '\n')
                txtErr.close ()
                continue
            
                       
            if paramsBS:
                # Folder to save segmentations Affine + Bspline
                SegFolder = os.path.join(O1,'Result_Registro','Output'+kname,'Segmentations_Af_Bs_R0',i)
                if not os.path.isdir(SegFolder):
                    os.makedirs(SegFolder)
                    
                # Folder to save warped images Affine + Bspline
                DefGray = os.path.join(O1,'Result_Registro','Output'+kname,'DeformedGray_Af_Bs_R0',i)
                if not os.path.isdir(DefGray):
                    os.makedirs(DefGray)
                    
                # Path Output Transformix Affine + Bspline
                pseg = os.path.join(outNoRigid,'pseg')
                if not os.path.isdir(pseg):
                    os.makedirs(pseg)
   
                
            basname = os.path.splitext(os.path.basename(k))[0]
            keyname,namestr = NumInString(basname)
            inima = os.path.join(os.path.dirname(k),basname[0:namestr]+keyname+'MASK.nii.gz')
            

            # apply transform  Affine + Bspline to mask
            if paramsBS:
                Seg = [transformix,'-in',inima,'-out',pseg,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                subprocess.run(Seg)
                
                
            namR=splitall(os.path.dirname(k))[-1][0] # Name patient from Atlas

            if paramsBS:
                sh.copy(os.path.join(outNoRigid,'result.0.nii.gz'), os.path.join(DefGray,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline
                sh.copy(os.path.join(pseg,'result.nii.gz'), os.path.join(SegFolder,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
   
            ## If you wanna deformate another mask based in the same deformation transform (ej: 2 labels mask, removing abdominal aorta )
            if Atla2:
                
                ## Folder to save all from transformix
                out2all = os.path.join(O1,'Result_Registro','OutputAll2'+kname,i,'R_'+PAname[len(PAname)-2])
                if not os.path.isdir(out2all):
                        os.makedirs(out2all)
                # Mask to warp
                if flagA ==1:
                    outAffine2 = os.path.join(out2all,'Affine')
                    #outAffine = os.path.join(outf,'Translation')
                    if not os.path.isdir(outAffine2):
                        os.makedirs(outAffine2)
                    Apat = k.split(os.path.sep)[-2]
                    inima2 = os.path.join(Atla2,Apat,basname[0:namestr]+keyname+'MASK.nii.gz')
                    
                    
                    # Folders to save affine OUTPUT
                    out2sA = os.path.join(O1,'Result_Registro','Output'+kname,'Segmentations_Af_R0_2',i)
                    if not os.path.isdir(out2sA):
                        os.makedirs(out2sA)
                        
                    
                    # apply transform  Affine to mask
                    SegA2 = [transformix,'-in',inima2,'-out',outAffine2,'-tp',os.path.join(outAffine,'TransformParameters.0BIN.txt')]
                    subprocess.run(SegA2)
                    
                    sh.copy(os.path.join(outAffine2,'result.nii.gz'), os.path.join(out2sA,i+'_'+namR+keyname+'.nii.gz'))
    
                if paramsBS and flagBs==1:
                    outBsp = os.path.join(out2all,'BS')
                    #outAffine = os.path.join(outf,'Translation')
                    if not os.path.isdir(outBsp):
                        os.makedirs(outBsp)
                    out2sAB = os.path.join(O1,'Result_Registro','Output'+kname,'Segmentations_Af_Bs_R0_2',i)
                    if not os.path.isdir(out2sAB):
                        os.makedirs(out2sAB)
                    
                    Seg2 = [transformix,'-in',inima2,'-out',outBsp,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                    subprocess.run(Seg2)
                    
                    sh.copy(os.path.join(outBsp,'result.nii.gz'), os.path.join(out2sAB,i+'_'+namR+keyname+'.nii.gz'))
              
            
            flagA = 0
            flagBs = 0
    ################### Multi Imagenes
      
    
            if args.Multi_Images:                
                print('Iteration 2: Multiple images Bspline registration')
                # Folder to save all deformed gray images after second iteration (OUT)
                DefGray1 = os.path.join(os.path.dirname(Atlas),'Output','DeformedGray_Af_MBs_R1',i) 
                if not os.path.isdir(DefGray1):
                    os.makedirs(DefGray1)
                
                # Folder to save all deformed masks for the patient after second iteration (OUT)
                SegFolder1 = os.path.join(os.path.dirname(Atlas),'Output','Segmentations_Af_MBs_R1',i)
                if not os.path.isdir(SegFolder1):
                    os.makedirs(SegFolder1)
                
                # folder to save registration result between target and patien i (OUTPUT_ALL)
                outNoRigid1 = os.path.join(outf,'Bspline1')
                if not os.path.isdir(outNoRigid1):
                    os.makedirs(outNoRigid1)
                
                # First automatic mask dilatation 
                DilM1 = os.path.join(outNoRigid1,'DilMaks1.nii.gz')
                ld=[os.path.join(args.Path_Clitk_Cluster,'clitkMorphoMath'),'-i',os.path.join(pseg,'result.nii.gz'),'-t','1','-o',DilM1]
                subprocess.run(ld)
                
                # Second iteration with multi-images Bspline registration
                m1 = glob.glob(os.path.dirname(k)+'/*.nii.gz')[0]
                r2=[elastix,'-f0', target,'-f1',os.path.join(pseg,'result.nii.gz'),'-fMask',DilM1,'-m0',k,
                    '-m1',m1,'-t0',os.path.join(outAffine,'TransformParameters.0.txt'),'-p',args.Multi_Images, '-out',outNoRigid1]
                subprocess.run(r2)
                
                
                # Apply transformation to get the mask from this Patient (second iteration)   
                ParametersR2 = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                              ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
                            
                # Change parameters in TransformParameters.0.txt files
                
                try:
                    ChangeParameters(os.path.join(outNoRigid1,'TransformParameters.0.txt'),os.path.join(outNoRigid1,'TransformParameters.0BIN.txt'),ParametersR2)
                
                except:
                    FileE = os.path.join(os.path.dirname(Atlas),'IT2_FailedRegistration'+os.path.splitext(os.path.basename(target))[0]+'.txt')
                    if not os.path.isfile(FileE):
                        txtErr = open(FileE,'w+')
                    else:
                        txtErr = open(FileE,'a')
                        
                    txtErr.write ( '######  Error In second iteration  ###### \n')
                    txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                    txtErr.write ('Error registering -f0 '+ target + '\n' +'  -f1 '+os.path.join(pseg,'result.nii.gz')+ '\n'+'  -fMask '+ DilM1+ '\n'+ '  -m0 '+ k+ '\n' +
                    '-m1  '+ m1 + '\n')
                    txtErr.close ()
                    continue
                
                # Path Output Transformix iteration 2
                psegR2 = os.path.join(outNoRigid1,'pseg')
                if not os.path.isdir(psegR2):
                    os.makedirs(psegR2)
                    
                SegR2 = [transformix,'-in',inima,'-out',psegR2,'-tp',os.path.join(outNoRigid1,'TransformParameters.0BIN.txt')]
                subprocess.run(SegR2)
                
                sh.copy(os.path.join(outNoRigid1,'result.0.nii.gz'), os.path.join(DefGray1,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline multi images
                sh.copy(os.path.join(psegR2,'result.nii.gz'), os.path.join(SegFolder1,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
                
        oAllP =os.path.normpath(os.path.join(O1,'Result_Registro','OutputAll'+kname,i))
        sh.rmtree(oAllP, ignore_errors=True)
    

##################################################################  Run code PC    ########################################################
else:    

    for i in natsorted(listdir(Atlas)): # i is the target patient in Atlas
        print('Registering ',i)
        pathTP = os.path.join(Atlas,i)
        target = glob.glob(pathTP+'/*.nii')[0] # we just have one gray image and one mask
        OrigMask = glob.glob(pathTP+'/*.nii.gz')[0] 
        
        # get path to all gray images and masks in Atlas to register them with target (Leve one out)
        GrayA = []
        masks = []
        for j in natsorted(listdir(Atlas)):
            pathAP = os.path.join(Atlas,j)
            if j!=i:
                ImAtlas = glob.glob(pathAP+'/*.nii')[0]
                GrayA.append(ImAtlas)
                MaskA = glob.glob(pathAP+'/*.nii.gz')[0]
                masks.append(MaskA)
        
    # Registration Target and each image        
        for k in GrayA: # Start registrations between target and image k in Atlas-i
            PAname = splitall(k)        
            outf = os.path.join(os.path.dirname(Atlas),'Result_Registro','OutputAll'+kname,i,'R_'+PAname[len(PAname)-2])
            if not os.path.isdir(outf):
                os.makedirs(outf)
            
            outAffine = os.path.join(outf,'Affine')
            #outAffine = os.path.join(outf,'Translation')
            if not os.path.isdir(outAffine):
                os.makedirs(outAffine)
             
            if paramsBS:
                outNoRigid = os.path.join(outf,'Bspline')
                if not os.path.isdir(outNoRigid):
                    os.makedirs(outNoRigid)
        
    # Image registration            
                
            
            # Affine registration
            Affine = ['elastix.exe','-f', target ,'-m',k,'-out',outAffine ,'-p',paramsAF]
            subprocess.run(Affine)
            
            # Bspline registration
            if paramsBS:
                NoRigid = ['elastix.exe','-f', target ,'-m',k,'-out',outNoRigid ,'-p',paramsBS,'-t0',os.path.join(outAffine,'TransformParameters.0.txt')]
                subprocess.run(NoRigid)
            
    # Apply transformation to the mask from this Patient 
            Parameters = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                          ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
            
            # Change parameters in TransformParameters.0.txt files
    
    
    ## If registration process was successful        
            try:
                ChangeParameters(os.path.join(outAffine,'TransformParameters.0.txt'),os.path.join(outAffine,'TransformParameters.0BIN.txt'),Parameters)
                flagA = 1
                if paramsBS:    
                    ChangeParameters(os.path.join(outNoRigid,'TransformParameters.0.txt'),os.path.join(outNoRigid,'TransformParameters.0BIN.txt'),Parameters)
                    flagBs = 1
            except:
                FileE = os.path.join(os.path.dirname(Atlas),'FailedRegistration.txt')
                if not os.path.isfile(FileE):
                    txtErr = open(FileE,'w+')
                else:
                        txtErr = open(FileE,'a')
                txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                txtErr.write ('Error registering -f '+ os.path.basename(target)+' -m '+ os.path.basename(k) + '\n')
                txtErr.close ()
                continue
            
            
            # Segmentation
            
            # Folder to save segmentations Affine
            SegFolderA = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'Segmentations_Af_R0',i)
            if not os.path.isdir(SegFolderA):
                os.makedirs(SegFolderA)
                
            DefGrayAf = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'DeformedGray_Af_R0',i)
            if not os.path.isdir(DefGrayAf):
                os.makedirs(DefGrayAf)
                
            # Path Output Transformix Affine 
            psegA = os.path.join(outAffine,'pseg')
            if not os.path.isdir(psegA):
                os.makedirs(psegA) 
            
            if paramsBS:
                # Folder to save segmentations Affine + Bspline
                SegFolder = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'Segmentations_Af_Bs_R0',i)
                if not os.path.isdir(SegFolder):
                    os.makedirs(SegFolder)
                    
                # Folder to save segmentations Affine + Bspline
                DefGray = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'DeformedGray_Af_Bs_R0',i)
                if not os.path.isdir(DefGray):
                    os.makedirs(DefGray)
                    
                # Path Output Transformix Affine + Bspline
                pseg = os.path.join(outNoRigid,'pseg')
                if not os.path.isdir(pseg):
                    os.makedirs(pseg)
                
                 
            basname = os.path.splitext(os.path.basename(k))[0]
            keyname,namestr = NumInString(basname)
            inima = os.path.join(os.path.dirname(k),basname[0:namestr]+keyname+'MASK.nii.gz')
            
            # apply transform  Affine + Bspline to mask
                        
            # apply transform  Affine to mask
            SegA = ['transformix.exe','-in',inima,'-out',psegA,'-tp',os.path.join(outAffine,'TransformParameters.0BIN.txt')]
            subprocess.run(SegA)
            
            if paramsBS:
                Seg = ['transformix.exe','-in',inima,'-out',pseg,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                subprocess.run(Seg)
            
            namR=splitall(os.path.dirname(k))[-1][0]
            
            sh.copy(os.path.join(outAffine,'result.0.nii.gz'), os.path.join(DefGrayAf,i+'_'+namR+keyname+'.nii.gz')) # Save segmentationaffine
            sh.copy(os.path.join(psegA,'result.nii.gz'), os.path.join(SegFolderA,i+'_'+namR+keyname+'.nii.gz')) # Save segmentationaffine
            
            if paramsBS:
                sh.copy(os.path.join(outNoRigid,'result.0.nii.gz'), os.path.join(DefGray,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline
                sh.copy(os.path.join(pseg,'result.nii.gz'), os.path.join(SegFolder,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
                
    
            ## If you wanna deformate another mask based in the same deformation transform (ej: 2 labels mask, removing abdominal aorta )
            if Atla2:
                out2all = os.path.join(os.path.dirname(Atlas),'Result_Registro','OutputAll2'+kname,i,'R_'+PAname[len(PAname)-2])
                if not os.path.isdir(out2all):
                        os.makedirs(out2all)
                
                # Mask to warp
                if flagA ==1:
                    outAffine2 = os.path.join(out2all,'Affine')                    
                    if not os.path.isdir(outAffine2):
                        os.makedirs(outAffine2)
                        
                    Apat = k.split(os.path.sep)[-2]
                    inima2 = os.path.join(Atla2,Apat,basname[0:namestr]+keyname+'MASK.nii.gz')
                    
                    ## Folder to save all from transformix OUTPUT
                    out2sA = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'Segmentations_Af_R0_2',i)
                    if not os.path.isdir(out2sA):
                        os.makedirs(out2sA)
                                            
                    
                    # apply transform  Affine to mask
                    SegA2 = ['transformix.exe','-in',inima2,'-out',outAffine2,'-tp',os.path.join(outAffine,'TransformParameters.0BIN.txt')]
                    subprocess.run(SegA2)
                    
                    sh.copy(os.path.join(outAffine2,'result.nii.gz'), os.path.join(out2sA,i+'_'+namR+keyname+'.nii.gz'))
        
                if paramsBS and flagBs==1:
                    outBsp = os.path.join(out2all,'BS')                    
                    if not os.path.isdir(outBsp):
                        os.makedirs(outBsp)
                        
                    out2sAB = os.path.join(os.path.dirname(Atlas),'Result_Registro','Output'+kname,'Segmentations_Af_Bs_R0_2',i)
                    if not os.path.isdir(out2sAB):
                        os.makedirs(out2sAB)
                    
                    Seg2 = ['transformix.exe','-in',inima2,'-out',outBsp,'-tp',os.path.join(outNoRigid,'TransformParameters.0BIN.txt')]
                    subprocess.run(Seg2)
                    
                    sh.copy(os.path.join(outBsp,'result.nii.gz'), os.path.join(out2sAB,i+'_'+namR+keyname+'.nii.gz'))
                        
    
            flagA = 0
            flagBs = 0
    
            if args.Multi_Images:                
                print('Iteration 2: Multiple images Bspline registration')
                # Folder to save all deformed gray images after second iteration (OUT)
                DefGray1 = os.path.join(os.path.dirname(Atlas),'Output','DeformedGray_Af_MBs_R1',i) 
                if not os.path.isdir(DefGray1):
                    os.makedirs(DefGray1)
                
                # Folder to save all deformed masks for the patient after second iteration (OUT)
                SegFolder1 = os.path.join(os.path.dirname(Atlas),'Output','Segmentations_Af_MBs_R1',i)
                if not os.path.isdir(SegFolder1):
                    os.makedirs(SegFolder1)
                
                # folder to save registration result between target and patien i (OUTPUT_ALL)
                outNoRigid1 = os.path.join(outf,'Bspline1')
                if not os.path.isdir(outNoRigid1):
                    os.makedirs(outNoRigid1)
                 
                # First automatic mask dilatation 
                DilM1 = os.path.join(outNoRigid1,'DilMaks1.nii.gz')
                ld=['clitkMorphoMath.exe','-i',os.path.join(pseg,'result.nii.gz'),'-t','1','-o',DilM1]
                subprocess.run(ld)
                
                # Second iteration with multi-images Bspline registration
                m1 = glob.glob(os.path.dirname(k)+'/*.nii.gz')[0]
                r2=['elastix.exe','-f0', target,'-f1',os.path.join(pseg,'result.nii.gz'),'-fMask',DilM1,'-m0',k,
                    '-m1',m1,'-t0',os.path.join(outAffine,'TransformParameters.0.txt'),'-p',args.Multi_Images, '-out',outNoRigid1]
                subprocess.run(r2)
                
                
                # Apply transformation to get the mask from this Patient (second iteration)   
                ParametersR2 = [('FinalBSplineInterpolationOrder','(FinalBSplineInterpolationOrder 0)'),
                              ('ResultImagePixelType','(ResultImagePixelType "unsigned char")')]
                            
                # Change parameters in TransformParameters.0.txt files
                
                try:
                    ChangeParameters(os.path.join(outNoRigid1,'TransformParameters.0.txt'),os.path.join(outNoRigid1,'TransformParameters.0BIN.txt'),ParametersR2)
                
                except:
                    FileE = os.path.join(os.path.dirname(Atlas),'FailedRegistration.txt')
                    if not os.path.isfile(FileE):
                        txtErr = open(FileE,'w+')
                    else:
                        txtErr = open(FileE,'a')
                        
                    txtErr.write ( '######  Error In second iteration  ###### \n')
                    txtErr.write ( '######  '+ os.path.basename(target)+'\n')
                    txtErr.write ('Error registering -f0 '+ target + '\n' +'  -f1 '+os.path.join(pseg,'result.nii.gz')+ '\n'+'  -fMask '+ DilM1+ '\n'+ '  -m0 '+ k+ '\n' +
                    '-m1  '+ m1 + '\n')
                    txtErr.close ()
                    continue
                
                # Path Output Transformix iteration 2
                psegR2 = os.path.join(outNoRigid1,'pseg')
                if not os.path.isdir(psegR2):
                    os.makedirs(psegR2)
                    
                SegR2 = ['transformix.exe','-in',inima,'-out',psegR2,'-tp',os.path.join(outNoRigid1,'TransformParameters.0BIN.txt')]
                subprocess.run(SegR2)
                
                sh.copy(os.path.join(outNoRigid1,'result.0.nii.gz'), os.path.join(DefGray1,i+'_'+namR+keyname+'.nii.gz')) # save deforme gray after affine + bspline multi images
                sh.copy(os.path.join(psegR2,'result.nii.gz'), os.path.join(SegFolder1,i+'_'+namR+keyname+'.nii.gz'))  # Save segmentation affine+bspline
                
            
elapsed = time.time() - t
print('Time = ',(elapsed/60)/60,' hours')

if 'FileE' in globals() or 'FileE' in locals():
    print('errors during registration saved in txt file')
    
    
    

