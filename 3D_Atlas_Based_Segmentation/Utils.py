### Function to change parameters in Transformparameter file
import numpy as np
import os
import subprocess
import nibabel as nib
from natsort import natsorted
import shutil as sh
from numpy.lib.stride_tricks import as_strided
import math
import time
from os.path import dirname as up
from skimage.measure import label
import sys


def ChangeParameters(fileIN,fileOUT,ListParameters):
    '''
    :param fileIN: Name of the original file
    :param fileOUT: Name of the new file with changes
    :param ListParameters: List of parameters to change
    :return:
    '''
    print('ChangeParameters runing...')
    
    fin = open(fileIN, "rt")
    fout = open(fileOUT, "wt")
    
    contLinesChanged = 0 # Count how many lines were changed, they must be the same of len(ListParameters)
    lineChangedPosition = []#It will saved the position of the lines
    for idx,line in enumerate(fin):
        done = False 
        for i in range(len(ListParameters)):#Look all parameters in current line                       
            if ListParameters[i][0] in line:#Is our sentence in that line?
                fout.write(ListParameters[i][1]+'\n')#Replace that line with the new one
                done = True
                contLinesChanged = contLinesChanged+1
                lineChangedPosition.append(idx)
        if done==False:#If it didn't find any parameter to change, just write the original line
            fout.write(line)
                        

    fin.close()
    fout.close()
    print(str(contLinesChanged)+' lines changed: '+str(lineChangedPosition))
    

##### Function to split paths

def splitall(path):
    '''
    This function splits one path in a list of their respective directories
    Parameters
    ----------
    path : TYPE path
        single path or list of paths to be splited

    Returns
    -------
    TYPE list
        list of one path splited or list of lists with paths splitted.

    '''
    if np.size(path)==1: # If just one path
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path: # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts
    else: # If there are multiple paths
        allpartsList = []
        for current_path in path:
            allparts = []
            while 1:
                parts = os.path.split(current_path)
                if parts[0] == current_path:  # sentinel for absolute paths
                    allparts.insert(0, parts[0])
                    break
                elif parts[1] == current_path: # sentinel for relative paths
                    allparts.insert(0, parts[1])
                    break
                else:
                    current_path = parts[0]
                    allparts.insert(0, parts[1])
            allpartsList.append(allparts)
        return allpartsList  


def NumInString(string):
    List = [ i for i,x in enumerate(string) if x.isdigit() == bool('True')]    
    name = string[List[0]:List[-1]+1]
    return name, List[0]
    
def LabelSelection(path_reg, TargetGray,phi,Best_N_Masks,MetricS,MetricW=None):
    
    """input:
            path_reg: path to folder with gray intensity images resulting from registration between atlas and target image
            TargetGray: path to the target gray intensity image
        
        Output:
            PathSel: Phats to selected images
            ri: score given to each resulting registration"""
            
    if MetricS == 'NCC':
        MetricSelection = 'clitkNormalizedCorrelationCoefficient.exe'
    elif MetricS == 'NMI':
        MetricSelection = 'clitkNormalizedMutualInformation.exe' 
    elif MetricS == 'MI':
        MetricSelection = 'clitkMattesMutualInformation.exe'
    elif MetricS == 'MS':
        MetricSelection = 'clitkMeanSquares.exe'
    if MetricW == 'NCC':
        MetricWeight = 'clitkNormalizedCorrelationCoefficient.exe'
    elif MetricW == 'NMI':
        MetricWeight = 'clitkNormalizedMutualInformation.exe' 
    elif MetricW == 'MI':
        MetricWeight = 'clitkMattesMutualInformation.exe'
    elif MetricW == 'MS':
        MetricWeight = 'clitkMeanSquares.exe'
        
    if MetricW == None or MetricS == MetricW:
        metric = []
        pathsAll = []   
        for i in natsorted(os.listdir(path_reg)):
              regn = os.path.join(path_reg,i)
              pathsAll.append(regn)  ##Gray images 
              
              Lcitk = [MetricSelection, '-i', TargetGray, '-j', regn]
              miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
              out = miN.communicate()[0]
              
              metric.append(abs(float((out.decode()).split()[-1])))
              
        MetricSel = np.array(metric)  
        
        if MetricS != 'MS':
            ri = list(MetricSel/max(MetricSel))
        elif MetricS == 'MS':
            MetricSel = MetricSel*-1
            ri = (MetricSel-min(MetricSel))/(max(MetricSel)-min(MetricSel))
        
        # Selestion considering Best_N_Masks (Sort approach)
        if Best_N_Masks=='True':
            OPathSelAll = [pathsAll for _,pathsAll in sorted(zip(ri,pathsAll),reverse=True)]
            OPathSel = OPathSelAll[0:int(phi)]
            Osri = sorted(ri,reverse=True)[0:int(phi)]
            OMetricSelAll = [metric for _,metric in sorted(zip(ri,metric),reverse=True)]
            OMetricSel = OMetricSelAll[0:int(phi)]
        else:
            # Selestion considering threshold
            OPathSel=[pathsAll[s] for s in range(len(ri)) if ri[s] >= float(phi)]
            Osri = [ri[s] for s in range(len(ri)) if ri[s] >= float(phi)]
            OMetricSel = [metric[s] for s in range(len(ri)) if ri[s] >= float(phi)]
            
            
    else:
        metricSel = []
        metricPeso = []
        pathsAll = []     
        for i in natsorted(os.listdir(path_reg)):
            regn = os.path.join(path_reg,i)
            pathsAll.append(regn)
       
            ### Compute metric for selection
            Lcitk = [MetricSelection, '-i', TargetGray, '-j', regn]
            miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
            outSel = miN.communicate()[0]
        
            ### Compute metric for weights
            Lcitk = [MetricWeight, '-i', TargetGray, '-j', regn]
            miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
            outW = miN.communicate()[0]
                                
            metricSel.append(abs(float((outSel.decode()).split()[-1])))
            metricPeso.append(abs(float((outW.decode()).split()[-1])))
        
        ## Weights
        MetricSel = np.array(metricSel)
        MetricWei = np.array(metricPeso)
        if MetricS != 'MS':
            riSel = list(MetricSel/max(MetricSel))
        elif MetricS == 'MS':
            MetricSel = MetricSel*-1
            riSel = (MetricSel-min(MetricSel))/(max(MetricSel)-min(MetricSel))
        if MetricW != 'MS':
            riw = list(MetricWei/max(MetricWei))
        elif MetricW == 'MS':
            MetricWei = MetricWei*-1
            riw = (MetricWei-min(MetricWei))/(max(MetricWei)-min(MetricWei))
        
        print('Pesos All: ',riw)
        ### Selection
        if Best_N_Masks=='True':
            OPathSelAll = [pathsAll for _,pathsAll in sorted(zip(riSel,pathsAll),reverse=True)]
            OPathSel = OPathSelAll[0:int(phi)]            
            sriw = [riw for _,riw in sorted(zip(riSel,riw),reverse=True)]
            Osri = sriw[0:int(phi)]
            Sel_Met_W = [MetricWei for _,MetricWei in sorted(zip(riSel,MetricWei),reverse=True)]
            OMetricSel = Sel_Met_W[0:int(phi)]
            
        else:
            OPathSel=[pathsAll[s] for s in range(len(riSel)) if riSel[s] >= float(phi)]
            Osri = [riw[s] for s in range(len(riSel)) if riSel[s] >= float(phi)]
            OMetricSel = [MetricWei[s] for s in range(len(riSel)) if riSel[s] >= float(phi)]
            
        print('Sel Rutas: ',OPathSel,'\n')
        print('Sel Rutas: ',Osri,'\n')
        
    return OPathSel, Osri, OMetricSel

def LabelSelection2(path_reg, TargetGray,phi,Best_N_Masks,MetricS=None,MetricW=None,Warped_masks=None):
    
    """input:
            path_reg: path to folder with gray intensity images resulting from registration between atlas and target image
            TargetGray: path to the target gray intensity image
        
        Output:
            PathSel: Phats to selected images
            ri: score given to each resulting registration"""
            
    if MetricS=='SSIM' and Warped_masks is None:
        print("SSIM metric  requieres Warped_masks input")
        sys.exit()
            
    if MetricS == 'NCC':
        MetricSelection = 'clitkNormalizedCorrelationCoefficient.exe'
    elif MetricS == 'NMI':
        MetricSelection = 'clitkNormalizedMutualInformation.exe' 
    elif MetricS == 'MI':
        MetricSelection = 'clitkMutualInformation.exe'
    elif MetricS == 'MS':
        MetricSelection = 'clitkMeanSquares.exe'
    elif MetricS == 'SSIM':
        MetricSelection = 'clitkStructuralSimilarityIndex.exe'
    if MetricW:
        if MetricW == 'NCC':
            MetricWeight = 'clitkNormalizedCorrelationCoefficient.exe'
        elif MetricW == 'NMI':
            MetricWeight = 'clitkNormalizedMutualInformation.exe' 
        elif MetricW == 'MI':
            MetricWeight = 'clitkMutualInformation.exe'
        elif MetricW == 'MS':
            MetricWeight = 'clitkMeanSquares.exe'
        elif MetricW == 'SSIM':
            MetricWeight = 'clitkStructuralSimilarityIndex.exe'
        
    if MetricW == None or MetricS == MetricW:
        metric = []
        pathsAll = []   
        for i in natsorted(os.listdir(path_reg)): # All registrations for patient n
              regn = os.path.join(path_reg,i)
              pathsAll.append(regn)  ##Gray images 
              
              if MetricS == 'SSIM':
                  temporal = os.path.join(up(up(Warped_masks)),'Temporal'+str(phi)+os.path.basename(os.path.dirname(TargetGray))) # Temporal folder to save croped images and SSI maps
                  if not os.path.isdir(temporal):
                      os.makedirs(temporal)
                   
                   # Compute and save SSIM 
                  NameMapSSI = os.path.join(temporal,'SSImap'+i)
                  Lcitk = [MetricSelection,'-s', '-i', TargetGray, '-j', regn,'-o',NameMapSSI,
                           '--minimum','0','--maximum','1']
                  subprocess.run(Lcitk)
                  # dilate mask
                  Dilate = os.path.join(temporal,'Dilate'+i)
                  Lcitk = ['clitkMorphoMath.exe','-i',os.path.join(Warped_masks,i),'-o',Dilate,'-t','1','-m','10']
                  subprocess.run(Lcitk)
                  # AutoCrop Dilated mask
                  AutoCrop = os.path.join(temporal,'CropMask'+i)
                  Lcitk = ['clitkAutoCrop.exe','-i',Dilate,'-o',AutoCrop]
                  subprocess.run(Lcitk)
                  #Crop SSI Map
                  cropSSIM = os.path.join(temporal,'CropSSIMap'+i)
                  Lcitk = ['clitkCropImage.exe','-i',NameMapSSI,'-o',cropSSIM,'--like',AutoCrop]
                  subprocess.run(Lcitk)
                  # Mean of croped SSI map
                  Lcitk = ['clitkImageStatistics.exe', '-i',cropSSIM]
                  miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
                  out = miN.communicate()[0]
                  AllMetrics=out.decode()
                  met = AllMetrics.split('\n')
                  
                  metric.append(met[2])
                  sh.rmtree(temporal, ignore_errors=True)  
                  
                  
              else:
                  Lcitk = [MetricSelection, '-i', TargetGray, '-j', regn]              
                  miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
                  out = miN.communicate()[0]
                  metric.append(abs(float((out.decode()).split()[-1])))
                 
               
              
              
        MetricSel = np.array(metric,dtype=float)        
        
        # Selestion considering Best_N_Masks (Sort approach)
        if Best_N_Masks=='True':
            OPathSelAll = [pathsAll for _,pathsAll in sorted(zip(list(MetricSel),pathsAll),reverse=True)]
            
            if MetricS != 'MS':               
                OPathSel = OPathSelAll[0:int(phi)]
                OMetricSel = sorted(list(MetricSel),reverse=True)[0:int(phi)]
            else:
                OPathSel = OPathSelAll[-int(phi):]
                OMetricSel = sorted(list(MetricSel),reverse=True)[-int(phi):]
                
            
        else:
            # Selestion considering threshold
            Range = max(MetricSel)-min(MetricSel)
            porc = Range*phi
            # Descartamos el threshold % de las peores máscaras
            if MetricS != 'MS':
                OPathSel=[pathsAll[s] for s in range(len(pathsAll)) if MetricSel[s] >= (porc+min(MetricSel))]
                OMetricSel = [MetricSel[s] for s in range(len(pathsAll)) if MetricSel[s] >= (porc+min(MetricSel))]
                
            # Si la metrica es MS tomamos el threshold % de las mejores máscaras
            elif MetricS == 'MS':
                OPathSel=[pathsAll[s] for s in range(len(pathsAll)) if MetricSel[s] <= (porc+min(MetricSel))]
                OMetricSel = [MetricSel[s] for s in range(len(pathsAll)) if MetricSel[s] <= (porc+min(MetricSel))]
            
            
    else:
        metricSel = []
        metricPeso = []
        pathsAll = []     
        for i in natsorted(os.listdir(path_reg)):
            regn = os.path.join(path_reg,i)
            pathsAll.append(regn)
       
            ### Compute metric for selection
            if MetricS == 'SSIM':
                
                temporal = os.path.join(up(up(Warped_masks)),'Temporal') # Temporal folder to save croped images and SSI maps
                if not os.path.isdir(temporal):
                    os.makedirs(temporal)
                 
                 # Compute and save SSIM 
                NameMapSSI = os.path.join(temporal,'SSImap'+i)
                Lcitk = [MetricSelection,'-s', '-i', TargetGray, '-j', regn,'-o',NameMapSSI,
                         '--minimum','0','--maximum','1']
                subprocess.run(Lcitk)
                # dilate mask
                Dilate = os.path.join(temporal,'Dilate'+i)
                Lcitk = ['clitkMorphoMath.exe','-i',os.path.join(Warped_masks,i),'-o',Dilate,'-t','1','-m','10']
                subprocess.run(Lcitk)
                # AutoCrop Dilated mask
                AutoCrop = os.path.join(temporal,'CropMask'+i)
                Lcitk = ['clitkAutoCrop.exe','-i',Dilate,'-o',AutoCrop]
                subprocess.run(Lcitk)
                #Crop SSI Map
                cropSSIM = os.path.join(temporal,'CropSSIMap'+i)
                Lcitk = ['clitkCropImage.exe','-i',NameMapSSI,'-o',cropSSIM,'--like',AutoCrop]
                subprocess.run(Lcitk)
                # Mean of croped SSI map
                Lcitk = ['clitkImageStatistics.exe', '-i',cropSSIM]
                miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
                out = miN.communicate()[0]
                AllMetrics=out.decode()
                met = AllMetrics.split('\n')
                
                metricSel.append(met[2])  
                sh.rmtree(temporal, ignore_errors=True)                
                  
            else:
                Lcitk = [MetricSelection, '-i', TargetGray, '-j', regn]              
                miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
                out = miN.communicate()[0]
                metricSel.append(abs(float((out.decode()).split()[-1])))
        
            ### Compute metric for weights
            
            if MetricW == 'SSIM':
                
                temporal = os.path.join(up(up(Warped_masks)),'Temporal') # Temporal folder to save croped images and SSI maps
                if not os.path.isdir(temporal):
                    os.makedirs(temporal)
                 
                 # Compute and save SSIM 
                NameMapSSI = os.path.join(temporal,'SSImap'+i)
                Lcitk = [MetricWeight,'-s', '-i', TargetGray, '-j', regn,'-o',NameMapSSI,
                         '--minimum','0','--maximum','1']
                subprocess.run(Lcitk)
                # dilate mask
                Dilate = os.path.join(temporal,'Dilate'+i)
                Lcitk = ['clitkMorphoMath.exe','-i',os.path.join(Warped_masks,i),'-o',Dilate,'-t','1','-m','10']
                subprocess.run(Lcitk)
                # AutoCrop Dilated mask
                AutoCrop = os.path.join(temporal,'CropMask'+i)
                Lcitk = ['clitkAutoCrop.exe','-i',Dilate,'-o',AutoCrop]
                subprocess.run(Lcitk)
                #Crop SSI Map
                cropSSIM = os.path.join(temporal,'CropSSIMap'+i)
                Lcitk = ['clitkCropImage.exe','-i',NameMapSSI,'-o',cropSSIM,'--like',AutoCrop]
                subprocess.run(Lcitk)
                # Mean of croped SSI map
                Lcitk = ['clitkImageStatistics.exe', '-i',cropSSIM]
                miN = subprocess.Popen(Lcitk, stdout=subprocess.PIPE)
                out = miN.communicate()[0]
                AllMetrics=out.decode()
                met = AllMetrics.split('\n')
                
                metricPeso.append(met[2])   
                sh.rmtree(temporal, ignore_errors=True)
                  
            else:
                Lcitkw = [MetricWeight, '-i', TargetGray, '-j', regn]              
                miN = subprocess.Popen(Lcitkw, stdout=subprocess.PIPE)
                out = miN.communicate()[0]
                metricPeso.append(abs(float((out.decode()).split()[-1])))
                              

        ## Weights
        MetricSel = np.array(metricSel,dtype=float)
        MetricWei = np.array(metricPeso,dtype=float)
        
        ### Selection
        if Best_N_Masks=='True':
            OPathSelAll = [pathsAll for _,pathsAll in sorted(zip(list(MetricSel),pathsAll),reverse=True)]
            # Devuelvo los pesos que se necesitan para weighted majority voting o patch wmv
            Sel_Met_W = [MetricWei for _,MetricWei in sorted(zip(list(MetricSel),MetricWei),reverse=True)]
            
            if MetricS != 'MS':                
                OPathSel = OPathSelAll[:int(phi)]     
                
                OMetricSel = Sel_Met_W[:int(phi)]
            else:
                OPathSel = OPathSelAll[-int(phi):]     
                
                OMetricSel = Sel_Met_W[-int(phi):]       
            
        else:
            Range = max(MetricSel)-min(MetricSel)
            porc = Range*phi
            # Descartamos el threshold % de las peores máscaras
            if MetricS != 'MS':
                OPathSel=[pathsAll[s] for s in range(len(MetricSel)) if MetricSel[s] >= (porc+min(MetricSel))]                
                OMetricSel = [MetricWei[s] for s in range(len(MetricSel)) if MetricSel[s] >= (porc+min(MetricSel))]
                
                # Si la metrica es MS tomamos el threshold % de las mejores máscaras
            elif MetricS == 'MS':
                OPathSel=[pathsAll[s] for s in range(len(MetricSel)) if MetricSel[s] <= (porc+min(MetricSel))]
                OMetricSel = [MetricWei[s] for s in range(len(MetricSel)) if MetricSel[s] <= (porc+min(MetricSel))]
            
    '''OpathSel = path a máscaras seleccionada   OMetricSel = Metrica que se necesita para procesos de weighted majority voting'''           
    return OPathSel,OMetricSel

  
def SelectionPWMV(path_reg=None,path_seg=None,target=None,phi=None,MetricS=None,Labels=None,outF=None):
    
    if MetricS== 'SSIM':
        MetricSelection = 'clitkStructuralSimilarityIndex.exe'
        
    elif MetricS == 'NCC':
        MetricSelection = 'clitkNormalizedCorrelationCoefficient.exe'
        
    # Compute SSI o NCC map between target and All image from Atlas
    AllMaps =[]
    namesIm = []
    for ni,i in enumerate(natsorted(os.listdir(path_reg))): # Patient from Atlas
        regn = os.path.join(path_reg,i)
        outT = os.path.join(up(up(path_reg)),'Temporal_p_'+str(phi)+'_'+os.path.basename(target).split('.')[0])        
        if not os.path.isdir(outT):
            os.makedirs(outT)
        if os.path.isfile(regn): 
            #Compute metric map between target and deformed image ni from Atlas
            o = os.path.join(outT,'Map'+i.split('.')[0]+'.nii.gz')
            if not os.path.isfile(o):
                Lcitk = [MetricSelection,'-s', '-i', target, '-j', regn,'-o',o,
                                '--minimum','0','--maximum','1']
                subprocess.run(Lcitk)
            AllMaps.append(o)
            namesIm.append(i)
            # Open All maps 
            globals()['map{}'.format(ni)]= nib.load(o).get_fdata()
  
    # find maximum and minimum by pixel position to compute Range (Selection)
    if phi > 0:
        for m in range(len(AllMaps)-1):
            if m == 0:
                m1 = globals()['map{}'.format(m)]
                m2 = globals()['map{}'.format(m+1)]
                minMap = np.minimum(m1,m2)
                maxMap = np.maximum(m1,m2)
            else:
                mn1 = minMap
                mx1 = maxMap
                m2 = globals()['map{}'.format(m+1)]            
                minMap = np.minimum(mn1,m2)
                maxMap = np.maximum(mx1,m2)    
                
            
        # Find pixel wise range  and threshold for selection    
        Range = maxMap-minMap
        porc = Range*phi
        umb = minMap + porc
        

        # know pixels above the threshold in each map
        for u in range(len(AllMaps)):
            globals()['TF{}'.format(u)]= globals()['map{}'.format(u)]>=umb        
            
    ## Just for masks with 1 objet (0 and 1 labels)
    L0 = np.empty_like(map0)  ### final L0 image for Argmin
    L1 = np.empty_like(map0)  ### final L1 image for Armin
    
    # WMV pixel wise  >> iterate L image  id selection pixel by pixel 
    t2=time.time()
    if phi>0:
        it = np.nditer(L0, flags=['multi_index'],op_flags=['readwrite'])
        for l0 in it:        #Pxel iterator
            l0px = 0
            l1px = 0
            sml0 = 0
            sml1 = 0
            for tf in range(len(AllMaps)): # all registered images 
                if globals()['TF{}'.format(tf)][it.multi_index]==1: # Selection
                    # threshold the selected mask with respect to the labels (0 and 1)
                    for l in range(Labels+1):
                        outImg =  os.path.join(outT,'L'+str(l)+namesIm[tf])
                        if not os.path.isfile(outImg):
                            inima = os.path.join(path_seg,namesIm[tf])
                            binLabl = ['clitkBinarizeImage.exe','-i',inima,'-l',str(l),'-u',str(l),'-o',outImg]
                            subprocess.run(binLabl)
                        
                        globals()['lab{}'.format(l)]= nib.load(outImg).get_fdata()
                        if l == 0:
                            lab0p =globals()['lab{}'.format(l)]
                        else:
                            lab1p =globals()['lab{}'.format(l)]
                        
                    # Multiply weight from SSIM map with both binarized images
                    #l0px += (lab0[it.multi_index]*(globals()['map{}'.format(tf)][it.multi_index]))
                    a1=lab0p[it.multi_index]
                    b1=globals()['map{}'.format(tf)][it.multi_index]
                    l0px += a1*b1
                    sml0 += (globals()['map{}'.format(tf)][it.multi_index])
                    
                    #l1px += (lab1[it.multi_index]*(globals()['map{}'.format(tf)][it.multi_index]))
                    a2=lab1p[it.multi_index]
                    b2=globals()['map{}'.format(tf)][it.multi_index]
                    l1px +=a2*b2
                    sml1 += (globals()['map{}'.format(tf)][it.multi_index])
                    
                    
            # Save to each pixel total multiplication divided by sum >> Generate L0 and L1 
            L0[it.multi_index] =  l0px/sml0
            L1[it.multi_index] = l1px/sml1
            print(it.multi_index)
            print('v0= ',l0px/sml0,' v1= ',l1px/sml1)
        elapsed = time.time() - t2   
        print('Time Selection patient= ' + os.path.basename(target) + ' ', (elapsed/60)/60,' Hours')
        
    else:
        ### Binarize all masks with respect to 1 and 0
        mul0 = 0
        mul1 = 0
        sumMaps = 0
        for tf in range(len(AllMaps)): # all registered masks
            for l in range(Labels+1):  ## Binarize all labels inside the mask 
                outImg =  os.path.join(outT,'L'+str(l)+namesIm[tf])
                if not os.path.isfile(outImg):
                    inima = os.path.join(path_seg,namesIm[tf])
                    binLabl = ['clitkBinarizeImage.exe','-i',inima,'-l',str(l),'-u',str(l),'-o',outImg]
                    subprocess.run(binLabl)
                
                binIm= nib.load(outImg).get_fdata()
                if l == 0:
                    mul0 += (binIm* globals()['map{}'.format(tf)])
                else:
                    mul1 += (binIm *globals()['map{}'.format(tf)])
             
            ## Sum all maps to divide and generate lo and l1
            sumMaps += globals()['map{}'.format(tf)]      

        ### Divide 
        L0 = mul0/sumMaps
        L1 = mul1/sumMaps
        
    ## Save L0 and L1 with or without selection
    Mat = nib.load(target).affine
    
    lab0 = nib.Nifti1Image(L0,Mat)
    lab1 = nib.Nifti1Image(L1,Mat)
    
    nib.save(lab0,os.path.join(outT,'l0.nii'))
    nib.save(lab1,os.path.join(outT,'l1.nii'))
    
    inArg=[os.path.join(outT,'l0.nii'),os.path.join(outT,'l1.nii')]
    inArg=','.join(inArg)
        
    
    ## Aggmax to get target mask
    AuMask = os.path.join(outF,'AutoMask'+os.path.basename(outF)+'.nii.gz')
    Larg = ['clitkArgmaxImage.exe','-i', inArg, '-o', AuMask]
    subprocess.run(Larg)
    sh.rmtree(outT, ignore_errors=True)
    
    return AuMask
    

def SelectionPWMV2(path_reg=None,path_seg=None,target=None,phi=None,MetricS=None,Labels=None,outF=None):
    
    if MetricS== 'SSIM':
        MetricSelection = 'clitkStructuralSimilarityIndex.exe'
        
    elif MetricS == 'NCC':
        MetricSelection = 'clitkNormalizedCorrelationCoefficient.exe'
        
    # Compute SSI o NCC map between target and All image from Atlas
    AllMaps =[]
    namesIm = []
    for ni,i in enumerate(natsorted(os.listdir(path_reg))): # Patient from Atlas
        regn = os.path.join(path_reg,i)
        outT = os.path.join(up(up(path_reg)),'Temporal_p_'+str(phi)+'_'+os.path.basename(target).split('.')[0])        
        if not os.path.isdir(outT):
            os.makedirs(outT)
        if os.path.isfile(regn): 
            #Compute metric map between target and deformed image ni from Atlas
            o = os.path.join(outT,'Map'+i.split('.')[0]+'.nii.gz')
            if not os.path.isfile(o):
                Lcitk = [MetricSelection,'-s', '-i', target, '-j', regn,'-o',o,
                                '--minimum','0','--maximum','1']
                subprocess.run(Lcitk)
            AllMaps.append(o)
            namesIm.append(i)
            # Open All maps 
            globals()['map{}'.format(ni)]= nib.load(o).get_fdata()
  
    # find maximum and minimum by pixel position to compute Range (Selection)
    if phi > 0:
        for m in range(len(AllMaps)-1):
            if m == 0:
                m1 = globals()['map{}'.format(m)]
                m2 = globals()['map{}'.format(m+1)]
                minMap = np.minimum(m1,m2)
                maxMap = np.maximum(m1,m2)
            else:
                mn1 = minMap
                mx1 = maxMap
                m2 = globals()['map{}'.format(m+1)]            
                minMap = np.minimum(mn1,m2)
                maxMap = np.maximum(mx1,m2)    
                
            
        # Find pixel wise range  and threshold for selection    
        Range = maxMap-minMap
        porc = Range*phi
        umb = minMap + porc
        

        # know pixels above the threshold in each map
        for u in range(len(AllMaps)):
            ## Generate False-True images considering the threshold
            globals()['TF{}'.format(u)]= globals()['map{}'.format(u)]>=umb
            ## Generate FTMap image multiplying TF image by respective SSIMap
            globals()['MapTF{}'.format(u)] = globals()['map{}'.format(u)] * globals()['TF{}'.format(u)]
            
    ## Just for masks with 1 objet (0 and 1 labels)
        
    # WMV pixel wise  >> iterate L image  id selection pixel by pixel 
    t2=time.time()
    if phi>0:
        SumTFMaps = 0
        l0 = 0
        l1 = 0
        for m in range(len(os.listdir(path_seg))):  ## Al the deformed masks from atlas             
            SumTFMaps += globals()['MapTF{}'.format(m)]
            for l in range(Labels+1):
                outImg =  os.path.join(outT,'L'+str(l)+namesIm[m])
                if not os.path.isfile(outImg):
                    inima = os.path.join(path_seg,namesIm[m])
                    binLabl = ['clitkBinarizeImage.exe','-i',inima,'-l',str(l),'-u',str(l),'-o',outImg]
                    subprocess.run(binLabl)
                
                if l == 0:
                    l0 += nib.load(outImg).get_fdata() * globals()['MapTF{}'.format(m)]
                    
                if l == 1:
                    l1 += nib.load(outImg).get_fdata() * globals()['MapTF{}'.format(m)]
                    
        L0 = l0/SumTFMaps
        L1 = l1/SumTFMaps
        
        elapsed = time.time() - t2   
        print('Time Selection patient= ' + os.path.basename(target) + ' ', (elapsed/60)/60,' Hours')
        
    else:
        ### Binarize all masks with respect to 1 and 0
        mul0 = 0
        mul1 = 0
        sumMaps = 0
        for tf in range(len(AllMaps)): # all registered masks
            for l in range(Labels+1):  ## Binarize all labels inside the mask 
                outImg =  os.path.join(outT,'L'+str(l)+namesIm[tf])
                if not os.path.isfile(outImg):
                    inima = os.path.join(path_seg,namesIm[tf])
                    binLabl = ['clitkBinarizeImage.exe','-i',inima,'-l',str(l),'-u',str(l),'-o',outImg]
                    subprocess.run(binLabl)
                
                binIm= nib.load(outImg).get_fdata()
                if l == 0:
                    mul0 += (binIm* globals()['map{}'.format(tf)])
                else:
                    mul1 += (binIm *globals()['map{}'.format(tf)])
             
            ## Sum all maps to divide and generate lo and l1
            sumMaps += globals()['map{}'.format(tf)]      

        ### Divide 
        L0 = mul0/sumMaps
        L1 = mul1/sumMaps
        
    ## Save L0 and L1 with or without selection
    Mat = nib.load(target).affine
    
    lab0 = nib.Nifti1Image(L0,Mat)
    lab1 = nib.Nifti1Image(L1,Mat)
    
    nib.save(lab0,os.path.join(outT,'l0.nii'))
    nib.save(lab1,os.path.join(outT,'l1.nii'))
    
    inArg=[os.path.join(outT,'l0.nii'),os.path.join(outT,'l1.nii')]
    inArg=','.join(inArg)
        
    
    ## Aggmax to get target mask
    AuMask = os.path.join(outF,'AutoMask'+os.path.basename(outF)+'.nii.gz')
    Larg = ['clitkArgmaxImage.exe','-i', inArg, '-o', AuMask]
    subprocess.run(Larg)
    sh.rmtree(outT, ignore_errors=True)
    
    return AuMask    
    
    
    
def BinaryWeightedMajorityVoting(Paths_Selected_Masks,ri_scores,out_Path):
    '''Inputs: 
        Paths_Selected_Masks: Path to folder with masks you want to include in WeightedMajorityVoting
        ri_scores: Similarity score between Target and Atlas registration
        out_Path: Path to save resulting mask
        
      Outputs:
          P_res: Path to resulting mask
        '''
    
    resulP=os.path.join(out_Path,'Steps')
    if not os.path.isdir(resulP):
        os.makedirs(resulP)  
    rixLAb=[]
    rixLAb_Com=[]
    sumris = sum(np.array(ri_scores))
    # Multuply masks by the respective ri score
    for i,im in enumerate(natsorted(Paths_Selected_Masks)):
        
        BxS = ['clitkImageArithm.exe','-i',im,'-o',os.path.join(resulP,'riX'+os.path.basename(im)),
               '-t','1','-s',str(ri_scores[i]),'-f']
        subprocess.run(BxS)
        rixLAb.append(os.path.join(out_Path,'Steps','riX'+os.path.basename(im)))
      
        Mask = nib.load(im).get_fdata()
        matrizM=nib.load(im).affine
        MaskcompXri = (1-Mask)*ri_scores[i] # clitkBinarizeImage.exe culd be used to get complement
        MaskcompNII = nib.Nifti1Image(MaskcompXri,matrizM)
        nib.save(MaskcompNII,os.path.join(resulP,'riX_COM_'+os.path.basename(im)))
        rixLAb_Com.append(os.path.join(resulP,'riX_COM_'+os.path.basename(im)))
     
    # Sum Masks multiplied by ri scores 
    

    if len(rixLAb)==1:
        outm1=rixLAb[0]
        outCm1=rixLAb_Com[0]
    else:       
        
        for s in range(len(rixLAb)-1):
            if s==0:
                # Sum Maks
                outm1=os.path.join(resulP,'Sum_Masks'+str(s)+'.nii')
                sumM = ['clitkImageArithm.exe','-i',rixLAb[s],'-j',rixLAb[s+1],'-o',outm1,'-t','0']
                subprocess.run(sumM)
                
                #Sum Complementary masks
                outCm1=os.path.join(resulP,'Sum_Masks_Com'+str(s)+'.nii')
                sumCM = ['clitkImageArithm.exe','-i',rixLAb_Com[s],'-j',rixLAb_Com[s+1],'-o',outCm1,'-t','0']
                subprocess.run(sumCM)
            else:
                # Sum Maks
                outm=os.path.join(resulP,'Sum_Masks'+str(s)+'.nii')
                sumM = ['clitkImageArithm.exe','-i',outm1,'-j',rixLAb[s+1],'-o',outm,'-t','0']
                subprocess.run(sumM)
                outm1 = outm
                
                #Sum Complementary masks
                outCm=os.path.join(resulP,'Sum_Masks_Com'+str(s)+'.nii')
                sumCM = ['clitkImageArithm.exe','-i',outCm1,'-j',rixLAb_Com[s+1],'-o',outCm,'-t','0']
                subprocess.run(sumCM)
                outCm1 = outCm
                
                
    # generate l1 and l0 dividing by sumris
    l1=['clitkImageArithm.exe','-i',outm1,'-s',str(sumris),'-t','11','-o',os.path.join(resulP,'l1.nii'),'-f']
    subprocess.run(l1)
    
    l0=['clitkImageArithm.exe','-i',outCm1,'-s',str(sumris),'-t','11','-o',os.path.join(resulP,'l0.nii'),'-f']
    subprocess.run(l0)
    
    P_res=os.path.join(out_Path,'Result_Mask_WMV.nii')
    ResM=['clitkBinarizeImage.exe','-i', os.path.join(resulP,'l0.nii'),'-o',P_res,'-u','0.5']
    subprocess.run(ResM)
    
    return P_res
    


def PC_MRA_Frame(path_Mgnitude,path_vx,path_vy,path_vz,out):
    '''Function to compute PC_MRA in systole frame, the out folder must has the subject name and specific number (ej: Patient1, Healthy1)
    All Paths are to nii image (path_Mgnitude = Systole Frame)'''
    
    outSteps = os.path.join(out,'Steeps')
    if not os.path.isdir(outSteps):
        os.makedirs(outSteps)
        
    
    #Normalize Images
    N_Mag = ['clitkImageArithm.exe', '-i', path_Mgnitude, '-t', '12', '-o',os.path.join(outSteps,'Norm_Magnitude.nii'),'-f']
    subprocess.run(N_Mag)
    
    N_vx = ['clitkImageArithm.exe', '-i', path_vx, '-t', '12', '-o',os.path.join(outSteps,'Norm_PhaseX.nii'),'-f']
    subprocess.run(N_vx)
    
    N_vy = ['clitkImageArithm.exe', '-i', path_vy, '-t', '12', '-o',os.path.join(outSteps,'Norm_PhaseY.nii'),'-f']
    subprocess.run(N_vy)
    
    N_vz = ['clitkImageArithm.exe', '-i', path_vz, '-t', '12', '-o',os.path.join(outSteps,'Norm_PhaseZ.nii'),'-f']
    subprocess.run(N_vz)
    
    # Compute PC_MRA 
    fx2 = os.path.join(outSteps,'x2'+os.path.basename(path_vx))
    x2=['clitkImageArithm.exe', '-i',os.path.join(outSteps,'Norm_PhaseX.nii'),'-j',os.path.join(outSteps,'Norm_PhaseX.nii'),'-t','1','-o',fx2]
    subprocess.run(x2)
    
    fy2 = os.path.join(outSteps,'y2'+os.path.basename(path_vy))
    y2=['clitkImageArithm.exe', '-i',os.path.join(outSteps,'Norm_PhaseY.nii'),'-j',os.path.join(outSteps,'Norm_PhaseY.nii'),'-t','1','-o',fy2]
    subprocess.run(y2)  
    
    fz2 = os.path.join(outSteps,'z2'+os.path.basename(path_vz))
    z2=['clitkImageArithm.exe', '-i',os.path.join(outSteps,'Norm_PhaseZ.nii'),'-j',os.path.join(outSteps,'Norm_PhaseZ.nii'),'-t','1','-o',fz2]       
    subprocess.run(z2)

    li=os.path.basename(out)
    keyname,namestr = NumInString(li)
    
    x2y2=['clitkImageArithm.exe','-i',fx2,'-j',fy2,'-t','0','-o',os.path.join(outSteps,'X2Y2'+li[0:namestr]+keyname+'.nii')]
    subprocess.run(x2y2)
                                                                                             
    x2y2z2=['clitkImageArithm.exe','-i',os.path.join(outSteps,'X2Y2'+li[0:namestr]+keyname),'-j',fz2,'-t','0','-o',os.path.join(outSteps,'X2Y2Z2'+li[0:namestr]+keyname)+'.nii']
    subprocess.run(x2y2z2)
    
    sqrt= ['clitkImageArithm.exe','-i', os.path.join(outSteps,'X2Y2Z2'+li[0:namestr]+keyname),'-t','9','-o',os.path.join(outSteps,'sqrt'+li[0:namestr]+keyname)+'.nii']
    subprocess.run(sqrt)
    
    mXsqrt = ['clitkImageArithm.exe','-i',os.path.join(outSteps,'Norm_Magnitude.nii'), '-j', os.path.join(outSteps,'sqrt'+li[0:namestr]+keyname+'.nii'),'-t','1','-o',os.path.join(out,'S'+keyname+'GRAY.nii')]
    subprocess.run(mXsqrt)    
       
    sh.rmtree(outSteps, ignore_errors=True)
    
    
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
    

   
def MultiLabWeightedMajorityVoting(Paths_Selected_Masks,ri_scores,out_Path,numLab):
    imagesArgM = []
    for s in range(numLab+1):
        globals()['paths_Threshold_{}'.format(s)] = []
    # Folder to save steps for every patient   
    TemOut=os.path.join(out_Path,'Temporal_Results')
    if not os.path.isdir(TemOut):
        os.makedirs(TemOut)
    # Binarize each selected mask with respect to each label >>   wi* D(c,MaskTransf)
    for i,m in enumerate(natsorted(Paths_Selected_Masks)):  # Selested masks      
        for l in range(numLab+1): # labels into mask
            outImg =  os.path.join(TemOut,'SelectedImage'+str(i)+'_Lab'+str(l)+'.nii.gz')
            binLabl = ['clitkBinarizeImage.exe','-i',m,'-l',str(l),'-u',str(l),'-o',outImg]
            subprocess.run(binLabl)
            
            # Multiply thresholded-selected image (recpect to label l) by the respetive weght
            outBxW = os.path.join(TemOut,'MulxW_'+'SelectedImage'+str(i)+'_Lab'+str(l)+'.nii.gz')
            mulImg = ['clitkImageArithm.exe','-i',outImg, '-t', '1','-s',str(ri_scores[i]),'-o',outBxW,'-f']
            subprocess.run(mulImg)
            globals()['paths_Threshold_{}'.format(l)].append(outBxW)
            
    # Sum images from same label     
    sumris = sum(np.array(ri_scores))
    for mw in range(numLab+1): # Labels into masks + 1 for background
        if len(globals()['paths_Threshold_{}'.format(mw)])==1: # If selected masks = 1 sum process isn't necesary
            outS=globals()['paths_Threshold_{}'.format(mw)][0]
            
            # generate ln normalized dividing by sumris =  sum all weights 
            
            ln=['clitkImageArithm.exe','-i',outS,'-s',str(sumris),'-t','11','-o',os.path.join(TemOut,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(TemOut,'l'+str(mw)+'.nii'))
                
        else:
            for n in range(len(natsorted(globals()['paths_Threshold_{}'.format(mw)]))-1):
                if n==0:
                    # Sum Maks
                    outS=os.path.join(TemOut,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n]
                    ImS1 = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',ImS,'-j',ImS1,'-o',outS,'-t','0','-f']
                    subprocess.run(sumM)
                
                else:
                    # Sum Maks
                    outS1=os.path.join(TemOut,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',outS,'-j',ImS,'-o',outS1,'-t','0','-f']
                    subprocess.run(sumM)
                    outS = outS1
                    
                    
            # generate ln normalized 

            ln=['clitkImageArithm.exe','-i',outS,'-s',str(sumris),'-t','11','-o',os.path.join(TemOut,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(TemOut,'l'+str(mw)+'.nii'))
        
    # Argmax
    arg = ['clitkArgmaxImage.exe']
    outarg = ['-o',os.path.join(out_Path,'Result_Mask_WMV2.nii')]
    inarg = arg + imagesArgM + outarg
    subprocess.run(inarg)        

    return os.path.join(out_Path,'Result_Mask_WMV2.nii'), TemOut

def sliding_window_view(arr, window_shape, steps):
    # example use y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def PatchWeighetMajorityVoting_MI(P_Selected_Images,Target,out_Path,Num_Labels,kernel=(3,3,3),stride=(1,1,1),Global_weights=None,Global_MInf=None,padding=None):
    '''Function to compute Local Mutual Information with patch-based approach, 
    if padding is different to None, the edge information is replicated 
    for padding considerig kernel size'''
    

    steps = os.path.join(out_Path,'Steps')
    if not os.path.isdir(steps):
        os.makedirs(steps)
    
    if padding==None:
        print('Padding = No, Kernel = ',kernel, ', Stride = ',stride)
        
        Y_Arrays = []
        
        # Extract Patchs for Target image
        ImT = nib.load(Target).get_fdata()
        ImT2 = np.array(ImT).copy()
        MT =nib.load(Target).affine
        YT = sliding_window_view(ImT2, window_shape=kernel, steps=stride)
        
        # Extract patchs for each selected image
        NamesWImages=[] 
        for i,im in enumerate(natsorted(P_Selected_Images)):            
            Imn = nib.load(im).get_fdata()
            Im2 = np.array(Imn).copy() 
            name = os.path.basename(im).split('.')[0]
            NamesWImages.append(np.zeros(np.shape(Im2), dtype=float))               
            y = sliding_window_view(Im2, window_shape=kernel, steps=stride)
            Y_Arrays.append(y)
        
        # Generate Image to save sum of weitghs from all selected images
        
        SumImage = np.zeros(np.shape(Im2), dtype=float) 
        s=np.asarray(np.shape(y))    
        st=list(stride)
        
        t = time.time()
        # Go through Y_Arrays and save all patches for all images
        for d in range(s[0]): # rows
            if d == 0:
                cof=math.floor(s[3]/2)
            else:
                cof+=st[0]
                
            for d1 in range(s[1]): # columns
                if d1 == 0:
                    coc=math.floor(s[4]/2)
                else:
                    coc+=st[1]
                for d2 in range(s[2]): # deep
                    if d2 == 0:
                        cod=math.floor(s[5]/2)
                    else:
                        cod+=st[2]
                    
                    MIPn = []
                    for yi, Y in enumerate(Y_Arrays):# Selected images                     
                        if yi == 0:
                            # Extract Patch for Target image
                            Patch_T = YT[d,d1,d2]
                            Nii_PT = nib.Nifti1Image(Patch_T,MT)
                            PPT = os.path.join(steps,'P_T_'+str(d)+'_'+str(d1)+'_'+str(d2)+'.nii')
                            nib.save(Nii_PT,PPT)
                        # Extract Patch for Nth Selected image   
                        Patch_Sn = Y[d,d1,d2]
                        Nii_PSn = nib.Nifti1Image(Patch_Sn,MT)
                        PPS = os.path.join(steps,'P_S'+str(yi)+'_'+str(d)+'_'+str(d1)+'_'+str(d2)+'.nii')
                        nib.save(Nii_PSn,PPS)                        
                    
                        # Calculate Mutual Information between Patch_T and Patch_Sn
                        try:
                            L1 = ['clitkMattesMutualInformation.exe', '-i', PPT, '-j', PPS]
                            GMI = subprocess.Popen(L1, stdout=subprocess.PIPE)
                            #keyboard.press('enter')                            
                            GMI_M = GMI.communicate()[0]
                            L=((GMI_M.decode()).split(': ')[-1])
                            MIPn.append(abs(float(L.rstrip())))
                            #os.remove(PPS)
                            
                        except:                            
                            MIPn.append('error')
                            #os.remove(PPS)
                            
                        
                        
                        os.remove(PPS)
                    
                    # assign the wight to the same patch to each selected image
                                            
                    if 'error' in MIPn:
                        mutualInWerror=[Global_MInf[it] if mu=='error' else float(mu) for it,mu in enumerate(MIPn)]
                        mutualInWerror = np.array(mutualInWerror)
                        ri = mutualInWerror/np.max(mutualInWerror)
                        sum_Patch = sum(ri)
                    else:
                        mutualIn = np.array(MIPn)
                        ri = mutualIn/np.max(mutualIn)
                        sum_Patch = sum(ri)
                    
                    # Put weights inside Resulting array
                    for ir,rim in enumerate(NamesWImages):
                        try:
                            NamesWImages[ir][cof:cof+st[0],coc:coc+st[1],cod:cod+st[2]] = ri[ir]
                            SumImage[cof:cof+st[0],coc:coc+st[1],cod:cod+st[2]] = sum_Patch
               
                        except:
                            NamesWImages[ir][cof,coc,cod]=ri[ir]
                            SumImage[cof,coc,cod] = sum_Patch
        
        # Put global weights on the edges and save resulting images with weights to each patient and the general sum image
        
        # Sum Image
        if Global_weights != None:
            GInd = np.where(SumImage == 0)
            SumImage[GInd] = sum(Global_weights)
            
            Nii_Sum = nib.Nifti1Image(SumImage,MT)
            PSumImage = os.path.join(steps,'sum_Image.nii')
            nib.save(Nii_Sum,PSumImage)
        
        # Resulting Images
        Images_W_Paths = []
        for e,rim2 in enumerate(NamesWImages):
            if Global_weights != None:
                ind = np.where(rim2==0)
                rim2[ind] = Global_weights[e]                
            
            Nii_WPSn = nib.Nifti1Image(rim2,MT)
            PWPS = os.path.join(steps,'W_Sel_Im_'+str(e)+'.nii')
            nib.save(Nii_WPSn,PWPS)
            Images_W_Paths.append(PWPS)
            

        
    else: 
        print('El padding es igual a ',padding)
    
    
    ###########################
    # After Patch approach to get local weights, conventional Weighted Majority Voting is applied  
    
    imagesArgM = []
    for ss in range(Num_Labels+1):
        globals()['paths_Threshold_{}'.format(ss)] = [] # Save paths paths to binarized images relative to the label x (lx image)
    
    pathSel_Seg=[]
    for paths in natsorted(P_Selected_Images): # Convert paths to gray images in path to segmentations 
        spt=splitall(paths)
        spt1=[r if r!='DeformedGray' else 'Segmentations' for r in spt]
        listToStr = os.path.sep.join(map(str, spt1))
        pathSel_Seg.append(listToStr)
        
    # Binarize each selected mask with respect to each label
    for ii,m in enumerate(natsorted(pathSel_Seg)):  # Selested masks      
        for l in range(Num_Labels+1): # labels into mask
            outImg =  os.path.join(steps,'SelectedImage'+str(ii)+'_Lab'+str(l)+'.nii.gz')
            binLabl = ['clitkBinarizeImage.exe','-i',m,'-l',str(l),'-u',str(l),'-o',outImg]
            subprocess.run(binLabl)
            
            convManual = ['clitkImageConvert.exe', '-i',outImg, '-o', outImg, '-t', 'float']
            subprocess.run(convManual)
            
            # Multiply thresholded-selected image (recpect to label l) by the respetive weght
            outBxW = os.path.join(steps,'MulxWIm_'+'SelectedImage'+str(ii)+'_Lab'+str(l)+'.nii.gz')
            mulImg = ['clitkImageArithm.exe','-i',outImg, '-t', '1','-j',Images_W_Paths[ii],'-o',outBxW,'-f']
            subprocess.run(mulImg)
            globals()['paths_Threshold_{}'.format(l)].append(outBxW)
            
    # Sum images from same label         
    for mw in range(Num_Labels+1): # Labels into masks + 1 for background
        if len(globals()['paths_Threshold_{}'.format(mw)])==1: # If selected masks = 1 sum process isn't necesary
            outS=globals()['paths_Threshold_{}'.format(mw)][0]
            
            # generate ln normalized dividing by SumImage =  sum all weights from all selected images
            
            ln=['clitkImageArithm.exe','-i',outS,'-j',PSumImage,'-t','2','-o',os.path.join(steps,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(steps,'l'+str(mw)+'.nii'))
                
        else:
            for n,mm in enumerate(natsorted(globals()['paths_Threshold_{}'.format(mw)])):
                if n==0:
                    # Sum Maks
                    outS=os.path.join(steps,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n]
                    ImS1 = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',ImS,'-j',ImS1,'-o',outS,'-t','0','-f']
                    subprocess.run(sumM)
                elif n==len(globals()['paths_Threshold_{}'.format(mw)])-1:
                    break
                else:
                    # Sum Maks
                    outS1=os.path.join(steps,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',outS,'-j',ImS,'-o',outS1,'-t','0','-f']
                    subprocess.run(sumM)
                    outS = outS1
        
                    
            # generate ln normalized 

            ln=['clitkImageArithm.exe','-i',outS,'-j',PSumImage,'-t','2','-o',os.path.join(steps,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(steps,'l'+str(mw)+'.nii'))
    
    #sh.rmtree(steps, ignore_errors=True)
    # Argmax
    arg = ['clitkArgmaxImage.exe']
    outarg = ['-o',os.path.join(out_Path,'Result_Mask_Patch_WMV.nii')]
    inarg = arg + imagesArgM + outarg
    subprocess.run(inarg)  

    elapsed = time.time() - t       
    print('Time = ',(elapsed/60),' minutes')      

    return os.path.join(out_Path,'Result_Mask_Patch_WMV.nii'),steps
        

def PatchWeighetMajorityVoting_NCC(P_Selected_Images,Target,out_Path,Num_Labels,kernel=[3,3,3],stride='1'):
    t = time.time()    
    steps = os.path.join(out_Path,'Steps')
    if not os.path.isdir(steps):
        os.makedirs(steps)
        
    print('Kernel = ',kernel, ', Stride = ',stride)
    
    K =  ','.join(kernel)
    
    ## Generate NCC images
    list_NCCAbs = []
    for i,im in enumerate(natsorted(P_Selected_Images)): 
        ## Compute NCC images 
        outNCCim = os.path.join(steps,'NCC_Im_SelIM_'+str(i)+'.nii')        
        argNCCIm = ['clitkNormalizedCorrelationCoefficient.exe','-i',im,'-j',Target,'-k',K,'-s',stride,'-o',outNCCim,'-f']
        subprocess.run(argNCCIm)
        
        ## Absolute value 
        outAbs = os.path.join(steps,'Abs_NCC_Im_SelIM_'+str(i)+'.nii')
        argAbs = ['clitkImageArithm.exe','-i',outNCCim,'-t','5','-s','0','-o',outAbs]
        subprocess.run(argAbs)
        list_NCCAbs.append(outAbs)
        
    ## Look for maximum between all NCC images to divide  and generate ri images 
    
    if len(list_NCCAbs)==1:
        maxIM=list_NCCAbs[0]
                   
    else:            
        for m in range(len(list_NCCAbs)-1):
            if m==0:
                # Maximum between NCCimages
                maxIM1=os.path.join(steps,'MAX_NCC'+str(m)+'.nii')
                Mx = ['clitkImageArithm.exe','-i',list_NCCAbs[m],'-j',list_NCCAbs[m+1],'-o',maxIM1,'-t','3']
                subprocess.run(Mx)                
              
            else:
                 # Maximum between NCCimages
                maxIM=os.path.join(steps,'MAX_NCC'+str(m)+'.nii')
                Mx = ['clitkImageArithm.exe','-i',maxIM1,'-j',list_NCCAbs[m+1],'-o',maxIM,'-t','3']
                subprocess.run(Mx)
                maxIM1 = maxIM
               
    ## Divide NCC images by MAX image to obtain ri images
    
    listRiIm = []
    for it,d in enumerate(natsorted(list_NCCAbs)):
        outDiv = os.path.join(steps,'Div_NCC_Sel(ri)'+str(it)+'.nii')
        listDiv = ['clitkImageArithm.exe','-i',d,'-j',maxIM,'-t','2','-o',outDiv]
        subprocess.run(listDiv)
        listRiIm.append(outDiv)
        
    ## Sum Ri images to divide and generate lx image
    
    if len(listRiIm)==1:
        sumRi1=listRiIm[0]        
    else:          
        for sr in range(len(listRiIm)-1):
            if sr==0:
                # Sum Maks
                sumRi1=os.path.join(steps,'Sum_Ri'+str(sr)+'.nii')
                sumM = ['clitkImageArithm.exe','-i',listRiIm[sr],'-j',listRiIm[sr+1],'-o',sumRi1,'-t','0']
                subprocess.run(sumM)

            else:
                # Sum Maks
                sumRi=os.path.join(steps,'Sum_Ri'+str(sr)+'.nii')
                sumM = ['clitkImageArithm.exe','-i',sumRi1,'-j',listRiIm[sr+1],'-o',sumRi,'-t','0']
                subprocess.run(sumM)
                sumRi1 = sumRi
                
            
        
    ###########################
    # After Patch approach to get local weights, conventional Weighted Majority Voting is applied  
    
    imagesArgM = []
    for ss in range(Num_Labels+1):
        globals()['paths_Threshold_{}'.format(ss)] = [] # Save paths paths to binarized images relative to the label x (lx image)
    
    pathSel_Seg=[]
    for paths in natsorted(P_Selected_Images): # Convert paths to gray images in path to segmentations 
        spt=splitall(paths)
        spt1=[r if r!='DeformedGray_Af_Bs_R0' else 'Segmentations_Af_Bs_R0' for r in spt] ########### Cambiar
        listToStr = os.path.sep.join(map(str, spt1))
        pathSel_Seg.append(listToStr)
        
    # Binarize each selected mask with respect to each label
    for ii,mm in enumerate(natsorted(pathSel_Seg)):  # Selested masks      
        for l in range(Num_Labels+1): # labels into mask
            outImg =  os.path.join(steps,'SelectedImage'+str(ii)+'_Lab'+str(l)+'.nii.gz')
            binLabl = ['clitkBinarizeImage.exe','-i',mm,'-l',str(l),'-u',str(l),'-o',outImg]
            subprocess.run(binLabl)
            
            convManual = ['clitkImageConvert.exe', '-i',outImg, '-o', outImg, '-t', 'float']
            subprocess.run(convManual)
            
            # Multiply thresholded-selected image (recpect to label l) by the respetive weight
            outBxW = os.path.join(steps,'MulxWIm_'+'SelectedImage'+str(ii)+'_Lab'+str(l)+'.nii.gz')
            mulImg = ['clitkImageArithm.exe','-i',outImg, '-t', '1','-j',listRiIm[ii],'-o',outBxW,'-f']
            subprocess.run(mulImg)
            globals()['paths_Threshold_{}'.format(l)].append(outBxW)
            
    # Sum images from same label         
    for mw in range(Num_Labels+1): # Labels into masks + 1 for background
        if len(globals()['paths_Threshold_{}'.format(mw)])==1: # If selected masks = 1 sum process isn't necesary
            outS=globals()['paths_Threshold_{}'.format(mw)][0]
            
            # generate ln normalized dividing by SumImage =  sum all weights from all selected images
            
            ln=['clitkImageArithm.exe','-i',outS,'-j',sumRi1,'-t','2','-o',os.path.join(steps,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(steps,'l'+str(mw)+'.nii'))
                
        else:
            for n in range(len(natsorted(globals()['paths_Threshold_{}'.format(mw)]))-1):
                if n==0:
                    # Sum Maks
                    outS=os.path.join(steps,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n]
                    ImS1 = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',ImS,'-j',ImS1,'-o',outS,'-t','0','-f']
                    subprocess.run(sumM)
                
                else:
                    # Sum Maks
                    outS1=os.path.join(steps,'Sum_Masks'+str(n)+'lab'+str(mw)+'.nii')
                    ImS = globals()['paths_Threshold_{}'.format(mw)][n+1]
                    sumM = ['clitkImageArithm.exe','-i',outS,'-j',ImS,'-o',outS1,'-t','0','-f']
                    subprocess.run(sumM)
                    outS = outS1
        
                    
            # generate ln normalized 

            ln=['clitkImageArithm.exe','-i',outS,'-j',sumRi1,'-t','2','-o',os.path.join(steps,'l'+str(mw)+'.nii'),'-f']
            subprocess.run(ln)
            imagesArgM.append('-i')
            imagesArgM.append(os.path.join(steps,'l'+str(mw)+'.nii'))
    
    #sh.rmtree(steps, ignore_errors=True)
    # Argmax
    arg = ['clitkArgmaxImage.exe']
    outarg = ['-o',os.path.join(out_Path,'Result_Mask_Patch_WMV_NCC.nii')]
    inarg = arg + imagesArgM + outarg
    subprocess.run(inarg)  

    elapsed = time.time() - t       
    print('Time = ',(elapsed/60),' minutes')      

    return os.path.join(out_Path,'Result_Mask_Patch_WMV_NCC.nii'),steps
        
    
def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    try:
        largest=max(list_seg, key=lambda x:x[1])[0]
        out=(labels == largest).astype(int)
    except:
        out = segmentation        
    return out
        
        
        
        
    
    
    
    
    
    
    
    
    
    
          
            

    
    