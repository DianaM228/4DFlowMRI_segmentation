# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:37:55 2021

@author: Diana_MARIN
"""

import argparse
import os
from os.path import dirname as up
import nibabel as nib
import torch.nn.functional as F
import copy
import sys
import numpy as np
from torch import nn

from UNetNetworks import UNet3DAntoine2
from UNetNetworks import UNet3DAntoine

import torch
from collections import OrderedDict
from scipy.spatial import Delaunay
from itertools import combinations
from topologylayer.functional.persistence import SimplicialComplex
import time
from custom_losses import DiceLoss2

from topologylayer.nn import LevelSetLayer, TopKBarcodeLengths, SumBarcodeLengths, PartialSumBarcodeLengths

parser = argparse.ArgumentParser(description='Function to propagate aorta boungaries')
parser.add_argument('-m','--model', help='name of the model to be used', required=True,choices=["UNet3DAntoine","UNet3DAntoine2"])
parser.add_argument('-s','--State_dict', help='Path to the saved model state dict (model.pth.tar)')
parser.add_argument('-p','--image', help='Path to image to be segmented with network input size', required=True)
parser.add_argument('-o','--output_path', help='path to folder to save results', required=True)
parser.add_argument('-w','--TopoLoss_Weight', help='Number between 0 and 1 to multoply (give relevance) to topological loss function', required=True)
parser.add_argument('-i','--iterations', help='Number of iterations with topological loss function')
parser.add_argument('-b','--Normalize_topoLoss', help='Option to normalize topo loss by selecting a number of bars b to be included in topo loss computation',default=None)
parser.add_argument('-l','--loss', help='Select classical loss function',choices=['Dice','MSE'], required=True)


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print('GPU : ',torch.cuda.is_available())

wt = args.TopoLoss_Weight

############## Functions to build 3D complex 
def unique_simplices(faces, dim):
    """
    obtain unique simplices up to dimension dim from faces
    """
    simplices = [[] for k in range(dim+1)]
    # loop over faces
    for face in faces:
        # loop over dimension
        for k in range(dim+1):
            # loop over simplices
            for s in combinations(face, k+1):
                simplices[k].append(np.sort(list(s)))

    s = SimplicialComplex()
    # loop over dimension
    for k in range(dim+1):
        kcells = np.unique(simplices[k], axis=0)
        for cell in kcells:
            s.append(cell)

    return s

def init_tri_complex3D(width, height, deep):
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    z = np.linspace(0, deep-1, deep)
    i_coords, j_coords,k_coords = np.meshgrid(x, y, z, indexing='ij')
    coordinate_grid = np.array([i_coords, j_coords, k_coords])
    coordinate_grid = coordinate_grid.reshape(3,-1) #imageDim = 2 in 2D and 3 in 3D
    coordinate_points = coordinate_grid.transpose()
    tri = Delaunay(coordinate_points)
        
    return unique_simplices(tri.simplices, 3)

##### Take the patient number from the file name
def NumInString(string):
    List = [ i for i,x in enumerate(string) if x.isdigit() == bool('True')]    
    name = string[List[0]:List[-1]+1]
    return name, List[0]

##### input parameters

network = args.model
StateDict = args.State_dict
image = args.image
out = args.output_path

### get patient number or key name to build the folder to save results
basname = (os.path.basename(image)).split('_')[-1]    
keyname,namestr = NumInString(basname)

if not args.Normalize_topoLoss:    
    output = os.path.join(out,'TopologyTunning'+args.loss,'Patient'+keyname+"wt"+wt)
else:
    output = os.path.join(out,'NormTopologyTunning'+args.loss,'Patient'+keyname+"wt"+wt)
    

if not os.path.isdir(output):
    os.makedirs(output)

#### Load the model and weights 
state_dict = torch.load(StateDict, map_location=torch.device(device))["state_dict"]

#### correct state dict names if it's necessary
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if 'module' in k:
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    else:
        name = k
        new_state_dict[name] = v


if network == 'UNet3DAntoine':
    model = UNet3DAntoine(in_channel=1, out_channel=2)  ## aorta+1 = because of the background
    print('UNet3DAntoine')
elif network == 'UNet3DAntoine2':
    model = UNet3DAntoine2(in_channel=1, out_channel=2)
    print('UNet3DAntoine2')
else:
    print('The model does not exist')
    sys.exit(1)

model.load_state_dict(new_state_dict)
model.to(device).eval()

###### Load image to be segmented and ground truth
img=nib.load(image)
image_array=img.get_fdata()
shape_img = image_array.shape
width, height, deep=np.shape(image_array)

##### Generate tensors 
np_img = image_array.reshape(1, 1, shape_img[0], shape_img[1], shape_img[2])
torch_image = torch.from_numpy(np_img).float()
torch_image = torch_image.permute(0, 1, 4, 3, 2)
torch_image = torch_image.to(device)

## Generate probability map with loaded model and save segmentation before topo tuning
with torch.no_grad():
    predOrig1 = model(torch_image) ### 1, 2, 44, 176, 146
    predOrig1.to(device)
    
original_model_output = torch.argmax(predOrig1[0, :, :, :, :], dim=0).float() # 44, 176, 146
original_model_output = original_model_output.permute(2, 1, 0)
original_model_output = torch.squeeze(original_model_output)
original_model_output = original_model_output.cpu().detach().numpy()
original_model_output = nib.Nifti1Image(original_model_output, affine=img.affine)
nib.save(original_model_output, os.path.join(output,'Original_Pred_Patient'+keyname+'.nii'))

#####
model_topo = copy.deepcopy(model)
model_topo.load_state_dict(new_state_dict)

optimizer = torch.optim.Adam(model_topo.parameters(), lr=1e-3)
num_iter_topo = int(args.iterations)

if args.loss =="Dice":
    similarity_loss = DiceLoss2(0)
    print("Similarity Metric: Dice")
elif args.loss == "MSE":
    similarity_loss = nn.MSELoss()
    print("Similarity Metric: MSE")

#### Build complex 

print('Computing complex 3D')
tcpx=time.time()
cpx = init_tri_complex3D(deep+2,height+2,width+2) #### Padding each side with zero line to avoid errors on homology computation 
elapsedcxp = time.time() - tcpx 
print('Time  complex= ',(elapsedcxp/60),' minutes')

dgminfo = LevelSetLayer(cpx, maxdim=2, sublevel=False)

### A priori knowledge of topology >> B0=1,B1=0,B2=0
H_ao = {0:1, 1:0, 2:0}  ## Aorta topology


model_topo.to(device).eval()
wt = float(wt)
wd = 1-float(wt)
l_s = []
l_t = []
l_g = []

for it in range(num_iter_topo):
    pred_topo1 = model_topo(torch_image) 
    pred_topo1.to(device)
    #### Probability MAP
    pred_topo = pred_topo1[:,1,:,:,:] 
    pred_topo = torch.reshape(pred_topo, (1, 1, shape_img[2], shape_img[1], shape_img[0])) 
    pred_topoH = F.pad(pred_topo,(1,1,1,1,1,1))
    
    ##Save prediction for each iteration
    imageTopo = torch.argmax(pred_topo1[0,:,:,:,:], dim=0).float()
    imageTopo = imageTopo.permute(2, 1, 0)
    imageTopo = torch.squeeze(imageTopo)
    imageTopo = imageTopo.cpu().detach().numpy()
    imageTopo = nib.Nifti1Image(imageTopo, affine=img.affine)
    nib.save(imageTopo, os.path.join(output,'TopoModelPredit'+str(it)+'.nii'))
    
    # Extract barcode diagram
    print('Computing Homology')
    thomo=time.time()
    a = dgminfo(pred_topoH)
    elapsedhomo = time.time() - thomo 
    print('Time  homology= ',(elapsedhomo/60),' minutes')
      
    
    if not args.Normalize_topoLoss:        
        # 0-dimensional topological features
        Z0 = (SumBarcodeLengths(dim=0)(a))
        # 1-dimensional topological features           
        Z1 = PartialSumBarcodeLengths(dim=1, skip=H_ao[1])(a) ## sum bars lengths skipping the first skip finite bars
        
        # 2-dimensional topological features           
        Z2 = PartialSumBarcodeLengths(dim=2, skip=H_ao[2])(a) ## sum bars lengths skipping the first skip finite bars
        
        
    else:
        max_k = int(args.Normalize_topoLoss)
        Z0 = (TopKBarcodeLengths(dim=0, k=max_k)(a)).sum()/max_k
      
        Z1= (TopKBarcodeLengths(dim=1, k=max_k)(a)).sum()/max_k
        
        Z2= (TopKBarcodeLengths(dim=2, k=max_k)(a)).sum()/max_k
        
           
    Sb = similarity_loss(pred_topo1, predOrig1)
    
                
    loss = wt*(Z0  + Z1  + Z2) + (wd*Sb)    
     
    optimizer.zero_grad()
    loss.backward()
    print('Loss_similarity: ',Sb)
    l_s.append(Sb.item())
    print("Topo_loss: ",(Z0  + Z1  + Z2))
    l_t.append((Z0  + Z1  + Z2).item())
    print('Global_loss',loss)    
    l_g.append(loss.item())
    #LD_list.append(loss.item())
    optimizer.step()

print("l_s: ",l_s)
print("l_g: ",l_g)

model_topo.eval()
with torch.no_grad():
    pred_topof = model_topo(torch_image)


PATH = os.path.join(os.path.dirname(args.State_dict),'TopoLossStateDictPatient'+keyname+'wt_'+str(wt)+'_it_'+args.iterations + args.loss+'.pth.tar')
torch.save(model_topo.state_dict(), PATH)
#### save segmentation after topo loss tuning

# topo_model_output = torch.argmax(pred_topof[0, :, :, :, :], dim=0).float()
# topo_model_output = topo_model_output.permute(2, 1, 0)
# topo_model_output = torch.squeeze(topo_model_output)
# topo_model_output = topo_model_output.cpu().detach().numpy()
# topo_model_output = nib.Nifti1Image(topo_model_output, affine=img.affine)
# nib.save(topo_model_output, os.path.join(output,'Topo_PredFinal_Patient_'+keyname+'.nii'))

