# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 08:29:42 2022

This code compute the compliance using 4D automatic segmentations. For this purpose a 2D plane perpendicular to the ascending aorta is 
located using the coordinates of a point passing through the plane (from csv file).


@author: Diana_MARIN
"""

import argparse
import matplotlib.pyplot as plt
import os
from os import listdir
import glob
import nibabel as nib
import numpy as np
from natsort import natsorted
import pandas as pd
from sklearn import linear_model
from UtilsDL import getLargestCC
from scipy import stats
from scipy.stats import wilcoxon
import subprocess
import statsmodels.api as sm


parser = argparse.ArgumentParser(description='Function to compute number of voxel on time and analyze the movement')
parser.add_argument('-i' ,'--input', help='Path to Folder with the results of the experiment from 4D deep learning')
parser.add_argument('-f' ,'--file', help='Path to csv File with the reference compliance')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
optional_named_args = parser.add_argument_group('optional named arguments')
optional_named_args.add_argument('-r', '--resolution', help='image resolution (To compute compliance)',default=None)
optional_named_args.add_argument('--radius', help='if you want to apply opening before computing the surface enter the radius of the ball element',default=None)




args = parser.parse_args()

path_results = args.input
file = args.file

if args.resolution:
    res = float(args.resolution)
else:
    res = 1.98864

    
patientsPred = []
patientsTarg = []

dataFile = pd.read_excel(file)
df = pd.DataFrame(columns=['Patient', 'Smin', 'Smax','compliance','Distensibility'])
for ind,fold in enumerate(natsorted(listdir(path_results))): # fold or patient
    
    patient = os.path.join(path_results,fold)    
    frames = natsorted(glob.glob(patient+'/*.nii.gz'))
    
    infoPatient = dataFile.loc[dataFile['Patient'] == fold]
    point = infoPatient['Point'][ind]
    point = point.split(',') ####### x, y, z 
    
    ####### loop over patient frames fo locate 2D plane and compute compliance 
    
    FrameSurface = []
    
    if args.radius: ### comppute surfaces with pre-processing
        # print('Aplying opening with radius ',args.radius)
        for indx,f in enumerate(frames): ### time steps (extract 2D +time)
            #### load image 
            time =nib.load(f).get_fdata()  
            # Select 2D plane at the level of AAo
            Plane2D=time[:,int(point[1]),:]
            plane2d = getLargestCC(Plane2D)
            
            if indx==0:
                x,y,z = nib.load(f).shape
                Two_D_Time = np.zeros((x,z,25))
            Two_D_Time[:,:,indx] = plane2d
        
        ##### Save 2D + time
        matrizM = nib.load(f).affine
        new_Mag = nib.Nifti1Image(Two_D_Time, affine=matrizM) 
        out = os.path.join(os.path.dirname(path_results),'Compliance',os.path.basename(path_results))
        if not os.path.isdir(out):
            os.makedirs(out)
        nib.save(new_Mag,os.path.join(out,'AAOTime'+fold+'.nii')) # AAOTimePatient1
        
        ##### Apply opening to 2D+Time image
        outOpen = os.path.join(out,'AAOTime'+fold+'_r'+args.radius+'.nii')
        op = ['clitkMorphoMath.exe','-i',os.path.join(out,'AAOTime'+fold+'.nii'),
              '-o',outOpen,'-t','3','-r',args.radius]
        subprocess.run(op)
        
        ########## Compute surface using pre-processed 2D+time
        for indx in range(25): ### time steps (extract 2D +time)
            #### load image 
            time2D =nib.load(outOpen).get_fdata()
            frame2D = time2D[:,:,indx]
            pixels = np.sum(frame2D)
            FrameSurface.append(pixels*res*res)  
        
        plt.plot(FrameSurface)
        plt.title(fold)
        plt.show()
        
        Smin= min(FrameSurface)
        Smax= max(FrameSurface)
        Ps = infoPatient['PS'][ind]
        Pd = infoPatient['PD'][ind]
        Compliance= (Smax-Smin)/(Ps-Pd)
        Distensibility = (Compliance/Smin)*1000
        df = df.append({'Patient':fold, 'Smin': Smin, 'Smax': Smax,'compliance':Compliance,'Distensibility':Distensibility}, ignore_index=True)
        
        
    else: ### comppute surfaces without pre-processing        
        for indx,f in enumerate(frames): ### time steps
            #### load image 
            time =nib.load(f).get_fdata()        
            # Select 2D plane at the level of AAo
            Plane2D=time[:,int(point[1]),:]
            plane2d = getLargestCC(Plane2D)
            pixels = np.sum(plane2d)
            FrameSurface.append(pixels*res*res)      
     
        plt.plot(FrameSurface)
        plt.title(fold)
        plt.show()
        
        Smin= min(FrameSurface)
        Smax= max(FrameSurface)
        Ps = infoPatient['PS'][ind]
        Pd = infoPatient['PD'][ind]
        Compliance= (Smax-Smin)/(Ps-Pd)
        Distensibility = (Compliance/Smin)*1000
        df = df.append({'Patient':fold, 'Smin': Smin, 'Smax': Smax,'compliance':Compliance,'Distensibility':Distensibility}, ignore_index=True)
        
####### Regresion for correlation analysis >> Compliance

dataFile = dataFile.drop(29)  ## No data
# dataFile = dataFile.drop(5)  ## negative outlier
# dataFile = dataFile.drop(35) ## Positive outlier
df = df.drop(29)
# df = df.drop(5)  ## negative outlier
# df = df.drop(35) ## Positive outlier

       
Xc = dataFile['GT_Compliance'].values.reshape(-1,1)
yc = df['compliance'].values

ols = linear_model.LinearRegression()
model = ols.fit(Xc, yc)
response = model.predict(Xc)

r2c = model.score(Xc, yc)
rc=np.sqrt(r2c)

fig, ax = plt.subplots()

ax.plot(Xc, response, color='k', label='Regression model')
ax.scatter(Xc, yc, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('compliance', fontsize=14)
ax.set_xlabel('GT compliance', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2c+'$  R= %.2f$'%rc, fontsize=18)

fig.tight_layout()

##################### Regresion for correlation analysis >> SMin

Xmin = dataFile['Smin'].values.reshape(-1,1)
ymin = df['Smin'].values

ols = linear_model.LinearRegression()
model = ols.fit(Xmin, ymin)
response = model.predict(Xmin)
inter = model.intercept_
coef = model.coef_

equ = 'y='+str(round(coef[0],2))+'x+'+str(round(inter,2))

r2min = model.score(Xmin, ymin)
rmin=np.sqrt(r2min)

fig, ax = plt.subplots()

ax.plot(Xmin, response, color='k', label='Regression model')
ax.scatter(Xmin, ymin, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('4D flow min surface [$mm^2$]', fontsize=14)
ax.set_xlabel('Cine-MRI min surface [$mm^2$]', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
#ax.set_title('$R^2=   %.2f  $' % r2min+'$ R= %.2f$'% rmin, fontsize=18)
ax.set_title('$r=   %.2f  $' %  rmin, fontsize=18)

fig.tight_layout()
##################### Regresion for correlation analysis >> SMax

Xmax = dataFile['Smax'].values.reshape(-1,1)
ymax = df['Smax'].values

ols = linear_model.LinearRegression()
model = ols.fit(Xmax, ymax)
response = model.predict(Xmax)

r2max = model.score(Xmax, ymax)
inter = model.intercept_
coef = model.coef_

equ = 'y='+str(round(coef[0],2))+'x+'+str(round(inter,2))

rmax=np.sqrt(r2max)

fig, ax = plt.subplots()

ax.plot(Xmax, response, color='k', label='Regression model')
ax.scatter(Xmax, ymax, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('4D flow max surface [$mm^2$]', fontsize=14)
ax.set_xlabel('Cine-MRI max surface [$mm^2$]', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$r=   %.2f  $' % rmax, fontsize=18)

fig.tight_layout()


####################################################################################
#                            Statistical test min max                              #
####################################################################################

#### Normality Test
print('statistical test for Smax')
_,spXmax = stats.shapiro(Xmax) ###<=0.05  not normal distributed p=0.31
_,spYmax = stats.shapiro(ymax)  ###<=0.05  not normal distributed p=0.24

if spXmax <=0.05 or spYmax <=0.05: 
    print('One of the samples is not normally distributed')
    w, pmax = wilcoxon(Xmax[:,0],y=ymax) ### <=0.05 There is high difference  p=0.00029
    print('wilcoxon p-valueSmax: ',pmax)
    if pmax <=0.05:
        print('There is high difference for Smax')
    else:
        print('There is NO high difference for Smax')
        
else:
    print('Both samples are normally distributed')
    ###### 2-tailed 2 sample equal variance t-test
    _,pmax=stats.ttest_rel(Xmax[:,0], ymax)  #### p<0.05 there is difference -->> pvalue= 0.01509381
    print('t-test p-valueSmax: ',pmax)
    if pmax <=0.05:
        print('There is high difference for Smax')
    else:
        print('There is NO high difference for Smax')
    
   ###################  SMIN 
print('statistical test for Smin')
_,spXmin= stats.shapiro(Xmin) ###<=0.05  not normal distributed p=0.34
_,spYmin= stats.shapiro(ymin)  ###<=0.05  not normal distributed p=0.67

if spXmin <=0.05 or spYmin <=0.05: 
    print('One of the samples is not normally distributed')
    w, pmin = wilcoxon(Xmin[:,0],y=ymin) ### <=0.05 There is height difference  p=0.079
    print('wilcoxon p-valueSmin: ',pmin)
    if pmin <=0.05:
        print('There is high difference for Smin')
    else:
        print('There is NO high difference for Smin')
        
else:
    print('Both samples are normally distributed')
    ###### 2-tailed 2 sample equal variance t-test #### paired t-test 
    _,pmin=stats.ttest_rel(Xmin[:,0], ymin) #### p>=0.05 there is NO difference -->> pvalue= 0.69508334

    print('t-test p-valueSmin: ',pmin)
    if pmin <=0.05:
        print('There is high difference for Smin')
    else:
        print('There is NO high difference for Smin')


####################################################################################
#                Statistical test Compliance and distensibility                    #
####################################################################################


#### Normality Test
print('statistical test for Compliance')
_,spXc= stats.shapiro(Xc) ###<=0.05  not normal distributed p=0.31
_,spYc= stats.shapiro(yc)  ###<=0.05  not normal distributed p=0.24

if spXc <=0.05 or spYc <=0.05: 
    print('One of the samples is not normally distributed')
    w, pc = wilcoxon(Xc[:,0],y=yc) ### <=0.05 There is high difference  p=0.00029
    print('wilcoxon p-valueCompliance: ',pc)
    if pc <=0.05:
        print('There is high difference for Compliance')
    else:
        print('There is NO high difference for Compliance')
        
else:
    print('Both samples are normally distributed')
    ###### 2-tailed 2 sample equal variance t-test
    _,pc=stats.ttest_rel(Xc[:,0], yc)  #### p<0.05 there is difference -->> pvalue= 0.01509381
    print('t-test p-valueCompliance: ',pc)
    if pc <=0.05:
        print('There is high difference for Compliance')
    else:
        print('There is NO high difference for Compliance')
    

######### Bland-altman plot

f, ax = plt.subplots(1, figsize = (8,5))
ax.set_title('4D flow MRI Smin  vs cine-MRI Smin',fontsize=18,fontweight="bold")
sm.graphics.mean_diff_plot(np.squeeze(Xmin), ymin, ax=ax)
plt.show()



f, ax1 = plt.subplots(1, figsize = (8,5))
ax1.set_title('4D flow MRI Smax  vs cine-MRI Smax',fontsize=18,fontweight="bold")
sm.graphics.mean_diff_plot(np.squeeze(Xmax), ymax, ax = ax1)
plt.show()

f, ax2 = plt.subplots(1, figsize = (8,5))
ax2.set_title('GT compliance vs  computed compliance')
sm.graphics.mean_diff_plot(np.squeeze(Xc), yc, ax = ax2)
plt.show()


