# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:01:55 2022

@author: Diana_MARIN


-i E:\4D_DeepLearning\Results\Final\Postprocessing25Frames\PostprocessData_Test_Results4Dwith3D_lr0.01_5lab_500e-Copy_R1  ### if I want to compute the results for chapter 5
-r 2 
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



############################### Bland-Altman
dataFile = dataFile.drop(29)  ## No data
df = df.drop(29)

Xmin = dataFile['Smin'].values.reshape(-1,1)
ymin = df['Smin'].values


df2 = pd.DataFrame({
    'Cine-MRI': dataFile['Smin'].values
        
        ,
    '4D-FLow': df['Smin'].values
})




means = df2.mean(axis=1)
diffs = df2.diff(axis=1).iloc[:, -1]
percent_diffs = diffs / means * 100
bias = np.mean(percent_diffs)
sd = np.std(percent_diffs, ddof=1)
upper_loa = bias + 1.96 * sd
lower_loa = bias - 1.96 * sd


# Sample size
n = df2.shape[0]
# Variance
var = sd**2
# Standard error of the bias
se_bias = np.sqrt(var / n)
# Standard error of the limits of agreement
se_loas = np.sqrt(3 * var / n)
# Endpoints of the range that contains 95% of the Student’s t distribution
t_interval = stats.t.interval(alpha=0.95, df=n - 1)
# Confidence intervals
ci_bias = bias + np.array(t_interval) * se_bias
ci_upperloa = upper_loa + np.array(t_interval) * se_loas
ci_lowerloa = lower_loa + np.array(t_interval) * se_loas


ax = plt.axes()
ax.set(
    title='4D flow MRI Smin  vs cine-MRI Smin',
    xlabel='Means', ylabel=r'Percentage Differences (%)'
)
# Scatter plot
ax.scatter(means, percent_diffs, c='k', s=20, alpha=0.6, marker='o')
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')
# Get axis limits
left, right = plt.xlim()
bottom, top = plt.ylim()
# Increase the y-axis limits to create space for the confidence intervals
max_y = max(abs(ci_lowerloa[0]), abs(ci_upperloa[1]), abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left - domain * 0.05, left + domain * 1.13)
# Add the annotations
ax.annotate('+1.96×SD', (right, upper_loa), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{upper_loa:+4.2f}%', (right, upper_loa), xytext=(0, -25), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{bias:+4.2f}%', (right, bias), xytext=(0, -25), textcoords='offset pixels')
ax.annotate('-1.96×SD', (right, lower_loa), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{lower_loa:+4.2f}%', (right, lower_loa), xytext=(0, -25), textcoords='offset pixels')
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--')
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--')
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--')
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--')
# Show plot
plt.show()

plt.savefig(r'C:\Users\Diana_MARIN\Documents\Diana\Publications\MRI\Figures\Bland-Altman_Smin_percentage.eps', format='eps')




####################################################  SMAX  ###########################################################

df2 = pd.DataFrame({
    'Cine-MRI': dataFile['Smax'].values
        
        ,
    '4D-FLow': df['Smax'].values
})




means = df2.mean(axis=1)
diffs = df2.diff(axis=1).iloc[:, -1]
percent_diffs = diffs / means * 100
bias = np.mean(percent_diffs)
sd = np.std(percent_diffs, ddof=1)
upper_loa = bias + 1.96 * sd
lower_loa = bias - 1.96 * sd


# Sample size
n = df2.shape[0]
# Variance
var = sd**2
# Standard error of the bias
se_bias = np.sqrt(var / n)
# Standard error of the limits of agreement
se_loas = np.sqrt(3 * var / n)
# Endpoints of the range that contains 95% of the Student’s t distribution
t_interval = stats.t.interval(alpha=0.95, df=n - 1)
# Confidence intervals
ci_bias = bias + np.array(t_interval) * se_bias
ci_upperloa = upper_loa + np.array(t_interval) * se_loas
ci_lowerloa = lower_loa + np.array(t_interval) * se_loas


ax = plt.axes()
ax.set(
    title='4D flow MRI Smin  vs cine-MRI Smax',
    xlabel='Means', ylabel=r'Percentage Differences (%)'
)
# Scatter plot
ax.scatter(means, percent_diffs, c='k', s=20, alpha=0.6, marker='o')
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')
# Get axis limits
left, right = plt.xlim()
bottom, top = plt.ylim()
# Increase the y-axis limits to create space for the confidence intervals
max_y = max(abs(ci_lowerloa[0]), abs(ci_upperloa[1]), abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left - domain * 0.05, left + domain * 1.13)
# Add the annotations
ax.annotate('+1.96×SD', (right, upper_loa), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{upper_loa:+4.2f}%', (right, upper_loa), xytext=(0, -25), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{bias:+4.2f}%', (right, bias), xytext=(0, -25), textcoords='offset pixels')
ax.annotate('-1.96×SD', (right, lower_loa), xytext=(0, 7), textcoords='offset pixels')
ax.annotate(fr'{lower_loa:+4.2f}%', (right, lower_loa), xytext=(0, -25), textcoords='offset pixels')
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--')
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--')
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--')
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--')
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--')
# Show plot
plt.show()
plt.savefig(r'C:\Users\Diana_MARIN\Documents\Diana\Publications\MRI\Figures\Bland-Altman_Smax_percentage.eps', format='eps')

















