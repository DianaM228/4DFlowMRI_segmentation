# -*- coding:utf-8 -*-
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import scipy.ndimage
import os
import nibabel as nib
from natsort import natsorted
from scipy.spatial import Delaunay
from itertools import combinations
from topologylayer.functional.persistence import SimplicialComplex
from skimage.measure import label



#
# Add Gaussian noise
#
class RandomNoise(object):
    """Add Gaussian noise to the image in a sample.

    Args:
        noise_strength: Desired noise strength.
    """

    def __init__(self, noise_strength):
        self.noise_strength = noise_strength  # std

    def __call__(self, augmentation_struct):
        noise_image, shape_image = augmentation_struct['noise_image'], augmentation_struct['shape_image']
        g_noise = np.random.normal(loc=0.0, scale=self.noise_strength, size=shape_image)
        g_noise = g_noise.astype('float32')
        noise_image = noise_image + g_noise
        augmentation_struct['noise_image'] = noise_image
        
        return augmentation_struct


#
#Data augmentation code for 3D models
#
class RandomFlipX(object):
    """Randomly flip the image in a sample.
    """

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Flip matrix
        sign = 1
        if np.random.random_sample() < 0.5:
            sign = -1
        flip = np.identity(4)
        flip[0, 0] = sign
        #
        flip_centre = np.matmul(flip,image_centre)
        #
        flip[0, 3] = (image_centre[0]-flip_centre[0])
        flip[1, 3] = (image_centre[1]-flip_centre[1])
        flip[2, 3] = (image_centre[2]-flip_centre[2])

        t_matrix = np.matmul(flip, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix
        # DEBUG
        #nib.save(image, os.path.join('out', 'Img_AfterRandomFlipX.nii.gz'))
        #nib.save(mask, os.path.join('out', 'Mask_AfterRandomFlipX.nii.gz'))
        # DEBUG
        return augmentation_struct


class RandomFlipY(object):
    """Randomly flip the image in a sample.
    """

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Flip matrix
        sign = 1
        if np.random.random_sample() < 0.5:
            sign = -1
        flip = np.identity(4)
        flip[1, 1] = sign
        #
        flip_centre = np.matmul(flip,image_centre)
        #
        flip[0, 3] = (image_centre[0]-flip_centre[0])
        flip[1, 3] = (image_centre[1]-flip_centre[1])
        flip[2, 3] = (image_centre[2]-flip_centre[2])

        t_matrix = np.matmul(flip, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomFlipZ(object):
    """Randomly flip the image in a sample.
    """

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Flip matrix
        sign = 1
        if np.random.random_sample() < 0.5:
            sign = -1
        flip = np.identity(4)
        flip[2, 2] = sign
        #
        flip_centre = np.matmul(flip,image_centre)
        #
        flip[0, 3] = (image_centre[0]-flip_centre[0])
        flip[1, 3] = (image_centre[1]-flip_centre[1])
        flip[2, 3] = (image_centre[2]-flip_centre[2])

        t_matrix = np.matmul(flip, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomRotationX(object):
    """Randomly rotate the image in a sample.

    Args:
        rotation_strength: Desired rotation strength.
    """

    def __init__(self, rotation_strength_1, rotation_strength_2):
        self.rotation_strength_1 = rotation_strength_1
        self.rotation_strength_2 = rotation_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Rotation matrix
        tetha = (self.rotation_strength_2 - self.rotation_strength_1) * np.random.random_sample() + self.rotation_strength_1
        rot = np.identity(4)
        rot[1, 1] =  np.cos(tetha)
        rot[1, 2] = -np.sin(tetha)
        rot[2, 1] =  np.sin(tetha)
        rot[2, 2] =  np.cos(tetha)
        #
        rot_centre = np.matmul(rot,image_centre)
        #
        rot[0, 3] = (image_centre[0]-rot_centre[0])
        rot[1, 3] = (image_centre[1]-rot_centre[1])
        rot[2, 3] = (image_centre[2]-rot_centre[2])

        t_matrix = np.matmul(rot, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomRotationY(object):
    """Randomly rotate the image in a sample.

    Args:
        rotation_strength: Desired rotation strength.
    """

    def __init__(self, rotation_strength_1, rotation_strength_2):
        self.rotation_strength_1 = rotation_strength_1
        self.rotation_strength_2 = rotation_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Rotation matrix
        tetha = (self.rotation_strength_2 - self.rotation_strength_1) * np.random.random_sample() + self.rotation_strength_1
        rot = np.identity(4)
        rot[0, 0] =  np.cos(tetha)
        rot[0, 2] =  np.sin(tetha)
        rot[2, 0] = -np.sin(tetha)
        rot[2, 2] =  np.cos(tetha)
        #
        rot_centre = np.matmul(rot,image_centre)
        #
        rot[0, 3] = (image_centre[0]-rot_centre[0])
        rot[1, 3] = (image_centre[1]-rot_centre[1])
        rot[2, 3] = (image_centre[2]-rot_centre[2])

        t_matrix = np.matmul(rot, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomRotationZ(object):
    """Randomly rotate the image in a sample.

    Args:
        rotation_strength: Desired rotation strength.
    """

    def __init__(self, rotation_strength_1, rotation_strength_2):
        self.rotation_strength_1 = rotation_strength_1
        self.rotation_strength_2 = rotation_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Rotation matrix
        tetha = (self.rotation_strength_2 - self.rotation_strength_1) * np.random.random_sample() + self.rotation_strength_1
        rot = np.identity(4)
        rot[0, 0] =  np.cos(tetha)
        rot[0, 1] = -np.sin(tetha)
        rot[1, 0] =  np.sin(tetha)
        rot[1, 1] =  np.cos(tetha)
        #
        rot_centre = np.matmul(rot,image_centre)
        #
        rot[0, 3] = (image_centre[0]-rot_centre[0])
        rot[1, 3] = (image_centre[1]-rot_centre[1])
        rot[2, 3] = (image_centre[2]-rot_centre[2])

        t_matrix = np.matmul(rot, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomShear(object):
    """Shear randomly the image in a sample.

    Args:
        shear_strength: Desired shear strength.
    """

    def __init__(self, shear_strength_1, shear_strength_2):
        self.shear_strength_1 = shear_strength_1
        self.shear_strength_2 = shear_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Shearing matrix
        sm = np.identity(4)
        sm[0, 1] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        sm[0, 2] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        sm[1, 0] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        sm[1, 2] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        sm[2, 0] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        sm[2, 1] = (self.shear_strength_2 - self.shear_strength_1) * np.random.random_sample() + self.shear_strength_1
        #
        sm_centre = np.matmul(sm, image_centre)
        #
        sm[0, 3] = (image_centre[0] - sm_centre[0])
        sm[1, 3] = (image_centre[1] - sm_centre[1])
        sm[2, 3] = (image_centre[2] - sm_centre[2])

        t_matrix = np.matmul(sm, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomTranslationX(object):
    """Translate randomly the image in a sample.

    Args:
        t_strength: Desired translation strength.
    """

    def __init__(self, t_strength_1, t_strength_2):
        self.t_strength_1 = t_strength_1
        self.t_strength_2 = t_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Translation matrix
        tm = np.identity(4)
        tm[0, 3] = (self.t_strength_2 - self.t_strength_1) * np.random.random_sample() + self.t_strength_1
        #
        tm_centre = np.matmul(tm, image_centre)
        #
        tm[0, 3] = (image_centre[0] - tm_centre[0])
        tm[1, 3] = (image_centre[1] - tm_centre[1])
        tm[2, 3] = (image_centre[2] - tm_centre[2])

        t_matrix = np.matmul(tm, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomTranslationY(object):
    """Translate randomly the image in a sample.

    Args:
        t_strength: Desired translation strength.
    """

    def __init__(self, t_strength_1, t_strength_2):
        self.t_strength_1 = t_strength_1
        self.t_strength_2 = t_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Translation matrix
        tm = np.identity(4)
        tm[1, 3] = (self.t_strength_2 - self.t_strength_1) * np.random.random_sample() + self.t_strength_1
        #
        tm_centre = np.matmul(tm, image_centre)
        #
        tm[0, 3] = (image_centre[0] - tm_centre[0])
        tm[1, 3] = (image_centre[1] - tm_centre[1])
        tm[2, 3] = (image_centre[2] - tm_centre[2])

        t_matrix = np.matmul(tm, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomTranslationZ(object):
    """Translate randomly the image in a sample.

    Args:
        t_strength: Desired translation strength.
    """

    def __init__(self, t_strength_1, t_strength_2):
        self.t_strength_1 = t_strength_1
        self.t_strength_2 = t_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Translation matrix
        tm = np.identity(4)
        tm[2, 3] = (self.t_strength_2 - self.t_strength_1) * np.random.random_sample() + self.t_strength_1
        #
        tm_centre = np.matmul(tm, image_centre)
        #
        tm[0, 3] = (image_centre[0] - tm_centre[0])
        tm[1, 3] = (image_centre[1] - tm_centre[1])
        tm[2, 3] = (image_centre[2] - tm_centre[2])

        t_matrix = np.matmul(tm, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class RandomUniformScaling(object):
    """Scale randomly the image in a sample.

    Args:
       s_strength: Desired scaling strength.
    """

    def __init__(self, s_strength_1, s_strength_2):
        self.s_strength_1 = s_strength_1
        self.s_strength_2 = s_strength_2

    def __call__(self, augmentation_struct):
        t_matrix, shape_image = augmentation_struct['transformation_matrix'], augmentation_struct['shape_image']
        # Centre of the image
        image_centre = 0.5 * np.array(shape_image) - 0.5
        image_centre = np.append(image_centre, 1)
        # Scaling matrix
        sc_m = np.identity(4)
        r = (self.s_strength_2 - self.s_strength_1) * np.random.random_sample() + self.s_strength_1
        sc_m[0, 0] = r
        sc_m[1, 1] = r
        sc_m[2, 2] = r
        #
        sc_m_centre = np.matmul(sc_m, image_centre)
        #
        sc_m[0, 3] = (image_centre[0] - sc_m_centre[0])
        sc_m[1, 3] = (image_centre[1] - sc_m_centre[1])
        sc_m[2, 3] = (image_centre[2] - sc_m_centre[2])

        t_matrix = np.matmul(sc_m, t_matrix)
        augmentation_struct['transformation_matrix'] = t_matrix

        return augmentation_struct


class ToTensor(object):
    """Convert nifty ndarrays to Tensors."""

    def __call__(self, sample):
        image, mask, affine = sample['image'], sample['mask'], sample['affine']
        shape_image = image.get_data().shape
        shape_mask = mask.get_data().shape
        np_img = image.get_data().reshape(1,shape_image[0],shape_image[1],shape_image[2])
        np_mask = mask.get_data().reshape(1,shape_mask[0],shape_mask[1],shape_mask[2])
        torch_image = torch.from_numpy(np_img).float()
        torch_image = torch_image.permute(0, 3, 2, 1)
        torch_mask = torch.from_numpy(np_mask).float()
        torch_mask = torch_mask.permute(0, 3, 2, 1)
        return {'image': torch_image,
                'mask':  torch_mask,
                'affine': torch.from_numpy(affine).float()}


class Normalize(object):
    """Normalize the Tensors – mean-std nomalization."""

    def __init__(self, standard_deviation, mean):
        self.standard_deviation = standard_deviation
        self.mean = mean

    def __call__(self, sample):
        image, mask, affine = sample['image'], sample['mask'], sample['affine']
        #current_tensor = torch.div(current_tensor, self.max_value)
        image = torch.add(image, -self.mean)
        image = torch.div(image, self.standard_deviation)
        return {'image': image,
                'mask': mask,
                'affine': affine}


class NormToTensor(object):
    """Normalize + convert nifty ndarrays to Tensors."""

    def __init__(self, div_value, num_organs):
        self.div_value = div_value
        self.num_organs = num_organs

    def __call__(self, sample):
        image, mask, affine,name = sample['image'], sample['mask'], sample['affine'], sample['name']
        np_img = image.get_data()
        np_mask = mask.get_data()
        shape_np_img = np_img.shape
        shape_np_mask = np_mask.shape
        np_img = np.divide(np_img, self.div_value, dtype=np.float32)        
        np_img = np_img.reshape(1,shape_np_img[0],shape_np_img[1],shape_np_img[2]) 
        torch_image = torch.from_numpy(np_img).float() 
        torch_image = torch_image.permute(0, 3, 2, 1)         
        num_classes = self.num_organs + 1
        new_shape_np_mask = shape_np_mask + (num_classes,)
        new_np_mask = np.zeros(shape=new_shape_np_mask, dtype=np.uint8)
        for i in range(0, num_classes):
            new_np_mask[:, :, :, i] = np.where(np_mask[:, :, :] == i, 1, 0)
        torch_mask = torch.from_numpy(new_np_mask).float()
        torch_mask = torch_mask.permute(3, 2, 1, 0)
        return {'image': torch_image,
                'mask':  torch_mask,
                'affine': torch.from_numpy(affine).float(),
                'name':name}


class Resampling(object):

    def __call__(self, sample, resampling_size):
        image, mask, affine,name = sample['image'], sample['mask'], sample['affine'], sample['affine']

        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)
        image = image.permute(0,1,4,3,2)
        mask = mask.permute(0,1,4,3,2)
        resample_image = F.interpolate(image, resampling_size, mode='trilinear', align_corners=True)
        resample_mask = F.interpolate(mask, resampling_size, mode='nearest')

        image = torch.squeeze(resample_image, 1)
        mask = torch.squeeze(resample_mask, 1)

        return {'image': image, 'mask': mask, 'affine': affine,'name':name}

def NumInString0(string):
    List = [ i for i,x in enumerate(string) if x.isdigit() == bool('True')]    
    name = string[List[0]:List[-1]+1]
    return name, List[0]

def NumInString(string):
    List=[]
    counter = -1
    for i,x in enumerate(string):
        if x.isdigit() == bool('True'):
            List.append(i)
        elif int(x.isdigit()) == 0:
            counter+=1
            if counter > 0: ### If starts agains an string 
                break  
    name = string[List[0]:List[-1]+1]
    return name, List[0]




class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(
        self, optimizer, patience=100, min_lr=1e-6, factor=0.1):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
        
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=300, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif (self.best_loss - val_loss) <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True




class AortaDataset(Dataset):
    """CMRI dataset for aorta segmentation on 3D"""

    def __init__(self, database_path, list_ids,path_all, norm_value, transform=None, num_organs=1):
        """
        Args:
            database_path (string): Directory where is the image database
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.database_path = database_path
        self.list_ids = list_ids
        self.norm_value = norm_value
        self.transform = transform
        self.num_organs = num_organs
        
        all_images = path_all
        self.img_files = []
        self.mask_files = []
        self.Images_Names = []
        for i in self.list_ids: ### folder per patient     
            ImagesPatient = natsorted(glob.glob(os.path.join(all_images[i],'*'))) 
            self.img_files = self.img_files +  ImagesPatient 
            basname = os.path.splitext(os.path.basename(ImagesPatient[0]))[0]
            keyname,namestr = NumInString(basname)
            ##### retrieve Masks
            pathMasks = os.path.join(os.path.dirname(os.path.dirname(all_images[i])),'Masks',os.path.basename(all_images[i]))
            MasksPatient = natsorted(glob.glob(os.path.join(pathMasks,'*')))
            self.mask_files = self.mask_files + MasksPatient
        #### To keep the name 
        for n in self.mask_files:
            self.Images_Names.append(os.path.basename(n))
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):  
        print('index: ',index)
        
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        mask_path = os.path.normpath(mask_path)
        Image_Name = self.Images_Names[index]
        print('image',Image_Name )
        

        # load images - image + mask
        image = nib.load(img_path)
        mask = nib.load(mask_path)
        if self.transform:
            augmentation_struct = {'transformation_matrix': np.identity(4, dtype=np.float32),
                                   'noise_image': np.zeros(image.shape, dtype=np.float32),
                                   'shape_image': image.shape}
           
            augmentation_struct = self.transform(augmentation_struct) 
            
            # apply compose transformations to image and mask            
            image_data = scipy.ndimage.affine_transform(image.get_data(),
                                                        np.linalg.inv(augmentation_struct['transformation_matrix']))
            mask_data = scipy.ndimage.affine_transform(mask.get_data(),
                                                       np.linalg.inv(augmentation_struct['transformation_matrix']),
                                                       order=0)
            
            image = nib.Nifti1Image(image_data, affine=image.affine)
            mask = nib.Nifti1Image(mask_data, affine=mask.affine)
            
        sample = {'image': image, 'mask': mask, 'affine': image.affine,'name':Image_Name}
        # Normalise the sample
        norm = NormToTensor(self.norm_value, self.num_organs)
        sample = norm(sample)
        # Return the sample
        return sample

############# Functions to compute topoloss complex

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
    
    
    