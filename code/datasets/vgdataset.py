import torch
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
import os
#category_info = '../data/VG/vg_category_1000.json'

# =============================================================================
# class VGDataset(data.Dataset):
#     def __init__(self, img_dir, img_list,fMRI_dir,fMRI_list, input_transform, label_path,num_class):
#         with open(img_list, 'r') as f:
#             self.img_names = f.readlines()
#         with open(fMRI_list,'r') as f:
#             self.fMRI_names = f.readlines()
#         with open(label_path, 'r') as f:
#             self.labels = f.readlines()
#         
#         self.input_transform = input_transform
#         self.img_dir = img_dir
#         self.fMRI_dir = fMRI_dir
#         self.num_classes= num_class
#     
#     def __getitem__(self, index):
#         name1 = self.img_names[index][:-1]
#         input_image = Image.open(os.path.join(self.img_dir, name1)).convert('RGB')
#         name2 = self.labels[index][:-1]
#         #b, g, r = input.split()
#         #input = Image.merge("RGB", (r, g, b))
#         if self.input_transform:
#            input_image = self.input_transform(input_image)
#         label = np.load(os.path.join(self.img_dir, name2))
#         name3 = self.fMRI_names[index][:-1]
#         #print('######name3:',name3)
#         input_fMRI = np.load(os.path.join(self.fMRI_dir,name3))
#         
#         return input_image,input_fMRI, label
# 
#     def __len__(self):
#         return len(self.img_names)
# 
# =============================================================================
class VGDataset(data.Dataset):
    def __init__(self, fMRI_dir,fMRI_list, input_transform, label_path,num_class,fMRI_length):

        with open(fMRI_list,'r') as f:
            self.fMRI_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels = f.readlines()
        
        self.input_transform = input_transform
        self.fMRI_dir = fMRI_dir
        self.num_classes= num_class
        self.fMRI_length = fMRI_length
    def __getitem__(self, index):
        name2 = self.labels[index][:-1]
        #b, g, r = input.split()
        #input = Image.merge("RGB", (r, g, b))

        label = np.load(os.path.join(self.fMRI_dir, name2))
        name3 = self.fMRI_names[index][:-1]
        #print('######name3:',name3)
        input_fMRI = np.load(os.path.join(self.fMRI_dir,name3))[0:self.fMRI_length,:]
        #print("==================",input_fMRI.shape)
        return input_fMRI, label

    def __len__(self):
        return len(self.fMRI_names)
