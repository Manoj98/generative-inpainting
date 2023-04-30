import sys
import torch
import torch.utils.data as data
from os import listdir
from utils.tools import default_loader, is_image_file, normalize
import os
import random

import torchvision.transforms as transforms
import deeplake
import numpy as np
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)

        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)

""" 
Custom DeeplakeDataset class for data downloaded from DeepLake
"""
class DeeplakeDataset(data.Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False, subset_size = 100):
        super(DeeplakeDataset, self).__init__()
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.data = deeplake.load(self.data_path)
        self.subset_size = subset_size

    def __getitem__(self, index):

        img = self.data.images[index].numpy()
        
        # Change to 256x256x3 from 256x256x1
        if img.shape[-1] == 1:
            img = np.stack([img.squeeze()]*3, axis=2) 
        
        # print("Image Numpy:", img)
        # Convert NumPy array to PIL image and then to RGB mode
        img = Image.fromarray(img).convert('RGB')
        # print("Image Type:", type(img))

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        # print("Image Before Normalize:", img)
        img = normalize(img)
        # print("Image After Normalize:", img)

        return img

    def __len__(self):
        return len(self.data)

    # def subset_data(self, subset_size = 100, seed = 0):
    #     """
    #     Randomly subset data from the original Dataset
    #     subset_size : Length of subset
    #     seed: Seed to reproduce the results
    #     """

    #     random.seed(seed)
    #     num_images = self.__len__()

    #     # Generate a random subset of indices
    #     subset_indices = random.sample(range(num_images), k=subset_size)
    #     subset_data = torch.utils.data.Subset(self.data, subset_indices)
    #     print("Size of the data subset: ", len(subset_data))
        
    #     return subset_data