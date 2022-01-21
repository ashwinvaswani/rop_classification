import torch
from skimage.io import imread
from torch.utils import data
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import rgb2gray, clahe_equalized, adjust_gamma

scale =300

def scaleRadius(img,scale):
  x=img[int(img.shape[0]/2),:,:].sum(1)
  r =(x>x.mean()/10).sum()/2
  s= scale*1.0/r
  return cv2.resize(img,(0,0),fx=s,fy=s)

def pre_proc(data):
    assert(len(data.shape)==3)
    assert (data.shape[0]==3)  #Use the original images
    # grayscale conversion
    train_imgs = rgb2gray(data)
    # train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)    
    train_imgs = adjust_gamma(np.expand_dims(train_imgs, axis=0), 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            x_img = cv2.imread(self.inputs[index])
            x_img = pre_proc(np.moveaxis(x_img, -1, 0))
            x_img = np.expand_dims(x_img, axis=2)

            y_img = cv2.imread(self.targets[index], cv2.IMREAD_GRAYSCALE)
            y_img[y_img<=25]=0
            y_img[y_img>25]=1
            
            x,y = self.transform[1](x_img,y_img)
    
        return torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype), self.inputs[index]

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(inp), imread(tar)
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar