# Imports
import pathlib
import numpy as np
import torch

from model import MF_U_Net

import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.models
from torch import nn

from torchvision.transforms import Lambda, Normalize 

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from monai.networks.nets import densenet121
from monai.transforms import (
    AddChannel,
    Compose,
    LoadPNG,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity, Transpose, 
    LoadImage,
    ToTensor,
    Resize,
    AsChannelFirst,
)
from monai.utils import set_determinism
from dataset import SegmentationDataSet
import cv2
from sklearn.externals._pilutil import bytescale
from utils import diceCoeff
from transformations import re_normalize
from utils import postprocess, dice_coef_multilabel

from transformations import Compose, Resize, DenseTarget
from transformations import MoveAxis, Normalize01

# training transformations and augmentations
transforms = Compose([
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])

df_test = pd.read_csv('db_test.csv')

val_image_org_list_updated = df_test['image_list'].tolist()
val_image_files_list_updated = df_test['mask_list'].tolist()

device = torch.device("cuda:0")

val_ds = SegmentationDataSet(val_image_org_list_updated, val_image_files_list_updated, [transforms, transforms])
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=0)

model = MF_U_Net().to(device)
model_name = './models/unet_model_155lradamrandomcropw1loss_newmodel_180000.pth'
model.load_state_dict(torch.load(model_name))
model.eval()

ctr = 0
dice_sum = 0
for batch_data in val_loader:

    try:
        inputs, labels, filenames = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]

        out = model(inputs)

        for i in range(inputs.shape[0]):
            ctr += 1
            f, axarr = plt.subplots(1,3, constrained_layout=True)
            
            inputs_rgb = cv2.cvtColor(inputs[i].permute(1,2,0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)

            axarr[0].imshow(inputs_rgb)
            axarr[1].imshow(labels[i].detach().cpu(), cmap='gray')
            post_image = postprocess(out[i].permute(1,2,0).detach().cpu())
            axarr[2].imshow(post_image, cmap='gray')
            plt.savefig('outputs/' + filenames[i].split('/')[-1][:-3] + 'png')
            print("Dice Coeff for file ", filenames[i].split('/')[-1][:-3], " is: ",diceCoeff(torch.Tensor(np.expand_dims(out[i].permute(1,2,0)[:,:,1].detach().cpu(), axis=0)).to(device), labels[i].unsqueeze(0)).item())
            dice_sum += diceCoeff(torch.Tensor(np.expand_dims(out[i].permute(1,2,0)[:,:,1].detach().cpu(), axis=0)).to(device), labels[i].unsqueeze(0)).item()
            print()
    except:
        pass

print("Final diceCoeff on data is: ", dice_sum / len(val_loader))
print("Outputs are stored in the outputs folder!")