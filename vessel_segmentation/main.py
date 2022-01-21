import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.models

from torchvision.transforms import Lambda, Normalize 
from torchvision import transforms

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
from utils import Trainer
from model import UNet, MF_U_Net

df_train = pd.read_csv('db_train.csv')
df_test = pd.read_csv('db_test.csv')

image_org_list_updated = df_train['image_list'].tolist()
image_files_list_updated = df_train['mask_list'].tolist()
val_image_org_list_updated = df_test['image_list'].tolist()
val_image_files_list_updated = df_test['mask_list'].tolist()

"""
transforms 
"""

from transformations import Compose, Resize, DenseTarget, AlbuSeg2d
from transformations import MoveAxis, Normalize01
import albumentations
# training transformations and augmentations
transforms = Compose([
    Normalize01(),
    AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    AlbuSeg2d(albu=albumentations.VerticalFlip(p=0.5)),
    AlbuSeg2d(albu=albumentations.Rotate(limit=(0, 360))),
    AlbuSeg2d(albu=albumentations.RandomCrop(64, 64)),
    DenseTarget(),
    MoveAxis(),

])

val_transforms = Compose([
    Normalize01(),
    DenseTarget(),
    MoveAxis(),
])

device = torch.device("cuda:0")

train_ds = SegmentationDataSet(image_org_list_updated, image_files_list_updated, [transforms, transforms])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, num_workers=0, shuffle=True)

val_ds = SegmentationDataSet(val_image_org_list_updated, val_image_files_list_updated, [val_transforms, transforms])
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, num_workers=0)

model = MF_U_Net().to(device)
model_name = './models/unet_model_155lradamrandomcropw1loss_newmodel_120000.pth'
model.load_state_dict(torch.load(model_name))

# criterion
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2,0.8]).to(device))
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_loader,
                  validation_DataLoader=val_loader,
                  lr_scheduler=None,
                  epochs=180000,
                  epoch=0,
                  notebook=False)
# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()