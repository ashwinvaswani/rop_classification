"""
pre-trained ResNet18
training on split 3 (James)
oversampling of Pre-Plus and Plus
originally trained using MONAI version: 0.3.0rc4
"""
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
)
from monai.utils import set_determinism
from dataset import ROPDataset
from utils import train, test

print("Environment onfiguration is:")
print_config()

dict_label2idx = {'No':0,'Pre-Plus':1,'Plus':2}

df_train = pd.read_csv('train.csv', usecols=['imageName','goldenstandardreading@ohsu.edu'])
df_train = df_train.replace({'goldenstandardreading@ohsu.edu': dict_label2idx})
df_train = df_train.rename(columns={"imageName": "imageName", "goldenstandardreading@ohsu.edu": "label"})

class_count = np.unique(df_train['label'], return_counts=True)[1]

df_test = pd.read_csv('test.csv', usecols=['imageName','goldenstandardreading@ohsu.edu'])
df_test = df_test.replace({'goldenstandardreading@ohsu.edu': dict_label2idx})
df_test = df_test.rename(columns={"imageName": "imageName", "goldenstandardreading@ohsu.edu": "label"})

class_count_test = np.unique(df_test['label'], return_counts=True)[1]

paths_old = df_train["imageName"].tolist()  # load the "old" paths as defined in James' spreadsheet
new_datadir = "./../segmented"

image_files_list = [os.path.join(new_datadir, i) for i in paths_old] # combine the image name with the new datapath 

val_paths_old = df_test["imageName"].tolist()  # load the "old" paths as defined in James' spreadsheet
val_image_files_list = [os.path.join(new_datadir, i) for i in val_paths_old] # combine the image name with the new datapath 

list_all_images = []
image_class = df_train['label'].tolist()
image_class_list = []
val_image_class = df_test['label'].tolist()
val_image_class_list = []

for elem in os.listdir('./../segmented/'):
    list_all_images.append(new_datadir + '/' + elem)

val_image_files_list_updated = []
image_files_list_updated = []
for i, elem in enumerate(val_image_files_list):
    if elem in list_all_images:
        val_image_files_list_updated.append(elem)
        val_image_class_list.append(val_image_class[i])

for i, elem in enumerate(image_files_list):
    if elem in list_all_images:
        image_files_list_updated.append(elem)
        image_class_list.append(image_class[i])


"""
transforms 
"""
train_transforms = Compose(
    [
        LoadPNG(image_only=True),
        AddChannel(),
        ScaleIntensity(), 
        RandRotate(range_x=15, prob=0.1, keep_size=True), # low probability for rotation 
        RandFlip(spatial_axis=0, prob=0.5),# left right flip 
        RandFlip(spatial_axis=1, prob=0.5), # horizontal flip
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5), 
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

val_transforms = Compose(
    [
        LoadPNG(image_only=True),
        # Resize((480,640)),
        AddChannel(), 
        ScaleIntensity(),
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
)

# Dataloader
# No: 80%, Pre-Plus: 14%, Plus: 6%
weights_class = [1.25, 7.14, 16.7] # manually define the weights for each class 
weights_images = [weights_class[image_class_item] for image_class_item in image_class_list]

# Weighted sampler for unbalanced dataset
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_images, len(image_class_list), replacement=True)   
batch_size = 8
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last = True)

train_ds = ROPDataset(image_files_list_updated, image_class_list, train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler = batch_sampler, num_workers=0)

val_ds = ROPDataset(val_image_files_list_updated, val_image_class_list, val_transforms)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, num_workers=0)

# Model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)
device = torch.device("cuda:1")
model.load_state_dict(torch.load('./../models/5th_weighting_strategy.pth',map_location='cuda:1'))
model.to(device)


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
val_interval = 1 # evaluate accuracy after each epoch

model_dir = "./../models/"
epoch_num = 1

# Training
best_metric = 0
best_metric_epoch = 0
epoch_loss_values = list()
metric_values = list()

for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    train(epoch, model, loss_function, optimizer, train_loader, val_loader, device, best_metric, best_metric_epoch, model_dir)
    if (epoch + 1) % val_interval == 0:
        print("Validating...")
        best_metric, best_metric_epoch = test(epoch, model, loss_function, optimizer, train_loader, val_loader, device, best_metric, best_metric_epoch, model_dir)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")