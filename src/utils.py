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

def train(epoch, model, criterion, optimizer, train_loader, val_loader, device, best_metric, best_metric_epoch, model_dir):
    total_correct = 0
    metric_values = list()
    epoch_loss_values = list()
    model.train()
    epoch_loss = 0
    step = 0
    num_elements = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        num_elements += inputs.shape[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_correct += torch.eq(outputs.argmax(dim=1), labels).sum().item()

   
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    acc_metric = float(total_correct) / num_elements
    metric_values.append(acc_metric)

    print(
        f"current epoch: {epoch + 1} current accuracy: {float(acc_metric):.4f}"
        f" best validation accuracy: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" average loss: {epoch_loss:.4f}"
    )

    torch.save(model.state_dict(), os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth"))
    print("saved model after epoch ", epoch+1)



def test(epoch, model, criterion, optimizer, train_loader, val_loader, device, best_metric, best_metric_epoch, model_dir):
    metric_values = list()
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        val_filenames_all = []
        for val_data in val_loader:
            val_images, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )
                
            val_filenames = val_data[2]
            val_filenames_all.extend(val_filenames)

            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)
            # auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
            # metric_values.append(auc_metric)

        df_out = pd.DataFrame({'filename':val_filenames_all, 'label': y_pred.argmax(dim=1).cpu().numpy()})
        df_out.to_csv('out.csv',index=False)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        metric_values.append(acc_metric)
        print(
            f"current epoch: {epoch + 1} current accuracy: {acc_metric:.4f}"
            f" best accuracy: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
        if acc_metric > best_metric:
            best_metric = acc_metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
            print("saved new best metric model")
    
    return best_metric, best_metric_epoch