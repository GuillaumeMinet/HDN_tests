# We import all our dependencies.
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import sys,os,shutil
import sys
sys.path.append(os.getcwd() + './')

from models.lvae import LadderVAE
from boilerplate import boilerplate
import lib.utils as utils
import training
import time
import glob
import zipfile
import urllib
from tifffile import imread, imsave
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

def crop_center(img,crop_size):

    if type(crop_size) == tuple:
        crop_x,crop_y = crop_size
    elif type(crop_size) == int:
        crop_x = crop_size
        crop_y = crop_size
    
    y,x = img.shape[-2::]
    startx = x//2-(crop_x//2)
    starty = y//2-(crop_y//2)        

    return img[...,starty:starty+crop_y,startx:startx+crop_x]



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

data_path = Path(r"E:\dl_monalisa\Data\Mito_live_2Dtimelapses\dump\2024-10-11")
crop_size = 400
# im_size = 1400
# start = im_size//2-crop_size//2
# stop = im_size//2+crop_size//2


# model
model = torch.load("MyScripts/Trained_model/model/Mito_live_2Dtimelapses_GMMmito_clip-5_5Lat_6Blocks_betaKL0.022_best_vae.net")
model.mode_pred=True
model.eval()

# saving
save_inp = False
suffix = "Mito0025"
overwrite = False
save_path = data_path / "denoised_HDN"
# save_path = data_path
if os.path.exists(save_path):
    if overwrite:
        print("Overwriting existing directory, pausing for 5sec in case you want to stop.")
        time.sleep(5)
        print("Proceeding")
        shutil.rmtree(save_path)
        os.makedirs(save_path)
else:
    os.makedirs(save_path)


# prediction

num_samples = 10
save_samples = False

# list_files = os.listdir(data_path)
list_files = ["c2_crop1.tif"]
print(list_files)

for k in range(len(list_files)):
    name_file = list_files[k]
    if name_file.split('.')[-1] not in ['tif','tiff']:
        print(f"Skipping {name_file}")
        continue

    print(f"Processing {name_file}")
    stack = imread(data_path / name_file)
    # stack = crop_center(stack,crop_size)
    print(stack.shape)
    stack[stack<-5]=-5
    # stack = (stack-(19.723637) ) / (36.286743)
    # stack = (stack-(26.494621) ) / (55.239285)
    # predict
    pred_stack = np.empty((stack.shape[0],stack.shape[1],stack.shape[2]))
    for i in range(stack.shape[0]):
        print(f"frame #{i}")
        frame = stack[i]
        img_mmse, samples = boilerplate.predict(frame,num_samples,model,None,device,False)
        pred_stack[i,...] = img_mmse
        if save_samples:
            imsave(save_path / (name_file.split('.')[0]+f"samples_{suffix}_frame{i}.tif"),samples)
    # saving
    full_save_path_inp = save_path / name_file
    full_save_path_pred = save_path / (name_file.split('.')[0]+f"pred_mmse_{suffix}.tif")
    if save_inp:
        imsave(full_save_path_inp,stack)
    imsave(full_save_path_pred,pred_stack)

    # if k>3:
    #     break