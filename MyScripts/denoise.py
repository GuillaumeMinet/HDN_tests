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




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# data
data_path = Path(r"E:\dl_monalisa\Data\Vim_live_timelapse_Monalisa1_35nm\dump\recon\dyn_time2")
crop_size = 260
im_size = 1060
start = im_size//2-crop_size//2
stop = im_size//2+crop_size//2


# model
model = torch.load("MyScripts/Trained_model/model/Vim_fixed_InpSingle_best_vae.net")
model.mode_pred=True
model.eval()

# saving
save_inp = True
overwrite = True
save_path = Path(r"E:\dl_monalisa\Data\Vim_live_timelapse_Monalisa1_35nm\dump\HDN_singleInpModel\dyn_time2")
if os.path.exists(save_path):
    if not overwrite:
        raise Exception("Save dir already exists")
    else:
        print("Overwriting existing directory, pausing for 3sec in case you want to stop.")
        time.sleep(3)
        print("Proceeding")
        shutil.rmtree(save_path)
        os.makedirs(save_path)
else:
    os.makedirs(save_path)


# prediction

num_samples = 20
list_files = os.listdir(data_path)
for k in range(len(list_files)):
    name_file = list_files[k]
    if name_file.split('.')[-1] not in ['tif','tiff']:
        print(f"Skipping {name_file}")
        continue

    print(f"Processing {name_file}")
    stack = imread(data_path / name_file)[:,start:stop,start:stop]
    print(stack.shape)
    stack[stack<0]=0

    # predict
    pred_stack = pred_timelapse = np.empty((stack.shape[0],stack.shape[1],stack.shape[2]))
    for i in range(stack.shape[0]):
        print(f"frame #{i}")
        frame = stack[i]
        img_mmse, _ = boilerplate.predict(frame,num_samples,model,None,device,False)
        pred_stack[i,...] = img_mmse
    # saving
    full_save_path_inp = save_path / name_file
    full_save_path_pred = save_path / (name_file.split('.')[0]+"pred_mmse.tif")
    if save_inp:
        imsave(full_save_path_inp,stack)
    imsave(full_save_path_pred,pred_stack)

    if k>3:
        break