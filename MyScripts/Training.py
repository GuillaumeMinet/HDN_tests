## imports ###

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import sys
sys.path.append('../')
from models.lvae import LadderVAE
from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from boilerplate import boilerplate
import lib.utils as utils
from lib import histNoiseModel
from lib.utils import plotProbabilityDistribution
import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
from skimage.transform import warp,AffineTransform

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

#### ALL PARAMETERS DEFINED HERE ####

display = True

# Data paths
basedir = Path(r"/group/jug/Anirban/Datasets/scilife_data")
data_path_signal = basedir / "training_data" / "avg_and_n2v"
data_path_obs = basedir /  "training_data" / "mltpl_snr_stacks"

# Noise Model
noiseModelsDir = basedir / "noiseModel"
GMMname = "GMM_Noise1_SigN2V_Clip-3.npz"
noise_model_params = np.load(str(noiseModelsDir / GMMname))
noiseModel = GaussianMixtureNoiseModel(params = noise_model_params, device = device)

# Data-related parameters

upsamp = 2 # upsampling factor (integer) NOTE: for now only upsamp by 2 possible in HDN

upsamp_beforeNN = False # if True, input upsampled w/ nearest neighbor before feeding into network. 
                        # if False, upsampled done in network.

target = "clean" # "noisy" or "clean"
augment = False
DataNoiseLvl = 1 # "all",list of int, or int 
normSignalToObs = True # put True if signal was normalized to observations when creating GMM
normGMM = True # put True if data was normalized to create the GMM
clip = -3 # False or clip value

# Model-specific
num_latents = 5
z_dim = 32
z_dims = [z_dim]*int(num_latents)
blocks_per_layer = 6
batchnorm = True
free_bits = 0
n_filters = 64

# Training prm
patch_size = 64
gaussian_noise_std = None
beta = 0.0001 # loss = recon_loss + beta * kl_loss
batch_size=64
virtual_batch = 8
lr=1e-4
max_epochs = 100
steps_per_epoch=200
test_batch_size=10

# Model name for saving
model_name_base = "Vim_mtlplSNR_Noise1_GMM1N2V_ConvTransposeBefore"
modelName = f"{model_name_base}_Upsamp{upsamp}_Clip{clip}_Lat{z_dim}x{num_latents}_{blocks_per_layer}Blocks_betaKL{beta}"
if not augment:
    modelName = modelName + "_NoAugment"

save_model_basedir = "./Trained_model_KL_weight_2/" 

print(f"Upsamp factor: {upsamp}")
print(f"Target: {target}")
print(f"Noise level: {DataNoiseLvl}")
print(f"GMM: {GMMname}")

print(f"Trained model will be saved at: {save_model_basedir}")
print(f"Model save name: {modelName}")


##### Load data ####
signal = []
observation = []
filters = ['tif','tiff']

files_signal = os.listdir(data_path_signal)
files_obs = os.listdir(data_path_obs)
files_signal.sort()
files_obs.sort()

for f in files_signal:
    if f.split('.')[-1] not in filters:
        print(f"removing {f} in signals because not in filters")
        files_signal.remove(f)

for f in files_obs:
    if f.split('.')[-1] not in filters:
        print(f"Removing {f} in observations because not in filters")
        files_obs.remove(f)

assert len(files_obs) == len(files_signal)
print(f"\nFound {len(files_signal)} files.\n")

if isinstance(DataNoiseLvl,list) or DataNoiseLvl == "all":
    mltplNoise = True
else:
    mltplNoise = False

for i in range (len(files_obs)):
    file_signal = files_signal [i]
    file_obs = files_obs [i]
    
    # load clean
    im_signal = imread(data_path_signal / file_signal)[0]
    
    # load noisy
    if DataNoiseLvl == "all":
        im_obs  = imread(data_path_obs / file_obs)[:5]
    elif isinstance(DataNoiseLvl,int) or isinstance(DataNoiseLvl,list):
        im_obs = imread(data_path_obs / file_obs)[DataNoiseLvl]

    # clip neg values
    if not isinstance(clip,bool):
        im_obs[im_obs<clip] = 0
        im_signal[im_signal<clip] = 0

    observation.append(im_obs)
    signal.append(im_signal)

    print(f"Signal {file_signal}:\tObservation {file_obs}:\t Shape: {im_obs.shape}")

signal = np.stack(signal)
observation = np.stack(observation)

if mltplNoise:
    nNoise = observation.shape[1]
    nrepeat = observation.shape[1]
    observation = np.reshape(observation,(observation.shape[0]*observation.shape[1],observation.shape[2],observation.shape[3]))    
    signal = np.repeat(signal,nrepeat,axis=0)

# square crop
if signal.shape[-1] != signal.shape[-2]:
    print("Cropping to square")
    a = min(signal.shape[-1],signal.shape[-2])
    signal = signal [...,0:a,0:a]
    observation = observation [...,0:a,0:a]

# opt. normalization signal to target 
if normSignalToObs:
    if mltplNoise:
        signal = (signal - np.mean(signal))/np.std(signal)
        for noise in range(nNoise):
            signal[noise::nNoise] = signal[noise::nNoise] * np.std(observation[noise::nNoise]) + np.mean(observation[noise::nNoise])
    else:
        signal = (signal - np.mean(signal))/np.std(signal)
        signal = signal * np.std(observation) + np.mean(observation)


print("Before normalization:")
print(f"Mean signal {np.mean(signal)}, std signal {np.std(signal)} ")
print(f"Mean observation {np.mean(observation)}, std observation {np.std(observation)} ")

# opt. normalization of data
if normGMM:
    signal = (signal - np.mean(observation)) / np.std(observation)
    observation = (observation - np.mean(observation)) / np.std(observation)

    print("After normalization:")
    print(f"Mean signal {np.mean(signal)}, std signal {np.std(signal)} ")
    print(f"Mean observation {np.mean(observation)}, std observation {np.std(observation)} ")

print(f"\n\nConcatenated arrays:\tSignal: {signal.shape}\tObservation: {observation.shape}")


### Train/val split and patch extraction ###

train_data = observation[:int(0.85*observation.shape[0])]
val_data= observation[int(0.85*observation.shape[0]):]
print("Shape of training images:", train_data.shape, "Shape of validation images:", val_data.shape)
if augment:
    train_data = utils.augment_data(train_data) ### Data augmentation disabled for fast training, but can be enabled


if target == "clean":
    train_data_gt = signal[:int(0.85*observation.shape[0])]
    val_data_gt = signal[int(0.85*observation.shape[0]):]
    print("Shape of GT training images:", train_data.shape, "Shape of validation images:", val_data.shape)
    if augment:
        train_data_gt = utils.augment_data(train_data_gt) ### Data augmentation disabled for fast training, but can be enabled


# Patches extraction
img_width = observation.shape[2]
img_height = observation.shape[1]
num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
print("Number of patches:", num_patches)
if target == "noisy":
    train_images = utils.extract_patches(train_data, patch_size, num_patches)
    val_images = utils.extract_patches(val_data, patch_size, num_patches)

elif target == "clean":
    train_images,train_images_gt = utils.extract_patches_supervised(train_data,train_data_gt, patch_size, num_patches)
    val_images,val_images_gt  = utils.extract_patches_supervised(val_data,val_data_gt, patch_size, num_patches)

# We limit validation patches to 1000 to speed up training but it is not necessary
val_images = val_images[:1000]
if target == "clean":
    val_images_gt = val_images_gt [:1000] 

print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)
img_shape = (train_images.shape[1], train_images.shape[2])

### Dataloaders ###

if target == "noisy":
    train_images_gt = train_images.copy()
    val_images_gt = val_images.copy()

train_loader, val_loader, data_mean, data_std = boilerplate._make_datamanager_supervised(train_images,train_images_gt,val_images,
                                                                                            val_images_gt,batch_size,test_batch_size,
                                                                                            upsamp=upsamp,upsamp_beforeNN = upsamp_beforeNN)


# To make sure that steps_per_epoch not bigger than len(train_loader)
steps_per_epoch=min(len(train_loader)-1,steps_per_epoch)
# print(steps_per_epoch)

# Display 1 patch example for each train and val loader
batch_idx, (x, y) = next(enumerate(train_loader))

if upsamp>1 and not upsamp_beforeNN:
    initial_upsamp = True
else: 
    initial_upsamp = False
    
initial_upsamp = False

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,n_filters=n_filters,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape,initial_upsamp=initial_upsamp).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=save_model_basedir,
                       train_loader=train_loader,val_loader=val_loader,virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=modelName,val_loss_patience=100,beta=beta)