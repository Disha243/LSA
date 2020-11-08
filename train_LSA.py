import os
import pickle
import json
import random
import logging
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchio
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.distributions as dist
import math
import import_ipynb
from functools import reduce
from operator import mul
from typing import List
from typing import Optional
from torchsummary import summary
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


num_workers = 0
mood_region = 'brain'
useCuda = True
do_val = True
gpuID = "0"
seed = 1701
trainID = "Anomaly_ModenaVAE2D"
batch_size = 2
num_epochs = 250
lr = 1e-4
patch_size = (256, 256, 1)  # Set it to None if not desired
patchQ_len = 512
patches_per_volume = 256
amp_level = 'O1'
log_freq = 10
preload_h5 = True
indicesOfImgVols = [1, 2]

IsVAE = True
input_shape = (256, 256, 256)
code_length = 64
cpd_channels = 100
n_starting_features = 32  # 32
n_channels = 1
out_sigmoid = True
AutoregLoss_weight = 1  # weight of the autoregression loss.
ce_factor = 0.5
beta = 0.01
vae_loss_ema = 1
theta = 1
use_geco = False


checkpoint2load = None
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

imgsh5_train = r"/nfs1/sagrawal/Data/Project_Anomaly/MOOD_train.h5"
imgsh5_val = r"/nfs1/sagrawal/Data/Project_Anomaly/MOOD_toytest_brain.h5"
log_path = r'/scratch/sagrawal/'
save_path = r'/scratch/sagrawal/'


if __name__ == "__main__":
    wandb.init()
    config = wandb.config

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and useCuda else "cpu")
    #device = torch.cuda.set_device(0)

    trainset = MoodTrainSet(indices=indicesOfImgVols, region=mood_region,
                            data_path=imgsh5_train, lazypatch=True if patch_size else False, preload=preload_h5)
    valset = MoodValSet(
        data_path=imgsh5_val, lazypatch=True if patch_size else False, preload=preload_h5)

    if patch_size:
        input_shape = tuple(x for x in patch_size if x != 1)
        trainset = torchio.data.Queue(
            subjects_dataset=trainset,
            max_length=patchQ_len,
            samples_per_volume=patches_per_volume,
            sampler=torchio.data.UniformSampler(patch_size=patch_size),
            # num_workers = num_workers
        )
        valset = torchio.data.Queue(
            subjects_dataset=valset,
            max_length=patchQ_len,
            samples_per_volume=patches_per_volume,
            sampler=torchio.data.UniformSampler(patch_size=patch_size),
            # num_workers = num_workers
        )

    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = None if (valset is None) or (not do_val) else DataLoader(
        dataset=valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = LSAMOOD(input_shape=(n_channels,)+input_shape, code_length=code_length, cpd_channels=cpd_channels,
                    n_starting_features=n_starting_features, n_channels=n_channels, out_sigmoid=out_sigmoid, vae_mode=IsVAE)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    checkpoint2load = None

    if checkpoint2load:
        chk = torch.load(checkpoint2load)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        amp.load_state_dict(chk['amp'])
        start_epoch = chk['epoch'] + 1
        best_loss = chk['loss']
    else:
        start_epoch = 0
        best_loss = float('inf')

    scaler = GradScaler()

    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        runningLoss = 0.0
        runningLossCounter = 0.0
        train_loss = 0.0
        print('Epoch ' + str(epoch) + ': Train')
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader):
                img = data['img']['data'].squeeze(-1)  # * 2 - 1 #For 2D cases
                images = Variable(img).to(device)
                optimizer.zero_grad()
                with autocast():
                    # VAE Part
                    loss_vae = 0
                    if ce_factor < 1:
                        x_r, _, z_dist = model(images)

                        kl_loss = 0
                        if beta > 0:
                            if IsVAE:
                                kl_loss = kl_loss_fn(
                                    z_dist, sumdim=(1,)) * beta
                            else:
                                sys.exit("KLD Not gonna work")
                                kl_loss = kl_loss_fn(
                                    z_dist, sumdim=(1, 2)) * beta
                        if model.d == 3:
                            rec_loss_vae = rec_loss_fn(
                                x_r, images, sumdim=(1, 2, 3, 4))
                        else:
                            rec_loss_vae = rec_loss_fn(
                                x_r, images, sumdim=(1, 2, 3))
                        loss_vae = kl_loss + rec_loss_vae * theta

                    # CE Part
                    loss_ce = 0
                    if ce_factor > 0:
                        ce_tensor = get_square_mask(
                            (images.size(0), n_channels)+input_shape,
                            square_size=(0, np.max(input_shape) // 2),
                            noise_val=(torch.min(img).item(),
                                       torch.max(img).item()),
                            n_squares=(0, 3),
                        )
                        ce_tensor = torch.from_numpy(ce_tensor).float()
                        inpt_noisy = torch.where(
                            ce_tensor != 0, ce_tensor, img)

                        inpt_noisy = inpt_noisy.to(device)
                        x_rec_ce, _, _ = model(inpt_noisy)
                        if model.d == 3:
                            rec_loss_ce = rec_loss_fn(
                                x_rec_ce, images, sumdim=(1, 2, 3, 4))
                        else:
                            rec_loss_ce = rec_loss_fn(
                                x_rec_ce, images, sumdim=(1, 2, 3))
                        loss_ce = rec_loss_ce

                    loss = (1.0 - ce_factor) * loss_vae + ce_factor * loss_ce

                    if use_geco and ce_factor < 1:
                        g_goal = 0.1
                        g_lr = 1e-4
                        vae_loss_ema = (1.0 - 0.9) * \
                            rec_loss_vae + 0.9 * vae_loss_ema
                        theta = geco_beta_update(
                            theta, vae_loss_ema, g_goal, g_lr, speedup=2)

                    if not torch.isfinite(loss):
                        logging.error(
                            'loss is not finite. Skipping the iteration.')
                        continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.step()
                loss = round(loss.data.item(), 4)
                train_loss += loss
                #print("\nTraining Loss: ", loss, "\n")
                runningLoss += loss
                runningLossCounter += 1
                logging.info('[%d/%d][%d/%d] Train Loss: %.4f' %
                             ((epoch+1), num_epochs, i, len(train_loader), loss))
                pbar.update(1)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))

        # Validate
        if val_loader:
            model.eval()
            with torch.no_grad():
                print('Epoch ' + str(epoch) + ': Val')
                with tqdm(total=len(val_loader)) as pbar:
                    for i, data in enumerate(val_loader):
                        # For 2D cases
                        img = data['img']['data'].squeeze(-1) * 2 - 1
                        images = Variable(img).to(device)
                        if torch.equal(data['gt']['data'], data['img']['data']):
                            gt = torch.zeros(images.shape)
                        else:
                            gt = data['gt']['data']

                        #y = [1 if gt_one.sum() > 0 else 0 for gt_one in gt]

                        with autocast():
                            x_r, z, z_dist = model(images, reparam=False)

                            kl_loss = 0
                            if beta > 0:
                                if IsVAE:
                                    kl_loss = kl_loss_fn(
                                        z_dist, sum_samples=False, sumdim=(1,)) * beta
                                else:
                                    sys.exit("KLD Not gonna work")
                                    kl_loss = kl_loss_fn(
                                        z_dist, sum_samples=False, sumdim=(1, 2)) * beta
                            if model.d == 3:
                                rec_loss_vae = rec_loss_fn(
                                    x_r, images, sum_samples=False, sumdim=(1, 2, 3, 4))
                            else:
                                rec_loss_vae = rec_loss_fn(
                                    x_r, images, sum_samples=False, sumdim=(1, 2, 3))
                            loss = kl_loss + rec_loss_vae * theta
                        #print("\Validation Loss: ", loss, "\n")
                        logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1),
                                                                        num_epochs, i, len(val_loader), loss.mean().item()))
                        pbar.update(1)
