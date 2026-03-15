#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import torch.nn.functional as F
import time
import json
import cv2

from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt 

from  tqdm import tqdm
from dataset.dd import DD


with open("train_config.json", 'r') as f:
	conf = json.load(f)

## parameters
P_dvc = torch.device('cuda:0')
P_optim_per_n_steps = conf['optim_per_steps']
P_seq_length = conf['sequence_len']
P_bs = conf['batch_sz']
P_train_datadir = conf['training_datadir']
P_train_epoch = conf['epoch']
P_lr = conf['lr']
P_usewandb = conf['use_wandb']
P_mtp_alpha = conf['mtp_alpha']
P_grad_clip = conf['grad_clip']

# init wandb
if P_usewandb:
  import wandb
  runner_w = wandb.init(project="trajectory prediction")
  wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 128
  }
  runner = wandb.Artifact( "trajectory-prediction", type="dataset", description="test artifact")



def main():
  # self.load_model()
  from models.resnet18 import Encoder, Generator

  train_steps = 0
  mpt_loss = MTPLoss()
  dvc = torch.device('cuda:0')
  encoder = Encoder(3).to(dvc)
  decoder = Generator().to(dvc)

  if P_usewandb:
    wandb.watch(encoder, log="gradients",  log_freq=10)
    wandb.watch(decoder, log="gradients",  log_freq=10)
  
  # mdoel optiomizter
  vae_optim = torch.optim.Adam([{'params': encoder.parameters()},
                                {'params': decoder.parameters(),}], lr=P_lr, betas=(0.5, 0.999), weight_decay=1e-3)


  # load dataset
  ds = DD()
  dd = DataLoader(ds, batch_size=P_bs, shuffle=True, drop_last=True)
  total_step_one_epoch = len(ds)/P_bs
  print(f"=========================== dataset length is {total_step_one_epoch}.")

  for ep in range(P_train_epoch):
    print(f'************************** trainig: {ep+1} **************************\n')
    # deal with one epoch data loop
    kld_w = 1

    with tqdm(total=int(total_step_one_epoch), desc='training process: ', colour='GREEN') as pbar:
      for tx in dd:
        train_steps += 1
        ## two continious image
        X_in = tx.cuda().float()/127.5 - 1
        # print(X_in.shape)

        ## first encoder it
        _, mu, std = encoder(X_in)

        ## then sample gauss sdistribution
        # std = mpt_loss.elu(logvar) + 1. + 1e-4 # more smoother
        eps = torch.randn_like(mu)  #eps: bs,latent_size
        z_noise = mu + eps*std  #z: bs,latent_size

        ## then get generated image
        gen_img_out = decoder(z_noise)

        ## last we caculate loss
        reg_loss = (mpt_loss.reg_loss_func_mse(gen_img_out, X_in)).mean() 
        kl_loss = ( -0.5 * (1 + torch.log(std*std) - mu*mu - std*std)).mean()

        # print(reg_loss, kl_loss)
        vae_loss = (reg_loss +  kl_loss * 512/(3*128*256)).mean()

        # update pbar
        pbar.update(1)
        pbar.set_description(f"epoch:{ep}, step:{train_steps}, loss:{vae_loss}, loss_kl:{kl_loss.mean()}, reg_loss:{reg_loss.mean()}")

        ## first log loss
        if P_usewandb:
          wandb.log({
				          	"miu": wandb.Histogram(mu.detach().cpu().numpy()),
				          	"std": wandb.Histogram(std.detach().cpu().numpy()),
				          	"vae_loss": vae_loss.cpu().detach().numpy(), 
                    "reg_loss": reg_loss.mean().cpu().detach().numpy()/(3*128*256), 
                    "loss_KLD": kl_loss.mean().cpu().detach().numpy()/512 })

        vae_optim.zero_grad()  # clear gradients for this training step
        vae_loss.backward()  # back propagation, compute gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 20.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 20.0)
        vae_optim.step()

        ## save model with 100 steps
        if train_steps % 1000==0:
          torch.save(encoder, 'out/encoder.pt')
          torch.save(decoder, 'out/decoder.pt')
          z_noise = torch.randn(16, 512).cuda()
          gen_img_bat = decoder(z_noise).reshape(16, 3,128,256).cpu().detach().numpy()

          for j in range(16):
            # print(gen_img_out.shape)
            gen_img_1 = gen_img_out[j,:,:,:].cpu().detach().numpy()
            gen_img_1 = (gen_img_1.transpose((1,2,0)) + 1)*127.5
            cv2.imwrite(f"./out/images_gen_vae/dec_img_{train_steps}_{j}.png", gen_img_1)

          for j in range(16):
            gen_img_2 = gen_img_bat[j,:,:,:]
            gen_img_2 = (gen_img_2.transpose((1,2,0)) + 1)*127.5
            cv2.imwrite(f"./out/images_gen_vae/gen_img_{train_steps}_{j}.png", gen_img_2)

if __name__=='__main__':
	main()