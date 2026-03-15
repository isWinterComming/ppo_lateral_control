#!/usr/bin/env python3
from lpips import RoadSegmentationPerceptualLoss
from dataload import datagen, total_data_size, to_rgb
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import random
import numpy as np
import glob
import sys
import os

sys.path.append(os.getcwd())


class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out)

        return out


class VQVAE_Encoder(nn.Module):
    """
    VQVAE编码器
    输入: [batch_size, 6, 128, 256]
    输出: [batch_size, embedding_dim, H/8, W/8]
    """

    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512], embedding_dim=16):
        super(VQVAE_Encoder, self).__init__()

        self.layers = nn.ModuleList()

        # 初始卷积层
        self.layers.append(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1)
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))

        # 下采样层
        self.layers.append(
            nn.Conv2d(
                hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[1], hidden_dims[1]))

        # 再下采样
        self.layers.append(
            nn.Conv2d(
                hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[2]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[2], hidden_dims[2]))

        # 再下采样
        self.layers.append(
            nn.Conv2d(
                hidden_dims[2], hidden_dims[3], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[3]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[3], hidden_dims[3]))

        # 输出卷积层
        self.layers.append(
            nn.Conv2d(hidden_dims[3], embedding_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VQVAE_Decoder(nn.Module):
    """
    VQVAE解码器
    输入: [batch_size, embedding_dim, H/8, W/8]
    输出: [batch_size, 6, 128, 256]
    """

    def __init__(
        self, out_channels=3, hidden_dims=[512, 256, 128, 64], embedding_dim=16
    ):
        super(VQVAE_Decoder, self).__init__()

        self.layers = nn.ModuleList()
        # 初始卷积层
        self.layers.append(
            nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=3, padding=1)
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))

        # 上采样层1
        self.layers.append(
            nn.ConvTranspose2d(
                hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[1], hidden_dims[1]))

        # 上采样层2
        self.layers.append(
            nn.ConvTranspose2d(
                hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[2]))
        self.layers.append(nn.ReLU())
        self.layers.append(ResidualBlock(hidden_dims[2], hidden_dims[2]))

        # 上采样层3
        self.layers.append(
            nn.ConvTranspose2d(
                hidden_dims[2], hidden_dims[3], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[3]))
        self.layers.append(nn.ReLU())

        # 上采样层3
        self.layers.append(
            nn.ConvTranspose2d(
                hidden_dims[3], hidden_dims[3], kernel_size=4, stride=2, padding=1
            )
        )
        self.layers.append(nn.BatchNorm2d(hidden_dims[3]))
        self.layers.append(nn.ReLU())

        # 输出层
        self.layers.append(
            nn.Conv2d(hidden_dims[3], out_channels, kernel_size=3, padding=1)
        )
        # self.layers.append(nn.Tanh())  # 假设输入图像已经归一化到[0,1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PatchMasker:
    """图像块掩码生成器"""
    def __init__(self, patch_size=16, mask_ratio=0.75):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def __call__(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 创建patch级别的掩码
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        
        mask = torch.ones(B, 1, H_p, W_p, device=device)
        num_patches = H_p * W_p
        num_mask = int(num_patches * self.mask_ratio)
        
        # 随机掩码
        for b in range(B):
            indices = torch.randperm(num_patches, device=device)[:num_mask]
            for idx in indices:
                h = idx // W_p
                w = idx % W_p
                mask[b, 0, h, w] = 0
        
        # 上采样到原图大小
        mask_full = F.interpolate(mask, size=(H, W), mode='nearest')
        return mask_full.expand(-1, 3, -1, -1)  # [B, C, H, W]


class VAE_8_16_32(nn.Module):
    def __init__(self, in_channels=3, latent_channels=10, SNR = 2**2 - 1):
        super(VAE_8_16_32, self).__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = VQVAE_Encoder(in_channels=3)

        # 解码器 - 输入 [8, 16, 32] -> 输出 [6, 128, 256]
        self.decoder = VQVAE_Decoder()

        self.SNR = SNR
        self.use_mask_token = False
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1))

    def encode(self, x):
        """编码输入到潜在分布参数"""
        z = self.encoder(x)  # [batch, latent_channels*2, 16, 32]
        # x0 = torch.einsum('b c h w -> b h w c', z).contiguous()
        x0 = z.view(-1, 16, 128)
        # mu = torch.zeros(h.shape, dtype=torch.float32).cuda()
        # noise generation
        # every chanel has 256 bit info ,total is  1024/2 * 6  = 3072 bits
        # 16 * 8 * ( 5 * 8)
        # 512 * log2(31 + 1) = 512 * 5 = 2560bits
        elapse_noise = (
            0.1 * torch.randn(x0.shape, device=x0.device)
            if self.training
            else torch.zeros_like(x0, device=x0.device)
        )

        z_out = torch.nn.functional.normalize(x0, p=2.0, dim=1, eps=1e-12) * np.sqrt(self.SNR * x0.shape[-1]) + elapse_noise
        # z_out = torch.einsum('b h w c -> b c h w', z_out)
        z_out = z_out.view(-1, 16, 8, 16)
        # mu, logvar = torch.chunk(h, 2, dim=1)  # 分割为均值和方差
        return z_out


    def decode(self, z):
        """从潜在变量解码"""
        return self.decoder(z)

    def apply_mask(self, x, mask):
        """应用掩码到输入图像"""
        if self.use_mask_token:
            # 方式1：使用可学习的mask token
            mask_token_full = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            masked_x = x * mask + mask_token_full * (1 - mask)
        else:
            # 方式2：用0填充（不推荐，但有论文这样做）
            # 通常会加上归一化，使0在合理范围内
            # masked_x = x * mask
            masked_x = x * mask + -1.0 * (1 - mask)
            # 注意：如果数据是标准化过的，0可能不在合理范围内
        
        return masked_x
        
    def forward(self, x, mask_ratio=0.7):
        B, C, H, W = x.shape
        
        # 1. 生成掩码
        masker = PatchMasker(patch_size=16, mask_ratio=mask_ratio)
        mask = masker(x)  # 1=可见, 0=掩码
        
        # 2. 应用掩码
        masked_x = self.apply_mask(x, mask)
        
        # 3. 编码
        latent = self.encoder(masked_x)
        # latent = self.encoder(torch.cat([masked_x, mask[:,0:1]], dim=1))
        
        # 4. 解码
        reconstruction = self.decoder(latent)
        
        # 5. 计算损失（只关注被掩码区域）
        # 注意：这里(1-mask)得到的是被掩码的区域
        masked_reconstruction = reconstruction * (1 - mask)
        masked_target = x * (1 - mask)

        rec_imgs = (1 - mask) * reconstruction + mask * x
        loss = F.smooth_l1_loss(127.5*masked_reconstruction , 127.5*masked_target)
        return loss, rec_imgs


 