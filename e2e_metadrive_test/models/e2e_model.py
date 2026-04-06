import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import numpy as np
#from utils import valid_segment_slice, ANCHOR_TIME
import torchvision.models as models
import time

class GRU1D(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU1D, self).__init__()
        self.fc1 = nn.Linear(input_size, 512 * 3)
        self.fc2 = nn.Linear(hidden_size, 512 * 3)

    def forward(self, x, hidden_st):
        x1 = self.fc1(x)
        h1 = self.fc2(hidden_st)

        s1 = torch.add(x1[:, 0:512], h1[:, 0:512]).sigmoid()
        s2 = torch.add(x1[:, 512 * 2: 512 * 3],
                       h1[:, 512 * 2: 512 * 3]).sigmoid()

        t = (x1[:, 512: 512 * 2] + torch.mul(h1[:, 512: 512 * 2], s2)).tanh()

        return t + torch.mul(s1, hidden_st - t)


class ResBlock1D(nn.Module):
    def __init__(self, in_size, res_size, drop_cof=0.1):
        super(ResBlock1D, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_size, res_size),
            nn.ReLU(),
            nn.Dropout(drop_cof),
            nn.Linear(res_size, in_size),
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.block(x))
        # return x + self.block(x)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 设置可以训练的参数矩阵
        self.w_xr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_xz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_xh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h):

        # --------------------------------GRU核心公式-----------------------------------
        # x形状是(batchsize,inputsize),w_xz矩阵形状是(inputsize,hiddensize)
        # torch.mm是矩阵乘法，这样(torch.mm(x,self.w_xz)的形状是(batchsize,hiddensize)

        z = torch.sigmoid(
            (torch.matmul(x, self.w_xz) + torch.matmul(h, self.w_hz) + self.b_z)
        )
        r = torch.sigmoid(
            (torch.matmul(x, self.w_xr) + torch.matmul(h, self.w_hr) + self.b_r)
        )
        h_tilde = torch.tanh(
            (torch.matmul(x, self.w_xh) + torch.matmul(r * h, self.w_hh) + self.b_h)
        )

        return (1 - z) * h + z * h_tilde

class TransformerEncoderProjection(nn.Module):
    def __init__(self, input_dim=128, target_dim=512):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        # 使用 [CLS] token 或池化
        self.pooling = nn.AdaptiveAvgPool1d(64)
        self.projection = nn.Linear(2048, target_dim)
    
    def forward(self, x):
        # x: [batch_size, 32, 128]
        # Transformer编码：保留所有序列信息
        encoded = self.transformer_encoder(x)  # [batch_size, 32, 128]
        
        # 全局池化（比简单平均更好）
        pooled = self.pooling(encoded)  # [batch_size, 128, 1]
        return self.projection(pooled.flatten(start_dim=1)).relu()  # [batch_size, 1024]


# ============= 位置编码 =============

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class SimpleTransformer(nn.Module):
    """
    处理 [B, 9, 512] 输入的简单 Transformer
    """
    def __init__(
        self,
        d_model=512,        # 特征维度
        nhead=8,            # 8头注意力
        num_layers=1,       # 3层Transformer
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        
        # 位置编码（9个位置）
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9)
        
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 输入是 [B, 9, 512]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        
    def forward(self, x):
        """
        x: [batch_size, 9, 512]
        """
        # 添加位置编码

        # print(x.shape)
        x = self.pos_encoder(x)  # [B, 9, 512]
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [B, 9, 512]
               
        # selce latest one
        out = x[:,-1]
        
        return out

class PlanningModel(nn.Module):
    traj_size = [5, 33, 3]

    def __init__(self):
        super(PlanningModel, self).__init__()
        self.enc = EfficientNet.from_pretrained('efficientnet-b1', in_channels=6)
        self.feat_head = nn.Sequential(
        # 6, 450, 800 -> 1408, 14, 25
        # nn.AdaptiveMaxPool2d((4, 8)),  # 1408, 4, 8
        nn.BatchNorm2d(1280),
        nn.Conv2d(1280, 32, 1),  # 32, 4, 8
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),
        )
        
        self.plan_head = nn.Sequential(
            nn.Linear(512 + 1024, 512),
            nn.ReLU(),
            ResBlock1D(512,1024),
            nn.Linear(512, 256),
            nn.ReLU(),
            ResBlock1D(256,256),
            nn.Linear(
                256,
                self.traj_size[0]
                + 2 * self.traj_size[0] *
                self.traj_size[1] * self.traj_size[2],
            ),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            ResBlock1D(32, 32),
            nn.Linear(32, 64),
        )
        # self.tf = SimpleTransformer()

        self.l2_norm = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            )

        self.lat_enc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
        )


    def forward(self, feed_imgs, feat_buff):
        # print(latent_feature.shape)
        big_imgs = feed_imgs/255. # to[0,1]
        IMAGENET_MEAN = [0.485, 0.456, 0.406] # rgb
        IMAGENET_STD = [0.229, 0.224, 0.225]
        big_imgs[:,0] = feed_imgs[:,0]/IMAGENET_STD[0] - IMAGENET_MEAN[0] # r
        big_imgs[:,1] = feed_imgs[:,1]/IMAGENET_STD[1] - IMAGENET_MEAN[1] # g
        big_imgs[:,2] = feed_imgs[:,2]/IMAGENET_STD[2] - IMAGENET_MEAN[2] # b
        big_imgs[:,3] = feed_imgs[:,3]/IMAGENET_STD[0] - IMAGENET_MEAN[0] # r
        big_imgs[:,4] = feed_imgs[:,4]/IMAGENET_STD[1] - IMAGENET_MEAN[1] # g
        big_imgs[:,5] = feed_imgs[:,5]/IMAGENET_STD[2] - IMAGENET_MEAN[2] # b
        
        xf = self.feat_head(self.enc.extract_features(big_imgs ) )
        x0 = self.l2_norm(xf)
        # print(x0.shape)
        # noise generation, C = 640.0 bits
        SNR = 2 ** 2.5 - 1
        if self.training:
            x1 = torch.nn.functional.normalize(x0, p=2.0, dim=1, eps=1e-12)*np.sqrt(
                SNR*x0.shape[-1]) + 1*torch.randn(x0.shape, device=x0.device)
        else:
            x1 = torch.nn.functional.normalize(
                x0, p=2.0, dim=1, eps=1e-12)*np.sqrt(SNR*x0.shape[-1])
        
        x2 = self.lat_enc(feat_buff[:, -1, :])
        x3 = self.lat_enc(feat_buff[:, -2, :])
        x4 = self.lat_enc(feat_buff[:, -3, :])
        x5 = self.lat_enc(feat_buff[:, -4, :])
        x6 = self.lat_enc(feat_buff[:, -5, :])
        x7 = self.lat_enc(feat_buff[:, -6, :])
        x8 = self.lat_enc(feat_buff[:, -7, :])
        x9 = self.lat_enc(feat_buff[:, -8, :])
        x_enc = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9], dim=1)
        # x_enc = self.tf(torch.cat([feat_buff[:, -8:, :], x1.view(-1, 1, 512)], dim=1))

        plan_preds = self.plan_head(x_enc)
        pose_preds = self.pose_head(x0)
        out_preds = torch.cat([x1, plan_preds, pose_preds], dim=1)
        return out_preds, x1
        