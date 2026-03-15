import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import numpy as np
import torchvision.models as models
import time

ANCHOR_TIME = np.array((0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
                        0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
                        0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
                        2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
                        3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
                        6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
                        8.7890625 ,  9.38476562, 10.))


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
            nn.Dropout(drop_cof),
            nn.ReLU(),
            nn.Linear(res_size, in_size),
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.block(x))


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

        self.feat_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
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
            ResBlock1D(64, 128),
            nn.Linear(64, 64),
        )
        # self.tf = SimpleTransformer()

        self.lat_enc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
        )


    def forward(self, latent_feature, feat_buff):
        # print(latent_feature.shape)
        x0 = self.feat_head(latent_feature)
        # print(x0.shape)
        # noise generation
        SNR = 2 ** 4 - 1
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
        pose_preds = self.pose_head(x1)
        out_preds = torch.cat([x1, plan_preds, pose_preds], dim=1)
        return out_preds, x1


class MTPLoss(nn.Module):
    """mulit predict trajectory , could refer ..."""

    def __init__(self):
        super(MTPLoss, self).__init__()
        self.reg_loss_func_L1 = nn.SmoothL1Loss(reduction="none")
        self.reg_loss_func_mse = nn.MSELoss(reduction="none")
        self.cls_loss_func = nn.CrossEntropyLoss()
        self.distance_func = nn.CosineSimilarity(dim=2)
        self.prob_func = nn.Softmax(dim=0)

        self.traj_len = 33
        self.traj_num = 5
        self.elu = nn.ELU()
        self.count = 0
        self.is_use_angle_dist = True

    def kl_reg_loss(
        self,
        eps: float,
        gt: torch.Tensor,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
        ep,
    ) -> torch.Tensor:
        # get mean error
        # mean_y = gt.mean().cpu().numpy()
        # mean_err = (mean_pred - gt).mean().cpu().numpy()
        # because we most want to get good mean value, should we train 5 steps eveary one varance update
        # if self.count %20==0:
        #   sigma = torch.add(self.elu(std_pred), 1+eps)
        # else:
        #   sigma = torch.add(self.elu(std_pred), 1+eps).detach()

        # sigma = torch.add(self.elu(std_pred), 1+eps)
        # prob = 1/(2*sigma)*torch.exp(-torch.abs(gt - mean_pred)/sigma)

        sigma = torch.add(self.elu(std_pred), 1 + eps)
        m_x = torch.distributions.laplace.Laplace(loc=mean_pred, scale=sigma)
        # with same prob, we want the simma more lower
        loss_x = (
            -1 * (m_x.log_prob(gt)) * min(1, (ep+1) / 8)
            + self.reg_loss_func_L1(mean_pred, gt) * max(0.1, 1 - (ep+1) / 8)
            + 0.1 * self.reg_loss_func_L1(sigma, 0 * sigma)
        )
        # loss_x = self.reg_loss_func_L1(mean_pred, gt)
        return loss_x

    def kl_reg_loss_L1(
        self,
        eps: float,
        gt: torch.Tensor,
        mean_pred: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        # because we most want to get good mean value, should we train 5 steps eveary one varance update
        if self.count % 100 == 0:
            sigma = torch.add(self.elu(log_sigma), 1 + eps)
            # sigma = torch.exp(log_std)
        else:
            sigma = torch.add(self.elu(log_sigma), 1 + eps).detach()
        # smooth l1 losss already divide by 2.
        return torch.log(sigma * sigma) / 2.0 + self.reg_loss_func_L1(mean_pred, gt) / (
            sigma * sigma
        )

    def kl_reg_loss_fix_std(
        self,
        eps: float,
        gt: torch.Tensor,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
    ) -> torch.Tensor:
        # return (2*torch.log(sigma) + smooth_L1_err/(sigma*sigma))/2.
        sigma = 1.0
        return (
            np.log(np.sqrt(2 * np.pi))
            + np.log(sigma)
            + self.reg_loss_func_mse(mean_pred, gt) / (2 * sigma * sigma)
        )
        # return np.log(np.sqrt(2*np.pi)) + torch.log(sigma) + self.reg_loss_func_mse(mean_pred, gt)/(2*sigma*sigma) + torch.log(sigma)

    def kl_reg_loss_L1_fix_std(
        self,
        eps: float,
        gt: torch.Tensor,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
    ) -> torch.Tensor:
        # return (2*torch.log(sigma) + smooth_L1_err/(sigma*sigma))/2.
        sigma = 1.0
        return (
            np.log(np.sqrt(2 * np.pi))
            + np.log(sigma)
            + self.reg_loss_func_L1(mean_pred, gt) / (sigma * sigma)
        )
        # return np.log(np.sqrt(2*np.pi)) + torch.log(sigma) + self.reg_loss_func_L1(mean_pred, gt)/(2*sigma)

    def get_mdn_loss(
        self,
        eps: float,
        gt: torch.Tensor,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
        norm_factor=1.0,
    ) -> torch.Tensor:

        # sigma_x = torch.add(self.elu(std_pred), 1+eps)
        sigma = torch.add(self.elu(std_pred), 1 + eps)
        # sigma   = torch.exp(std_pred)
        # sigma   = torch.exp(std_pred)
        m_x = torch.distributions.Normal(
            loc=mean_pred / norm_factor, scale=sigma / norm_factor
        )
        # with same prob, we want the simma more lower
        loss_x = -1 * (m_x.log_prob(gt / norm_factor))  # + torch.log(sigma)
        return loss_x

    def forward(
        self,
        is_closeloop,
        cls_weight: float,
        pred_buffer: torch.Tensor,
        gt: torch.Tensor,
        gt_pose: torch.Tensor,
        ep: float,
    ) -> torch.Tensor:
        """current use angle distance to get closest trajectory"""
        st = time.time()
        # resample
        bat_sz = gt.shape[0]
        gt_traj = gt.reshape(bat_sz, 1, 33, 3).expand(
            -1, self.traj_num, 33, 3
        )  # B,1,3->B,M,3

        # ## reconstruct image loss.
        # recon_loss = F.smooth_l1_loss(127.5*(recon_images + 1.0), real_img_bat, reduction='none').mean()

        # trajectory loss.
        cls_sz = self.traj_num
        traj_sz = self.traj_len * self.traj_num * 3

        plan_cls = pred_buffer[:, 0:cls_sz]
        pred_traj = pred_buffer[:, cls_sz: cls_sz + traj_sz].reshape(
            -1, self.traj_num, self.traj_len, 3
        )
        pred_traj_std = pred_buffer[
            :, cls_sz + traj_sz: cls_sz + traj_sz + traj_sz
        ].reshape(-1, self.traj_num, self.traj_len, 3)

        outputs = pred_buffer[:, cls_sz + traj_sz + traj_sz:]

        pred_pose = outputs[:, 0:6]
        pred_pose_std = outputs[:, 6:12]

        m_loss_x_L1 = self.reg_loss_func_L1(
            pred_traj[:, :, :, 0], gt_traj[:, :, :, 0]
        ).mean(dim=2)
        m_loss_y_L1 = self.reg_loss_func_L1(
            pred_traj[:, :, :, 1], gt_traj[:, :, :, 1]
        ).mean(dim=2)
        # m_loss_z_L1 = self.reg_loss_func_L1( pred_traj[:,:,:, 2], gt_traj[:, :,:, 2]).mean(dim=2)

        m_loss_vx = self.reg_loss_func_L1(
            pred_pose[:, 0], gt_pose[:, 0]).mean()
        m_loss_wz = self.reg_loss_func_L1(
            pred_pose[:, 5], gt_pose[:, 5]).mean()
        # print( (time.time() - st) * 15)

        # -------> caculate regression loss
        vx_loss = self.kl_reg_loss(
            1e-6, gt_pose[:, 0], pred_pose[:, 0], pred_pose_std[:, 0], ep
        ).mean()
        # vy_loss = self.kl_reg_loss(1e-6, gt_pose[:,1], pred_pose[:, 1], pred_pose_std[:, 1]).mean()
        # vz_loss = self.kl_reg_loss(1e-6, gt_pose[:,2], pred_pose[:, 2], pred_pose_std[:, 2]).mean()
        # wx_loss = self.kl_reg_loss(1e-6, gt_pose[:,3], pred_pose[:, 3], pred_pose_std[:, 3]).mean()
        # wy_loss = self.kl_reg_loss(1e-6, gt_pose[:,4], pred_pose[:, 4], pred_pose_std[:, 4]).mean()
        w_wz = 0 if is_closeloop else 1  # closeloop not training wz loss
        wz_loss = (
            w_wz
            * self.kl_reg_loss(
                1e-6, gt_pose[:, 5], pred_pose[:, 5], pred_pose_std[:, 5], ep
            ).mean()
        )

        # wz is not correct now, so we not train it!!!
        pose_loss = vx_loss + wz_loss

        # -------> caculate distance loss
        # loss_x  = self.get_mdn_loss(1e-3, gt_traj[:,:,:, 0], pred_traj[:, :,:, 0], pred_traj[:, :,:, 3])
        loss_x = self.kl_reg_loss(
            1e-6,
            gt_traj[:, :, :, 0],
            pred_traj[:, :, :, 0],
            pred_traj_std[:, :, :, 0],
            ep,
        ).mean(dim=2)
        loss_y = self.kl_reg_loss(
            1e-6,
            gt_traj[:, :, :, 1],
            pred_traj[:, :, :, 1],
            pred_traj_std[:, :, :, 1],
            ep,
        ).mean(dim=2)
        # loss_z = self.kl_reg_loss(
        #     1e-6,
        #     gt_traj[:, :, :, 2],
        #     pred_traj[:, :, :, 2],
        #     pred_traj_std[:, :, :, 2],
        #     ep,
        # ).mean(dim=2)

        # -------> add angle loss
        # delta cost
        gt_dx = torch.clip(gt_traj[:, :, 1:33, 0] -
                           gt_traj[:, :, 0:32, 0], min=0.001)
        gt_dy = gt_traj[:, :, 1:33, 1] - gt_traj[:, :, 0:32, 1]
        gt_theta = torch.atan2(gt_dy, gt_dx) * (180 / np.pi)  # deg

        DT = (
            torch.Tensor(ANCHOR_TIME[1:33] - ANCHOR_TIME[0:32])
            .expand([gt_theta.shape[0], 5, 32])
            .to(gt_theta.device)
        )
        gt_ds = (gt_dx.pow(2) + gt_dy.pow(2)).sqrt() / DT

        pred_dx = pred_traj[:, :, 1:33, 0] - pred_traj[:, :, 0:32, 0]
        pred_dy = pred_traj[:, :, 1:33, 1] - pred_traj[:, :, 0:32, 1]
        pred_theta = torch.atan2(pred_dy, pred_dx) * (180 / np.pi)  # deg

        pred_ds = (pred_dx.pow(2) + pred_dy.pow(2)).sqrt() / DT

        loss_delta_theta = self.reg_loss_func_L1(pred_theta, gt_theta).mean(
            dim=(2)
        )  # averge point angle loss
        loss_delta_ds = self.reg_loss_func_L1(pred_ds, gt_ds).mean(
            dim=(2)
        )  # averge point angle loss

        # add y-loss, most care about!
        traj_loss = (
            (loss_x + 5 * loss_y) +
            loss_delta_theta + loss_delta_ds
        )
        # -------> find best trajectory index, use angle minimum or distance minimun
        if self.is_use_angle_dist:
            angle_dist = 1 - self.distance_func(
                pred_traj[:, :, -1, 0:2], gt_traj[:, :, -1, 0:2]
            )  # B, M
            index = angle_dist.argmin(dim=1)
            pred_index = plan_cls.softmax(dim=1).argmax(dim=1)
            # min_angle_dist = angle_dist[torch.tensor(range(len(index)), device=index.device), index, ...]
            pred_angle_dist = angle_dist[torch.tensor(range(len(pred_index)), device=pred_index.device), pred_index, ...]
            pred_x = pred_traj[torch.tensor(range(len(pred_index)), device=pred_index.device), pred_index, -1,0]
            index = torch.where(pred_x * pred_angle_dist > 2.8, index, pred_index)
        else:
            index = traj_loss.argmin(dim=1)

        # first 3 epoch, training every head
        if ep <= 4:
            index = torch.randint(
                0, 5, (traj_loss.shape[0],)).to(traj_loss.device)
            cls_weight = 0
        # -------> caculate best regression trajectory loss
        reg_loss_best = traj_loss[
            torch.tensor(range(len(index)), device=index.device), index, ...
        ].mean()
        # -------> caculate best regression class loss
        gt_cls = index
        cls_loss = self.cls_loss_func(plan_cls, gt_cls)
        # we not traing class, we random train every at early  cls_weight
        total_loss = pose_loss + reg_loss_best + cls_weight * cls_loss
        self.count += 1

        # -------> send best trajectory for closed loop online control
        best_traj = pred_traj[
            torch.tensor(range(len(index)), device=index.device), index, ...
        ]

        # return
        return (
            total_loss,
            reg_loss_best,
            cls_loss,
            m_loss_x_L1[
                torch.tensor(range(len(index)),
                             device=index.device), index, ...
            ].mean(),
            m_loss_y_L1[
                torch.tensor(range(len(index)),
                             device=index.device), index, ...
            ].mean(),
            loss_delta_theta[
                torch.tensor(range(len(index)),
                             device=index.device), index, ...
            ].mean(),
            m_loss_vx,
            m_loss_wz,
        ), best_traj
