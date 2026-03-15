import numpy as np
import onnxruntime as ort
import cv2
import torch
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
from common.transformations.camera import get_view_frame_from_road_frame

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
    
MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                                [0.0,  910.0,   47.6],
                                [0.0,  0.0,     1.0]])

ANCHOR_TIME = np.array((0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
                        0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
                        0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
                        2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
                        3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
                        6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
                        8.7890625 ,  9.38476562, 10.))

C2_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 1164],
                            [0.0,  910.0,   0.5 * 874],
                            [0.0,  0.0,     1.0]])


def reshape_yuv(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)

  in_img1[0] = frames[0:H:2, 0::2]
  in_img1[1] = frames[1:H:2, 0::2]
  in_img1[2] = frames[0:H:2, 1::2]
  in_img1[3] = frames[1:H:2, 1::2]
  in_img1[4] = frames[H:H+H//4].reshape((H//2, W//2))
  in_img1[5] = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return in_img1


def draw_path(device_path, img, width=0.6, height=1.22, fill_color=(255,255,255), line_color=(0,255,0)) -> None:
  bs = device_path.shape[0]
  device_path_l = device_path + np.array([0, 0, height])                                                                    
  device_path_r = device_path + np.array([0, 0, height]) 

  # calib frame to raod frame                                                         
  device_path_l[:,1] -= width                                                                                               
  device_path_r[:,1] += width
  device_path_l[:,2] = -1*device_path_l[:,2]  
  device_path_l[:,1] = -1*device_path_l[:,1] 
  device_path_r[:,2] = -1*device_path_r[:,2]  
  device_path_r[:,1] = -1*device_path_r[:,1] 

  m1 = get_view_frame_from_road_frame(0, 0, 0, 0,0)
  calib_pts = np.vstack((device_path_l.T, np.ones((1,bs)) ))
  view_pts = m1 @ calib_pts
  for i in range(bs):
    view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
    view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
    view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

  img_pts_l = MEDMDL_INSMATRIX @ view_pts
  img_pts_l = img_pts_l.astype(int)
  calib_pts = np.vstack((device_path_r.T, np.ones((1,bs)) ))
  view_pts = m1 @ calib_pts
  for i in range(bs):
    view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
    view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
    view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

  img_pts_r = MEDMDL_INSMATRIX @ view_pts
  img_pts_r = img_pts_r.astype(int)
  for i in range(1, img_pts_l.shape[1]):
    #check valid
    if img_pts_l[2,i] >0 and img_pts_r[2,i] >0:
      u1 = img_pts_l[0, i-1]
      v1 = img_pts_l[1, i-1]
      u2 = img_pts_r[0, i-1]
      v2 = img_pts_r[1, i-1]
      u3 = img_pts_l[0, i]
      v3 = img_pts_l[1, i]
      u4 = img_pts_r[0, i]
      v4 = img_pts_r[1, i]
      pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
      cv2.fillPoly(img,[pts],fill_color)
      cv2.polylines(img,[pts],True,line_color)



def get_calib_matrix(cam_insmatrixs=C2_INSMATRIX, pos_bias=0, theta_bias=0, 
                     ang_x=0, ang_y=0, ang_z=0, dev_height=1.22, lat_bias=0) -> np.array:
	"""
		Func: if the camera heading angle of position has been changed, the trasform matrix is changed too.
	"""
	camera_frame_from_ground = np.dot(cam_insmatrixs,
																			get_view_frame_from_road_frame(ang_x, ang_y, ang_z, dev_height, lat_bias))[:, (0, 1, 3)]
	calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
																			get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
	calib_msg = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

	return calib_msg
  

## onnx planning demo model runner, get images, pred trajectory!
class PlanModel():
	def __init__(self, cuda=False):
		options = ort.SessionOptions()
		provider = 'CUDAExecutionProvider' if cuda else 'CPUExecutionProvider'
		self.session = ort.InferenceSession(f'/home/chaoqun/Desktop/ai_project/end2end_lateral/out/planning_model.onnx', options, [provider])

		# print shapes
		input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
		output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
		print('input shapes : ', input_shapes)
		print('output shapes: ', output_shapes)

		self.recurrent_state = np.zeros((1, 512)).astype(np.float32)
		self.last_feed_imgs = np.zeros((1, 6, 128, 256)).astype(np.uint8)

	#staticmethod
	def my_softmax(x):
			exp_x = np.exp(x)
			return exp_x/np.sum(exp_x)

	def run(self, feed_img):
		big_imgs = np.concatenate([self.last_feed_imgs, feed_img], axis=1).astype(np.float32)
		model_out = self.session.run(None, {'big_imgs': big_imgs, 'hiddenst_in': self.recurrent_state})
		preds_buffer, self.recurrent_state = model_out[0], model_out[1]
		pred_cls_obj = preds_buffer[:,0:5]
		pred_traj_obj = preds_buffer[:,5:5+5*33*6].reshape(5, 33, 6)

		# print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]) )
		traj_prob = PlanModel.my_softmax(pred_cls_obj)
		traj_batch = []
		for j in range(5):
			traj_x  = (pred_traj_obj[j, :, 0]) 
			traj_y  =  (pred_traj_obj[j, :, 1] )
			traj_xstd  = (pred_traj_obj[j, :, 2]) 
			traj_ystd  =  (pred_traj_obj[j, :, 3] )
			traj_batch.append((traj_prob[0,j],traj_x, traj_y, traj_xstd, traj_ystd ))
		
		self.last_feed_imgs = feed_img
		return traj_batch



## onnx planning demo model runner, get images, pred trajectory!
class PlanModelV2():
	def __init__(self, cuda=False):
		self.devc = torch.device('cpu')
		vae_model = VAE_8_16_32()
		vae_model = torch.load("mae_cnn_model.pt", weights_only=False).to(self.devc)
		vae_model.eval()

		# from models.e2e_model import PlanningModel
		plan_model = torch.load('planning_model.pt', weights_only=False).to(self.devc)    # 默认GPU
		plan_model.eval()

		self.vision_net = vae_model
		self.policy_net = plan_model


		self.feat_buff = torch.zeros((1, 20, 512)).to(self.devc)
		self.last_img_latent = torch.zeros((1, 16, 8, 16)).to(self.devc)

	#staticmethod
	def my_softmax(x):
			exp_x = np.exp(x)
			return exp_x/np.sum(exp_x)

	def run(self, feed_img):
		input_bev = torch.from_numpy(np.ascontiguousarray(feed_img)).to(self.devc)
		now_img_latent = self.vision_net.encoder(input_bev/127.5 - 1.)
		latent_feature = torch.cat([self.last_img_latent, now_img_latent], dim=1)

		# Trainging started with the 2rd image.
		out_preds, latent_mem = self.policy_net(latent_feature, self.feat_buff)

		# Time series stitch
		feat_buff = self.feat_buff.roll(shifts=-1, dims=1)
		feat_buff[:, -1, :] = latent_mem
		self.last_img_latent = now_img_latent.detach().clone()

		out_preds = out_preds    
		preds_buffer = out_preds[:, 512:]
		preds_buffer = out_preds[:, 512:]
		pred_cls_obj = preds_buffer[:,0:5]
		pred_traj_obj = preds_buffer[:,5:5+5*33*3].reshape(5, 33, 3)
		pred_traj_obj_std = preds_buffer[:,5+5*33*3:5+5*33*3*2].reshape(5, 33, 3)

		elu = torch.nn.ELU()
		print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]),  elu(preds_buffer[:,5+5*33*6 + 6]) + 1.0)
		traj_prob = pred_cls_obj.softmax(dim=1).cpu().detach().numpy()
		
		traj_batch = []
		for j in range(5):
				
				traj_x  = pred_traj_obj[j, :, 0].cpu().detach().numpy()
				traj_y  = pred_traj_obj[j, :, 1].cpu().detach().numpy()
				traj_xstd  = elu(pred_traj_obj_std[j, :, 0]).cpu().detach().numpy() + 1.
				traj_ystd  =  elu(pred_traj_obj_std[j, :, 1]).cpu().detach().numpy() + 1.
				traj_batch.append((traj_prob[0,j],traj_x, traj_y, traj_xstd, traj_ystd))
                    
				print(traj_prob[0,j],traj_x, traj_y,)
		
		self.last_feed_imgs = feed_img
		return traj_batch
	
# pm = PlanModelV2()
# pm.run(np.zeros((1, 3, 128, 256)).astype(np.float32))

def trans_global_t0_local(x0, y0, theta0):
	transform_maxtrix = np.array([[np.cos(theta0), np.sin(theta0),  -(x0*np.cos(theta0) + y0*np.sin(theta0))], 
												[-np.sin(theta0), np.cos(theta0),  (x0*np.sin(theta0)  -  y0*np.cos(theta0))], 
												[0, 0,  1 ] ])

	return transform_maxtrix


def trans_local_t0_global(x0, y0, theta0):
	transform_maxtrix = np.array([[np.cos(theta0), np.sin(theta0), 0], 
																[-np.sin(theta0), np.cos(theta0), 0],
																[0,0,1]])

	return transform_maxtrix
