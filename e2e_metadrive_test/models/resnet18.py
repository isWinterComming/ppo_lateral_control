import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import numpy as np
from utils import valid_segment_slice, ANCHOR_TIME


class GRU1D(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU1D, self).__init__()
        self.fc1 = nn.Linear(input_size, 512*3)
        self.fc2 = nn.Linear(hidden_size, 512*3)

    def forward(self, x, hidden_st):
        x1 = self.fc1(x)
        h1 = self.fc2(hidden_st)

        s1 = torch.add(x1[:, 0:512], h1[:, 0:512]).sigmoid()
        s2 = torch.add(x1[:, 512*2:512*3], h1[:, 512*2:512*3]).sigmoid()

        t = (x1[:, 512:512*2] + torch.mul(h1[:, 512:512*2], s2)).tanh()

        return t + torch.mul(s1, hidden_st - t)

class ResBlock1D(nn.Module):
  def __init__(self, in_size, res_size, drop_cof=0.1):
      super(ResBlock1D, self).__init__()

      self.block = nn.Sequential(
          nn.Linear(in_size, res_size),
          # nn.Dropout(drop_cof),
          nn.ReLU(),
          nn.Linear(res_size, in_size),
      )

  def forward(self, x):
    return torch.nn.functional.relu(x + self.block(x))


# todo Bottleneck
class Bottleneck(nn.Module):
  """
  __init__
      in_channel：残差块输入通道数
      out_channel：残差块输出通道数
      stride：卷积步长
      downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
  """
  expansion = 3   # 残差块第3个卷积层的通道膨胀倍率
  def __init__(self, in_channel, out_channel, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                            kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
    self.bn1 = nn.BatchNorm2d(num_features=out_channel)
    self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                            kernel_size=3, stride=stride, bias=False, padding=1)  # H/2，W/2。C不变
    self.bn2 = nn.BatchNorm2d(num_features=out_channel)
    self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, 
                            kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
    self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    identity = x    # 将原始输入暂存为shortcut的输出
    if self.downsample is not None:
        # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)
        identity = self.downsample(x)   
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    out += identity     # 残差连接
    out = self.relu(out)
    return out
  

# todo ResNet
class ResNet(nn.Module):
  def __init__(self, block, block_num, num_classes=1000):
    """
    block: 堆叠的基本模块
    block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
    num_classes: 全连接之后的分类特征维度
    """
    super(ResNet, self).__init__()
    self.in_channel = 64    # conv1的输出维度
    self.conv1 = nn.Conv2d(in_channels=12, out_channels=self.in_channel, 
                            kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
    self.bn1 = nn.BatchNorm2d(self.in_channel)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变

    # H,W不变。downsample控制的shortcut，out_channel=64x4=256
    self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   
    # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
    self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  
    # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
    self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  
    # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048
    self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  
    # self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
    # self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)
    self.layer5 = nn.Sequential(
        nn.Conv2d(1536, 32, 1),  # 32, 4, 8
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(1024, 512),
    )
    # 权重初始化
    for m in self.modules():    
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def _make_layer(self, block, channel, block_num, stride=1):
    """
    block: 堆叠的基本模块
    channel: 每个stage中堆叠模块的第一个卷积的卷积核个数，对resnet50分别是:64,128,256,512
    block_num: 当期stage堆叠block个数
    stride: 默认卷积步长
    """
    downsample = None   # 用于控制shorcut路的
    # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4
    # 因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
    if stride != 1 or self.in_channel != channel*block.expansion: 
      # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2  
      downsample = nn.Sequential(
        nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, 
                  kernel_size=1, stride=stride, bias=False), 
        nn.BatchNorm2d(num_features=channel*block.expansion))
    # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
    layers = [] 

    # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
    layers.append(block(in_channel=self.in_channel, out_channel=channel, 
                        downsample=downsample, stride=stride)) 
    
    # 在下一次调用_make_layer函数的时候，self.in_channel已经x4
    self.in_channel = channel*block.expansion   

    # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
    for _ in range(1, block_num):  
        layers.append(block(in_channel=self.in_channel, out_channel=channel))
    return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # print(x.shape)
    x = self.layer5(x)
    # print(x.shape)
    return x


import math
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

        z = torch.sigmoid((torch.matmul(x, self.w_xz) + torch.matmul(h, self.w_hz) + self.b_z))
        r = torch.sigmoid((torch.matmul(x, self.w_xr) + torch.matmul(h, self.w_hr) + self.b_r))
        h_tilde = torch.tanh((torch.matmul(x, self.w_xh) + torch.matmul(r * h, self.w_hh) + self.b_h))

        return (1 - z) * h + z * h_tilde
    

class PlanningModel(nn.Module):
  traj_size=[5,33,3]
  def __init__(self, use_resnet = True):
    super(PlanningModel, self).__init__()
    self.use_resnet = use_resnet
    if use_resnet:
      self.enc = ResNet(block=Bottleneck, block_num=[3, 4, 6, 3])
    else:
      self.enc = EfficientNet.from_pretrained('efficientnet-b2', in_channels=12)
      self.feat_head = nn.Sequential(
      # 6, 450, 800 -> 1408, 14, 25
      # nn.AdaptiveMaxPool2d((4, 8)),  # 1408, 4, 8
      nn.BatchNorm2d(1408),
      nn.Conv2d(1408, 32, 1),  # 32, 4, 8
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      )
    
    # self.feat_norm = nn.Sequential(
    #    ResBlock1D(512, 1024),
    #    nn.Linear(512, 512),
    # )

    self.plan_head = nn.Sequential(
      nn.Linear(512+512, 512),
      nn.ReLU(),
      # ResBlock1D(512, 1024),
      # nn.Linear(512, 256),
      # nn.ReLU(),
      # ResBlock1D(256,256),
      nn.Linear(512, self.traj_size[0] + 2*self.traj_size[0]*self.traj_size[1]*self.traj_size[2]),
    )

    self.pose_head = nn.Sequential(
      nn.Linear(512, 64),
      nn.ReLU(),
      # nn.Linear(256, 64),
      # nn.ReLU(),
      # ResBlock1D(64, 64),
      nn.Linear(64, 12),
    )

    # self.context_gru = nn.GRU(input_size=1024, 
    #                           hidden_size=512, 
    #                           bidirectional=False, 
    #                           batch_first=True)  # 1024 out

    # self.context_gru = GRU1D(1024,512)
    self.context_gru = GRU(input_size=512, hidden_size=512)
    # self.temporal = nn.Sequential(nn.Linear(512, 64), nn.ReLU())

    # # initial weights
    # for name, m in self.named_modules():
    #     # print(name=='plan_head.6', type(m))
    #     # if isinstance(m, nn.Conv2d):
    #     #     nn.init.kaiming_normal_(m.weight, mode="fan_out")
    #     #     if m.bias is not None:
    #     #         nn.init.zeros_(m.bias)
    #     # elif isinstance(m, nn.BatchNorm2d):
    #     #     nn.init.ones_(m.weight)
    #     #     nn.init.zeros_(m.bias)
    #     # elif isinstance(m, nn.Linear) and name=='plan_head.6':
    #     #     nn.init.normal_(m.weight, 0, 0.1)
    #     #     nn.init.zeros_(m.bias)
    #     # elif isinstance(m, nn.Linear):
    #     #     nn.init.normal_(m.weight, 0, 0.01)
    #     #     nn.init.zeros_(m.bias)
    #     if isinstance(m, nn.Linear) and name=='plan_head.6':
    #         nn.init.normal_(m.weight, 0, 0.1)
    #         nn.init.zeros_(m.bias)
       
  def forward(self, x_in, feat_buff):

    bev_img = x_in/127.5 - 1

    if self.use_resnet:
      x0 = self.enc(bev_img)
    else:
      x0 = self.feat_head(self.enc.extract_features(bev_img ) )
    # x1_norm = torch.clamp(torch.norm(x0, p=2, dim=1, keepdim=True), min=1e-12)
    # x1 = x0 / (x1_norm.expand(-1, 512) )
    SNR = 20
    if self.training:
      x1 = torch.nn.functional.normalize(x0, p=2.0, dim=1, eps=1e-12)*np.sqrt(SNR*x0.shape[-1]) + 1*torch.randn(x0.shape, device=x0.device)
    else:
      x1 = torch.nn.functional.normalize(x0, p=2.0, dim=1, eps=1e-12)*np.sqrt(SNR*x0.shape[-1])

    history_feat = self.context_gru(x1, feat_buff)   #  (seq_lenght, batch_size, input_size)
    # print(history_feat.shape, history_feat.shape)

    # history_feat = torch.cat((feat_buff[:,-2,:],  feat_buff[:,-4,:] , feat_buff[:,-6,:],  feat_buff[:,-8,:], 
    #                           feat_buff[:,-10,:], feat_buff[:,-12,:] ,feat_buff[:,-14,:], feat_buff[:,-16,:]), dim=1)

    plan_preds = self.plan_head(torch.cat([history_feat, x1], dim=1))
    pose_preds = self.pose_head(x1)
    out_preds = torch.cat([x1, history_feat, plan_preds, pose_preds], dim=1)

    return out_preds
    

class MTPLoss(nn.Module):
  """ mulit predict trajectory , could refer ..."""
  def __init__(self):
    super(MTPLoss, self).__init__()
    self.reg_loss_func_L1 = nn.SmoothL1Loss(reduction='none')
    self.reg_loss_func_mse = nn.MSELoss(reduction='none')
    self.cls_loss_func = nn.CrossEntropyLoss()
    self.distance_func = nn.CosineSimilarity(dim=2)
    self.prob_func = nn.Softmax(dim=0)

    self.traj_len = 33
    self.traj_num = 5
    self.elu = nn.ELU()
    self.count = 0
    self.is_use_angle_dist = True

  def kl_reg_loss(self, eps: float, gt: torch.Tensor, mean_pred: torch.Tensor, std_pred: torch.Tensor, ep) -> torch.Tensor :
    ## get mean error
    # mean_y = gt.mean().cpu().numpy()
    # mean_err = (mean_pred - gt).mean().cpu().numpy()
    ## because we most want to get good mean value, should we train 5 steps eveary one varance update
    # if self.count %20==0:
    #   sigma = torch.add(self.elu(std_pred), 1+eps) 
    # else:
    #   sigma = torch.add(self.elu(std_pred), 1+eps).detach()

    # sigma = torch.add(self.elu(std_pred), 1+eps) 
    # prob = 1/(2*sigma)*torch.exp(-torch.abs(gt - mean_pred)/sigma)

    sigma = torch.add(self.elu(std_pred), 1+eps)
    m_x     = torch.distributions.laplace.Laplace(loc=mean_pred, scale=sigma)
    # with same prob, we want the simma more lower
    loss_x  = -1 * (m_x.log_prob(gt))*min(1, ep/12) \
          +  0.25*self.reg_loss_func_L1(sigma, 0*sigma) \
          + self.reg_loss_func_L1(mean_pred, gt)*max(0.1, 1 - ep/8)

    # loss_x = self.reg_loss_func_L1(mean_pred, gt)
    return loss_x
    

  def kl_reg_loss_L1(self, eps: float, gt: torch.Tensor, 
                        mean_pred: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor :
    ## because we most want to get good mean value, should we train 5 steps eveary one varance update
    if self.count %100==0:
      sigma = torch.add(self.elu(log_sigma), 1+eps)
      # sigma = torch.exp(log_std)
    else:
      sigma = torch.add(self.elu(log_sigma), 1+eps).detach()
    return torch.log(sigma*sigma)/2. + self.reg_loss_func_L1(mean_pred, gt)/(sigma*sigma)  # smooth l1 losss already divide by 2.


  def kl_reg_loss_fix_std(self, eps: float, gt: torch.Tensor, 
                        mean_pred: torch.Tensor, std_pred: torch.Tensor) -> torch.Tensor :
    # return (2*torch.log(sigma) + smooth_L1_err/(sigma*sigma))/2.
    sigma = 1.0
    return np.log(np.sqrt(2*np.pi)) + np.log(sigma) + self.reg_loss_func_mse(mean_pred, gt)/(2*sigma*sigma)
    # return np.log(np.sqrt(2*np.pi)) + torch.log(sigma) + self.reg_loss_func_mse(mean_pred, gt)/(2*sigma*sigma) + torch.log(sigma)

  def kl_reg_loss_L1_fix_std(self, eps: float, gt: torch.Tensor, 
                        mean_pred: torch.Tensor, std_pred: torch.Tensor) -> torch.Tensor :
    # return (2*torch.log(sigma) + smooth_L1_err/(sigma*sigma))/2.
    sigma = 1.0
    return np.log(np.sqrt(2*np.pi)) + np.log(sigma) + self.reg_loss_func_L1(mean_pred, gt)/(sigma*sigma)
    # return np.log(np.sqrt(2*np.pi)) + torch.log(sigma) + self.reg_loss_func_L1(mean_pred, gt)/(2*sigma)


  def get_mdn_loss(self, eps: float, gt: torch.Tensor, 
                         mean_pred: torch.Tensor, std_pred: torch.Tensor, norm_factor=1.0) -> torch.Tensor :
    
    # sigma_x = torch.add(self.elu(std_pred), 1+eps)  
    sigma = torch.add(self.elu(std_pred), 1+eps)
    # sigma   = torch.exp(std_pred)
    # sigma   = torch.exp(std_pred)
    m_x     = torch.distributions.Normal(loc=mean_pred/norm_factor, scale=sigma/norm_factor)
    # with same prob, we want the simma more lower
    loss_x  = -1 * (m_x.log_prob(gt/norm_factor))# + torch.log(sigma)
    return loss_x

  def forward(self, cls_weight: float, pred_buffer: torch.Tensor, 
                    gt: torch.Tensor, gt_pose: torch.Tensor, feat_vec: torch.Tensor, ep: float) -> torch.Tensor :
    """ current use angle distance to get closest trajectory """
    # resample
    bat_sz = gt.shape[0]
    gt_traj = gt.reshape(bat_sz, 1, 33, 3).expand(-1, self.traj_num, 33, 3)  # B,1,3->B,M,3

    cls_sz = self.traj_num
    traj_sz = self.traj_len*self.traj_num*3
    pose_sz = 6

    plan_cls = pred_buffer[:, 0:cls_sz]
    pred_traj = pred_buffer[:, cls_sz:cls_sz+traj_sz].reshape(-1,self.traj_num, self.traj_len, 3)
    pred_traj_std = pred_buffer[:, cls_sz+traj_sz:cls_sz+traj_sz+traj_sz].reshape(-1,self.traj_num, self.traj_len, 3)

    pred_pose = pred_buffer[:, cls_sz+traj_sz+traj_sz:cls_sz+traj_sz+traj_sz+pose_sz]
    pred_pose_std = pred_buffer[:, cls_sz+traj_sz+traj_sz+pose_sz:cls_sz+traj_sz+traj_sz+pose_sz+pose_sz]

    m_loss_x_L1 = self.reg_loss_func_L1( pred_traj[:,:,:, 0], gt_traj[:, :,:, 0]).mean(dim=2)
    m_loss_y_L1 = self.reg_loss_func_L1( pred_traj[:,:,:, 1], gt_traj[:, :,:, 1]).mean(dim=2)
    # m_loss_z_L1 = self.reg_loss_func_L1( pred_traj[:,:,:, 2], gt_traj[:, :,:, 2]).mean(dim=2)

    m_loss_vx = self.reg_loss_func_L1( pred_pose[:, 0], gt_pose[:,0]).mean()
    m_loss_wz = self.reg_loss_func_L1( pred_pose[:, 5], gt_pose[:,5]).mean()


    ## -------> caculate regression loss  
    vx_loss = self.kl_reg_loss(1e-6, gt_pose[:,0], pred_pose[:, 0], pred_pose_std[:, 0], ep).mean()
    # vy_loss = self.kl_reg_loss(1e-6, gt_pose[:,1], pred_pose[:, 1], pred_pose_std[:, 1]).mean()
    # vz_loss = self.kl_reg_loss(1e-6, gt_pose[:,2], pred_pose[:, 2], pred_pose_std[:, 2]).mean()
    # wx_loss = self.kl_reg_loss(1e-6, gt_pose[:,3], pred_pose[:, 3], pred_pose_std[:, 3]).mean()
    # wy_loss = self.kl_reg_loss(1e-6, gt_pose[:,4], pred_pose[:, 4], pred_pose_std[:, 4]).mean()
    w_wz = 0 if ep > 3  else 1 # closeloop not training wz loss
    wz_loss = w_wz * self.kl_reg_loss(1e-6, gt_pose[:,5], pred_pose[:, 5], pred_pose_std[:, 5], ep).mean()

    # pose_loss = vx_loss + vy_loss + vz_loss + wx_loss + wy_loss + wz_loss
    # wz is not correct now, so we not train it!!!
    pose_loss = vx_loss + wz_loss

    ## -------> caculate distance loss
    # loss_x  = self.get_mdn_loss(1e-3, gt_traj[:,:,:, 0], pred_traj[:, :,:, 0], pred_traj[:, :,:, 3])
    loss_x = self.kl_reg_loss(1e-6, gt_traj[:,:,:, 0], pred_traj[:, :,:, 0], pred_traj_std[:, :,:, 0], ep ).mean(dim=2)
    loss_y = self.kl_reg_loss(1e-6, gt_traj[:,:,:, 1], pred_traj[:, :,:, 1], pred_traj_std[:, :,:, 1], ep ).mean(dim=2)
    loss_z = self.kl_reg_loss(1e-6, gt_traj[:,:,:, 2], pred_traj[:, :,:, 2], pred_traj_std[:, :,:, 2], ep ).mean(dim=2)


    ## -------> add angle loss
		# delta cost
    gt_dx = torch.clip(gt_traj[:,:,1:33,0] - gt_traj[:,:,0:32,0], min=0.001)
    gt_dy = gt_traj[:,:,1:33,1] - gt_traj[:,:,0:32,1]
    gt_theta = torch.atan2(gt_dy, gt_dx) * (180/np.pi) # deg

    DT = torch.Tensor(ANCHOR_TIME[1:33] - ANCHOR_TIME[0:32]).expand([gt_theta.shape[0], 5, 32]).to(gt_theta.device)
    gt_ds = (gt_dx.pow(2)   + gt_dy.pow(2) ).sqrt() / DT

    pred_dx = pred_traj[:,:,1:33,0] - pred_traj[:,:,0:32,0]
    pred_dy = pred_traj[:,:,1:33,1] - pred_traj[:,:,0:32,1]
    pred_theta = torch.atan2(pred_dy, pred_dx) * (180/np.pi) # deg

    pred_ds = (pred_dx.pow(2) + pred_dy.pow(2) ).sqrt() / DT

    loss_delta_theta = self.reg_loss_func_L1(pred_theta, gt_theta).mean(dim=(2)) # averge point angle loss
    loss_delta_ds = self.reg_loss_func_L1(pred_ds, gt_ds).mean(dim=(2)) # averge point angle loss

    ## add y-loss, most care about!
    traj_loss  = (loss_x + 5*loss_y + 1*loss_z)  + loss_delta_theta + loss_delta_ds
    ## -------> find best trajectory index, use angle minimum or distance minimun
    if self.is_use_angle_dist:
      angle_dist = 1 - self.distance_func(pred_traj[:,:,-1,0:2], gt_traj[:,:,-1,0:2])  # B, M
      index = angle_dist.argmin(dim=1)
      # index = loss_delta_theta.argmin(dim=1)
    else:
      index = traj_loss.argmin(dim=1)

    ## -------> caculate best regression trajectory loss
    reg_loss_best = traj_loss[torch.tensor(range(len(index)), device=index.device), index, ...].mean()
    ## -------> caculate best regression class loss
    gt_cls = index
    cls_loss  = self.cls_loss_func(plan_cls, gt_cls) 
    # we not traing class, we random train every at early  cls_weight
    total_loss = pose_loss + reg_loss_best + cls_weight*cls_loss#*(min(self.count/300000, 1.0)) #*(min(self.count/10000, 1.0)) # class loss warm start
    self.count += 1

    ## -------> send best trajectory for closed loop online control
    best_traj = pred_traj[torch.tensor(range(len(index)), device=index.device), index, ...]

    ## return
    return (total_loss, reg_loss_best, cls_loss, 
            m_loss_x_L1[torch.tensor(range(len(index)), device=index.device), index, ...].mean(), 
            m_loss_y_L1[torch.tensor(range(len(index)), device=index.device), index, ...].mean(), 
            loss_delta_theta[torch.tensor(range(len(index)), device=index.device), index, ...].mean() , m_loss_vx, m_loss_wz), best_traj