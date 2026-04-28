import math
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn

from diffusion_es import SinusoidalPosEmb
from diffusion_es import ParallelAttentionLayer
from diffusion_es import RotaryPositionEncoding


@dataclass
class FlowMathingConfig:
    # FM parameter
    infer_steps: int=10
    train_scale: float=0.1
    test_scale: float=0.1
    training: bool=True
    x_scale: float = 60
    y_scale: float =15

    # 多模态轨迹数量
    anchor_size: int=10

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0



class GoalFlowTrajModel(nn.Module):
    def __init__(self, config:FlowMathingConfig):
        super().__init__()

        self._config = config

        # usually, the BEV features are variable in size.
        self._downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._keyval_embedding = nn.Embedding(
            8**2, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(1, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self._config.tf_d_model),
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model)
        )
        self.sigma_proj_layer = nn.Linear(self._config.tf_d_model * 2, self._config.tf_d_model)

        self.trajectory_encoder = nn.Linear(3, self._config.tf_d_model)
        self.trajectory_time_embeddings = RotaryPositionEncoding(self._config.tf_d_model)
        self.type_embedding = nn.Embedding(3, self._config.tf_d_model) # trajectory, noise token

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self._config.tf_d_model, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True
            )
        for _ in range(1)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model, 3)
        )

    def forward(self, camera_feature, gt_trajs) -> Dict[str, torch.Tensor]:
        batch_size = camera_feature.shape[0]

        # camera_feature(bz,C,H,W)
        x = self._downscale(camera_feature)
        camera_feature = x.view(x.shape[0], x.shape[1], -1)

        # camera_feature(bz,C,H*W)
        camera_feature = camera_feature.permute(0, 2, 1)
        # 维度重排 camera_feature(bz,H*W,C)
        # 将gt_trajs移动到与status_feature 相同的设备和数据类型
        gt_trajs=gt_trajs.to(camera_feature)
        # print('gt_trajs shape:', gt_trajs.shape)
        dtype=camera_feature.dtype
        device=camera_feature.device

        # =================================== feature decoder ==================================================
        keyval = camera_feature + self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        # cross attention
        trajectory_query = self._tf_decoder(query, keyval)


        # =================================== flow ==================================================
        gt_trajs_=gt_trajs.clone()

        # 占位符张量，推理时为零张量
        target=torch.zeros_like(gt_trajs_)

        normal_trajs = self.normalize_xy(gt_trajs_, N=gt_trajs_.shape[-2]).to(gt_trajs_)
        print('normal_trajs shape:',normal_trajs.shape)
        if self._config.training:
            noise=torch.randn(size=(batch_size,3,33),device=normal_trajs.device,dtype=dtype).to(trajectory_query)*self._config.train_scale
        else:
            noise=torch.randn(size=(batch_size*self._config.anchor_size,3,33),dtype=dtype,device=device)*self._config.test_scale
        global_feature=self.encode_scene_features(trajectory_query)
        loss = 0

        # =================================== flow training ==================================================
        if self._config.training:
            batch_size = normal_trajs.shape[0]

            
            noisy_traj_points,t,target=get_train_tuple(z0=noise,z1=normal_trajs)

            timesteps=t*self._config.infer_steps

            pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,3)
            # print('pred shape:',pred.shape,'target shape:',target.shape)
            target = target.permute(0,2,1)

            loss_x = (pred[..., 0] - target[..., 0]).square().mean()
            loss_y = (pred[..., 1] - target[..., 1]).square().mean()
            loss_heading = (pred[..., 2] - target[..., 2]).square().mean()
            loss = loss_x + loss_y + loss_heading

        # =================================== flow sampling ==================================================
        else:
            trajs=noise

            features=global_feature[0].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,1,self._config.tf_d_model)
            embedding=global_feature[1].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,1,self._config.tf_d_model)
            global_feature=(features,embedding)
            timesteps=torch.arange(self._config.infer_steps).to(device)

            # ODE求解，欧拉法
            for t in timesteps:
                net_output_nonavi=self.denoise(trajs,t,global_feature)
                net_output=net_output_nonavi.reshape(self._config.anchor_size*batch_size,3,33)
                # print('net_output_nonavi',net_output_nonavi.shape, net_output.shape)
                trajs=trajs.detach().clone()+net_output*(1 / self._config.infer_steps)
        
            diffusion_output = self.denormalize_xy(trajs, N=gt_trajs.shape[-2])
            pred_trajs=diffusion_output.reshape(batch_size,self._config.anchor_size,-1,3)

            # ========================== trajectory scorer ========================================
            # TODO:selec the best trajectry
            pred=pred_trajs[:,:,:33,:].mean(1) 

        output: Dict[str, torch.Tensor] = {'trajectory':pred}        
        output.update({'target':target})
        output.update({'loss':loss})

        return output       
    
    def denoise(self, ego_trajectory, sigma, state_features):
        batch_size = ego_trajectory.shape[0]

        state_features, state_type_embedding = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],33,3)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        # 为轨迹点生成类型嵌入
        trajectory_type_embedding = self.type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,33,1)

        # Concatenate all features
        all_features = torch.cat([state_features, trajectory_features], dim=1)
        all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)

        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self._config.infer_steps
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self._config.tf_d_model)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        # Generate attention mask
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        dists = (indices[None] - indices[:,None]).abs()
        attn_mask = dists > 1       # TODO: magic number

        # Generate relative temporal embeddings
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size,1))

        # Global self-attentions
        for layer in self.global_attention_layers:            
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=all_type_embedding,
                attn_mask_11=attn_mask
            )

        trajectory_features = all_features[:,-33:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out 


    def denormalize_xy(self, trajectory, N=33):
        final_trajectory = trajectory.permute(0,2,1)
        
        final_trajectory = final_trajectory[:, :, :3]
        final_trajectory[:, :, 0] *= self._config.x_scale
        final_trajectory[:, :, 1] *= self._config.y_scale
        final_trajectory[:,:,2]=final_trajectory[:,:,2].tanh() * math.pi
        return final_trajectory   
    def normalize_xy(self, trajectory, N=33):
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 60
        y_scale = 15
        heading_scale = math.pi
        downsample_trajectory[:, :, 0] /= self._config.x_scale
        downsample_trajectory[:, :, 1] /= self._config.y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        downsample_trajectory[:,:,2]=downsample_trajectory[:,:,2]
        x = downsample_trajectory[:,:,2]
        eps = 1e-7
        x_clamped = torch.clamp(x, -0.999 + eps, 0.999 - eps)
        # 使用对数公式计算 atanh
        downsample_trajectory[:,:,2] = 0.5 * torch.log((1 + x_clamped) / (1 - x_clamped))

        
        
        trajectory = downsample_trajectory.permute(0,2,1)
        return trajectory
    
    def encode_scene_features(self, ego_agent_features):
        ego_features = ego_agent_features

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding
    
def get_train_tuple(z0=None, z1=None):
    t = torch.rand(z1.shape[0], 1, 1).to(z0.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0
    return z_t.float(), t.float(), target.float()



