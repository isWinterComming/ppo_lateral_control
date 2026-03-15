# import torch
# import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# 设置字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. 修正流 (Rectified Flow) 核心 ====================





import tinygrad.nn as nn
import math
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters


def clip_grad_norm_where(parameters, max_norm):
    """
    使用where实现基于范数的梯度裁剪
    """
    # 计算总范数
    total_norm = Tensor.zeros(1, requires_grad=False)
    for param in parameters:
        if param.grad is not None:
            total_norm += (param.grad * param.grad).sum()
    
    total_norm = total_norm.sqrt()
    
    # 使用where进行裁剪
    scale = max_norm / total_norm
    for param in parameters:
        if param.grad is not None:
            param.grad = Tensor.where(
                total_norm > max_norm,
                param.grad * scale,
                param.grad
            )
    
    return total_norm


class RectifiedFlow:
    """
    修正流 (Rectified Flow) 核心实现
    
    原理: 在先验分布 p0 和数据分布 p1 之间构建线性路径
    x_t = (1-t) * x_0 + t * x_1, t ∈ [0,1]
    
    学习目标: 预测速度场 v(x_t, t) ≈ (x_1 - x_0)
    """
    
    def __init__(self, num_timesteps=100):
        self.num_timesteps = num_timesteps
    
    @staticmethod
    def compute_linear_path(x_0, x_1, t):
        """
        计算线性插值路径
        
        Args:
            x_0: [B, D] 先验样本（高斯噪声）
            x_1: [B, D] 真实轨迹
            t: [B] 或标量，时间步 [0,1]
        
        Returns:
            x_t: [B, D] 插值后的样本
            target_v: [B, D] 目标速度 (x_1 - x_0)
        """
        if isinstance(t, (int, float)):
            t = Tensor.ones(x_0.shape[0] ) * t
            
        t = t.view(-1, 1)  # [B, 1]
        
        # 线性插值: x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # 目标速度: dx/dt = x_1 - x_0 (常数速度)
        target_v = x_1 - x_0
        
        return x_t, target_v
    
    def sample_timesteps(self, batch_size):
        """采样时间步 t ~ Uniform(0,1)"""
        return Tensor.rand(batch_size, 1)
    
    def sample_prior(self, batch_size, dim):
        """从先验分布采样（标准高斯）"""
        return Tensor.randn(batch_size, dim)
    

# ==================== 2. 条件速度场网络 ====================
class SinusoidalPositionEmbeddings:
    """时间步 t 的正弦位置编码 - tinygrad版本"""
    
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, time: Tensor) -> Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = (Tensor.arange(half_dim) * -embeddings).exp()
        embeddings = time * embeddings
        embeddings = Tensor.cat(embeddings.sin(), embeddings.cos(), dim=1)
        return embeddings

class VelocityFieldNetwork:
    """
    速度场网络 v_θ(x_t, t, condition) - tinygrad版本
    """
    
    def __init__(self,
                 trajectory_dim=60,    # T*2
                 condition_dim=60,
                 hidden_dim=512):
        
        self.trajectory_dim = trajectory_dim
        
        # 时间步编码
        self.time_sinusoidal = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 条件编码
        self.cond_linear1 = nn.Linear(condition_dim, hidden_dim)
        self.cond_linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 轨迹编码
        self.traj_linear1 = nn.Linear(trajectory_dim, hidden_dim)
        self.traj_linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 主网络
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, trajectory_dim)
        
    def __call__(self, x_t: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        """
        Args:
            x_t: [B, D] 插值轨迹
            t: [B, 1] 时间步
            condition: [B, D_cond] 车道线条件
        """
        # 1. 时间步编码
 
        t_emb = self.time_sinusoidal(t)              # [B, hidden_dim]
        t_emb = t_emb.silu()
        t_emb = self.time_linear1(t_emb)
        t_emb = t_emb.silu()
        t_emb = self.time_linear2(t_emb)             # [B, hidden_dim]
        
        # 2. 条件编码
        cond_emb = self.cond_linear1(condition)      # [B, hidden_dim]
        cond_emb = cond_emb.silu()
        cond_emb = self.cond_linear2(cond_emb)       # [B, hidden_dim]
        
        # 3. 轨迹编码
        traj_emb = self.traj_linear1(x_t)            # [B, hidden_dim]
        traj_emb = traj_emb.silu()
        traj_emb = self.traj_linear2(traj_emb)       # [B, hidden_dim]
        
        # 4. 融合
        # print(traj_emb.shape, t_emb.shape, cond_emb.shape)
        fused = Tensor.cat(traj_emb, t_emb, cond_emb, dim=1)  # [B, hidden_dim*3]
        
        # 5. 主网络
        h = self.fc1(fused)
        h = h.silu()
        h = h.dropout(0.1)
        
        h = self.fc2(h)
        h = h.silu()
        h = h.dropout(0.1)
        
        h = self.fc3(h)
        h = h.silu()
        
        v_pred = self.fc4(h)  # [B, trajectory_dim]
        
        return v_pred

class FlowMatchingTrainer:
    """Flow Matching 训练器"""
    
    def __init__(self, model, flow, device, lr=1e-4):
        self.model = model
        self.flow = flow
        self.device = device

        params = []
        params.extend(get_parameters(self.model))
        print('model parameters size: ', len(params))
        # params.extend(get_parameters(self.flow))
        
        print(len(params))
        self.optimizer = nn.optim.AdamW(params, lr, weight_decay=1e-5)
        self.params = params
        
    @TinyJit
    @Tensor.train()  
    def train_step(self, true_traj, condition):
        # 从先验采样
        x_0 = self.flow.sample_prior(true_traj.shape[0], true_traj.shape[1])#.to(self.device)
        x_1 = true_traj
        
        # 采样时间步
        t = self.flow.sample_timesteps(true_traj.shape[0])#.to(self.device)
        
        # print(x_0.shape, x_1.shape, condition.shape)
        
        # 计算线性路径
        x_t, target_v = RectifiedFlow.compute_linear_path(x_0, x_1, t)
        
        # 预测速度场
        v_pred = self.model(x_t, t, condition)
        
        # Flow Matching 损失: L = ||v_θ(x_t, t, c) - (x_1 - x_0)||^2
        loss = (v_pred -  target_v).square().mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_where(self.params, 1.0)
        self.optimizer.step()
        
        return loss.realize()
        
    def train_epoch(self, dataloader, batch_size):
        """训练一个epoch"""
        Tensor.training = True
        total_loss = 0
        
        for batch in dataloader:
            # 真实轨迹和条件
            true_traj = Tensor(batch['trajectory'].numpy().astype(np.float32)).view(-1,60)#.to(self.device).view(-1,60)  # [B, D]
            condition = Tensor(batch['condition'].numpy().astype(np.float32)).view(-1,60)#.to(self.device).view(-1,60)   # [B, D]
            if condition.shape[0] != batch_size:
                continue
            loss = self.train_step(true_traj, condition)
            
            total_loss += loss.item()
        
        # self.scheduler.step()
        return total_loss / len(dataloader)
    
    def sample(self, condition, num_steps=10, method='euler'):
        """
        采样生成轨迹
        
        Args:
            condition: [B, D] 车道线条件
            num_steps: 采样步数 (Flow Matching 只需1-10步)
            method: 'euler' 或 'rk4'
        """

        condition = Tensor(condition.astype(np.float32)).view(-1,60)
        Tensor.training = False
        print(condition.shape)
        B = condition.shape[0]
        D = condition.shape[1]
        
        # 从先验开始
        x = self.flow.sample_prior(B, D)#.to(self.device)
        
        # 时间步长
        dt = 1.0 / num_steps
        
        # ODE求解
        for i in range(num_steps):
            t = Tensor.ones(B,1) * (i * dt)
            
            if method == 'euler':
                # 欧拉法
                v = self.model(x, t, condition)
                x = x + v * dt
                
            elif method == 'rk4':
                # 四阶龙格-库塔法
                t_mid = t + dt/2
                
                k1 = self.model(x, t, condition)
                k2 = self.model(x + k1 * dt/2, t_mid, condition)
                k3 = self.model(x + k2 * dt/2, t_mid, condition)
                k4 = self.model(x + k3 * dt, t + dt, condition)
                
                x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
        return x
    
class LaneKeepingDataset(Dataset):
    """车道保持数据集 - 条件和轨迹配对"""
    def __init__(self, num_samples=10000, trajectory_length=30):
        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        samples = []
        for _ in range(self.num_samples):
            # 随机生成道路形状参数
            curvature = np.random.uniform(-0.01, 0.01)  # 曲率
            start_offset = np.random.uniform(-3.0, 3.0)  # 起始偏移
            start_heading = np.random.uniform(-0.1, 0.1)  # 起始偏移
            road_width = np.random.uniform(3.0, 3.5)  # 车道宽
            
            # 生成车道中心线 (条件)
            lane_center = np.zeros((self.trajectory_length, 2))
            for i in range(self.trajectory_length):
                # 使用二次曲线模拟车道
                lane_center[i,0] = start_offset + start_heading * i + curvature * (i ** 2) / 2.
                lane_center[i,1] = 10 * (start_heading + curvature * (i ) / 1. )
            
            # 生成真实轨迹 (人类驾驶轨迹)
            # 在车道中心线附近加一些自然波动
            f_idx = np.random.randint(15, 25)
            # use pp to generate gt path.
            control_kappa = 2 * lane_center[f_idx, 0] /  ((f_idx)**2 + lane_center[f_idx, 0]**2)
            trajectory = np.zeros((self.trajectory_length, 2))
            for i in range(self.trajectory_length):
                
                if f_idx > i:
                    trajectory[i,0] = control_kappa * (i ** 2) / 2.
                    trajectory[i,1] = 10 * control_kappa * (i) / 1.
                else:
                    trajectory[i,0] = lane_center[i,0]
                    trajectory[i,1] = 10 * curvature * (i) / 1.

            # print(lane_center, trajectory, control_kappa)
            
            # 添加平滑的随机波动 (模拟驾驶习惯)
            smooth_noise = self._generate_smooth_noise(self.trajectory_length)
            # trajectory[:,0] += smooth_noise * 0.1
            # trajectory[:,1] += smooth_noise * 0.2
            
            # 确保轨迹在车道内
            max_deviation = road_width / 1
            trajectory = np.clip(trajectory, 
                                lane_center - max_deviation,
                                lane_center + max_deviation)
            # Flatten
            lane_center = lane_center.flatten()
            trajectory = trajectory.flatten()
            # 转换为tensor
            samples.append({
                'condition': lane_center,#.unsqueeze(-1),  # [T, 1]
                'trajectory': trajectory,#.unsqueeze(-1),  # [T, 1]
                'curvature': curvature,
                'road_width': road_width
            })
        
        return samples
    
    def _generate_smooth_noise(self, length):
        """生成平滑的噪声序列"""
        noise = np.cumsum(np.random.normal(0, 0.05, length))
        noise = noise - np.mean(noise)  # 零均值
        return noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]
    

def main():
    """训练Flow Matching车道保持规划器"""
    
    print("=" * 60)
    print("Flow Matching 车道保持规划器")
    print("=" * 60)
    
    # 参数
    trajectory_length = 30
    input_dim = trajectory_length * 2
    batch_size = 64
    num_epochs = 100
    device = 'CL'
    
    # 数据集
    print(" start make dataset.")
    dataset = LaneKeepingDataset(num_samples=10000, trajectory_length=trajectory_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    print(" start make model.")
    # 可选: VelocityFieldNetwork 或 UNetVelocityField
    model = VelocityFieldNetwork(
        trajectory_dim=input_dim,
        condition_dim=input_dim,
        hidden_dim=512,    )
    
    # 创建Flow
    flow = RectifiedFlow(num_timesteps=100)
    
    # 创建训练器
    trainer = FlowMatchingTrainer(model, flow, device, lr=1e-4)
    
    # 训练
    print("\n开始训练...")
    losses = []
    
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(train_loader, batch_size)
        losses.append(loss)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Flow Matching Training Loss')
    plt.grid(True)
    plt.show()
    
    # 测试采样
    print("\n测试采样...")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    sample_idx = np.random.randint(0,100, 6).tolist()
    
    for idx, value in enumerate(sample_idx):
        print(idx, value)
        # 不同步数的采样
        condition = dataset[value]['condition']
        ax = axes[idx]
        for steps in [1, 5, 10]:
            traj = trainer.sample(condition, num_steps=steps)
            
            # 恢复为轨迹点
            traj_points = traj.numpy().reshape(-1, 2)
            lane_points = condition.reshape(-1, 2)
            
            ax.plot(traj_points[:, 0], 'o--', label=f'{steps} steps')
        
        ax.plot(lane_points[:, 0], 'g--', label='车道线', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Flow Matching 采样结果')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    plt.show()
main()