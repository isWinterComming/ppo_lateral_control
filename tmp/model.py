import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


# ==================== 2. 策略网络 ====================
class ResBlock1D(nn.Module):
    def __init__(self, in_size, res_size, drop_cof=0.1):
        super(ResBlock1D, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_size, res_size),
            nn.Dropout(drop_cof),
            nn.Tanh(),
            nn.Linear(res_size, in_size),
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.block(x))

# 策略网络（Actor-Critic架构）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # # 为每个原始特征创建embedding
        # self.dymiac_embeddings =  nn.Sequential(
        #         nn.Linear(3, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 32),
        #     )
        # self.refline_embeddings =  nn.Sequential(
        #         nn.Linear(4, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 32),
        #     )

        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )

        # 演员网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(3, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim * 1),  # 输出均值和标准差
        )

        # 评论家网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(3, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(1.0)), requires_grad=True)
        # 应用初始化
        # self.apply(self._init_weights)
        self.std_factor = 1.0
        self.scalar_factor = 10.0

    def set_std_factor(self, std_factor = 1.0):
        self.std_factor = std_factor

    def set_scalar_factor(self, scalar_factor = 1.0):
        self.scalar_factor = scalar_factor

    def _init_weights(self, module):
        """统一的权重初始化函数"""
        # 对于线性层
        if isinstance(module, nn.Linear):
            # Xavier/Glorot 初始化，适合Tanh激活
            gain = nn.init.calculate_gain('tanh')

            if module is self.actor:
                # 策略头：较小的初始化，防止初始动作太大
                # nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.xavier_uniform_(module.weight, gain=gain*0.1)
                nn.init.constant_(module.bias, 0.0)

            elif module is self.critic:
                # 价值头：较小的初始化，防止初始价值估计太大
                # nn.init.orthogonal_(module.weight, gain=0.3)
                nn.init.xavier_uniform_(module.weight, gain=gain*0.2)
                nn.init.constant_(module.bias, 0.0)

            else:
                # 共享层：标准初始化
                # nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        # features = self.shared_layers(state[:, 0:6])
        features = state[:, 0:3]

        # Actor输出
        action_mean = self.actor(features)
        # action_mean, action_logstd = actor_output.chunk(2, dim=-1)
        # action_std = action_logstd.sigmoid()

        # Critic输出
        value = self.critic(features)
        # action_std = self.log_std.exp().clamp(0.5, 2.5)
        return self.scalar_factor * action_mean.tanh(),  self.std_factor, value

    def act(self, state):
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # print(action_std)
        return log_prob, entropy, value