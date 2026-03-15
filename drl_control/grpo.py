import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np
from drving_simulator_v2 import DrivingSimulator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import random
import time
import os


idx = 21
np.random.seed(idx)
random.seed(idx)
torch.manual_seed(idx)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.states = []
        self.actions = []
        self.log_probs = []
        # self.values = []
        self.rewards = []
        # self.dones = []
        self.capacity = capacity
        self.advantages = []
        # self.returns = []

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        # self.values.append(value)
        self.rewards.append(reward)
        # self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        # self.values.clear()
        self.rewards.clear()
        # self.dones.clear()
        self.advantages.clear()
        # self.returns.clear()

    def __len__(self):
        return len(self.states)


class PI_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound) -> None:
        super().__init__()
        # (self.lower_bound, self.upper_bound) = (
        #     torch.tensor(lower_bound, dtype=torch.float32),
        #     torch.tensor(upper_bound, dtype=torch.float32),
        # )
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        action = F.tanh(self.fc3(y))

        action = (action + 1) * (
            self.upper_bound - self.lower_bound
        ) / 2 + self.lower_bound

        return action


class V_Network(nn.Module):
    def __init__(self, obs_dim) -> None:
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        values = self.fc3(y)

        return values


class PPOPolicy(nn.Module):
    def __init__(
        self,
        learning_rate=1e-4,
        clip_range=0.2,
        value_coeff=0.5,
        obs_dim=4,
        action_dim=1,
        initial_std=2.0,
        max_grad_norm=0.5,
        device="cpu",
    ):
        super().__init__()
        self.low_bound = -10
        self.up_bound = 10
        self.pi_network = PI_Network(obs_dim, 1, self.low_bound, self.up_bound).to(device)
        self.v_network = V_Network(obs_dim).to(device)
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.counter = 0

        # Gaussian policy with learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(self.action_dim, device=device)
            * torch.log(torch.tensor(initial_std)),
            requires_grad=True,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(2048 * 8)

    def forward(self, obs):
        pi_out = self.pi_network(obs)
        dist_out = Normal(pi_out, torch.exp(self.log_std))
        v_out = self.v_network(obs)
        return dist_out, v_out

    @torch.no_grad()
    def get_action(self, obs):
        """Sample action based on current policy"""
        obs_torch = torch.FloatTensor(obs).to(self.device)
        dist, values = self.forward(obs_torch)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return (action.cpu().item(), log_prob.cpu().item(), values.cpu().item())

    @torch.no_grad()
    def get_values(self, obs):
        """Return value of the state"""
        obs_torch = torch.FloatTensor(obs).to(self.device)
        _, values = self.forward(obs_torch)
        return values.cpu().item()

    def evaluate_action(self, obs_batch, action_batch):
        """Evaluate taken action."""
        dist, values = self.forward(obs_batch)
        log_prob = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)
        return log_prob, values

    def update(
        self, obs_batch, action_batch, log_prob_batch, advantage_batch):
        """
        Performs one step gradient update of policy and value network.

        Args:
            obs_batch: (batch_size, obs_dim)
            action_batch: (batch_size, action_dim)
            log_prob_batch: (batch_size, 1)
            advantage_batch: (batch_size, 1)
            return_batch: (batch_size, 1)
        """
        # Get new log probabilities and values
        new_log_prob, values = self.evaluate_action(obs_batch, action_batch)

        # PPO clipping objective
        ratio = torch.exp(new_log_prob - log_prob_batch)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        
        # print(ratio)

        # Value function loss
        # value_loss = self.value_coeff * torch.mean((values - return_batch) ** 2)
        kl_div = torch.mean(log_prob_batch - new_log_prob) * 0.0

        # Total loss
        total_loss = pi_loss + kl_div

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Calculate approximate KL divergence

        return {
            "pi_loss": pi_loss.item(),
            "value_loss": pi_loss.item(),
            "total_loss": total_loss.item(),
            "kl_divergence": kl_div.item(),
            "std_dev": torch.exp(self.log_std).cpu().detach().numpy(),
            "mean_ratio": ratio.mean().item(),
            "clip_fraction": (
                (ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)
            )
            .float()
            .mean()
            .item(),
        }

    def save(self, filepath):
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "log_std": self.log_std,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def compute_return_advantage(
        self, rewards, values, last_value, is_last_terminal=True, gamma=0.99, gae_lambda=0.95
    ):
        """
        Computes returns and advantage based on generalized advantage estimation.
        """
        N = len(rewards)
        advantages = np.zeros((N, 1), dtype=np.float32)

        tmp = 0.0
        for k in reversed(range(N)):
            if k == N - 1:
                next_non_terminal = 1 - is_last_terminal
                next_values = last_value
            else:
                next_non_terminal = 1
                next_values = values[k + 1]

            delta = rewards[k] + gamma * next_non_terminal * next_values - values[k]
            tmp = delta + gamma * gae_lambda * next_non_terminal * tmp

            advantages[k] = tmp
            
        returns = advantages + np.array(values).reshape((N,1))
        return advantages.tolist(), returns.tolist()

    def roll_once(self, idx):
        # 为每个进程设置独特的随机种子
        
        group_num = 8
        group_epside_rewards = []
        group_epside_values = []
        group_epside_dones = []
        group_epside_action = []
        group_epside_log_prob = []
        group_epside_state = []
        group_epside_advantage = []
        group_returns = []
        group_lengths = []

        env = DrivingSimulator()
        _, obs_init = env.reset()
        for k in range(group_num):

            obs,_ = env.reset_state(obs_init[0], obs_init[1], obs_init[2], obs_init[3], obs_init[4])
            one_epside_rewards = []
            one_epside_values = []
            one_epside_dones = []
            one_epside_action = []
            one_epside_log_prob = []
            one_epside_state = []
            while True:
                # 选择动作
                # print('obs_base---', obs, k)
                action, log_prob, values = self.get_action(obs)
                
                clipped_action = np.clip(action, self.low_bound, self.up_bound)
                next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
                done = terminated or truncated
                # print(done)

                one_epside_rewards.append(reward)
                one_epside_values.append(values)
                one_epside_dones.append(done)
                one_epside_log_prob.append(log_prob)
                one_epside_action.append(action)
                one_epside_state += obs.tolist()

                # 更新状态
                obs = next_obs
                if done > 0:
                    break
                # print('obs_base---', obs, k)
            group_lengths.append(len(one_epside_rewards))
                
            gama = 1.0
            ep_returns = sum([one_epside_rewards[i] * gama ** i for i in range(len(one_epside_rewards))])
            group_returns.append(ep_returns)
            group_epside_rewards.extend(one_epside_rewards)
            group_epside_values.extend(one_epside_values)
            group_epside_dones.extend(one_epside_dones)
            group_epside_log_prob.extend(one_epside_log_prob)
            group_epside_action.extend(one_epside_action)
            group_epside_state.extend(one_epside_state)
            
        group_returns = np.array(group_returns)
        group_mean = np.mean(group_returns)
        group_std = np.std(group_returns) + 1e-8
        group_advantages = ((group_returns - group_mean) / group_std).tolist()
        for adv, ep_length in zip(group_advantages, group_lengths):
            group_epside_advantage.extend([adv for i in range(ep_length)])
            
        # print(group_returns)
        # print(len(one_epside_values), time.time() - st)
        return (
            group_epside_action,
            group_epside_log_prob,
            group_epside_rewards,
            group_epside_state,
            group_epside_advantage,
            group_mean
        )

    def get_batch_data(self, max_workers=16):
        
        ep_reward = 0.0
        ep_rolls = 0

        idx = 32
        while len(self.buffer.actions) < self.buffer.capacity:


            res = self.roll_once(idx)
            self.buffer.actions += res[0]
            self.buffer.log_probs += res[1]
            self.buffer.rewards += res[2]
            self.buffer.states += res[3]
            self.buffer.advantages += res[4]
            ep_reward += res[5]
            ep_rolls += 1

            idx += 1

        return ep_reward / (ep_rolls)

    def training_policy(self, Season=1):

        # get batch data.
        mean_reward = self.get_batch_data()

        # self.pi_network.train()
        # self.v_network.train()

        # 准备数据
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(
            self.device
        )
        old_log_probs = torch.tensor(
            np.array(self.buffer.log_probs), dtype=torch.float32
        ).to(self.device)

        advantages = torch.tensor(
            np.array(self.buffer.advantages), dtype=torch.float32
        ).to(self.device)


        # 多轮PPO更新
        pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
        batch_size = 128
        for _ in range(10):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            # 小批量训练
            for start_idx in range(0, len(states), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) < batch_size:
                    continue

                # 获取批次数据
                obs_batch = states[batch_indices].view(-1,self.obs_dim)
                action_batch = actions[batch_indices].view(-1,1)
                log_prob_batch = old_log_probs[batch_indices].view(-1,1)
                advantage_batch = advantages[batch_indices].view(-1,1)
                
                # # print(advantage_batch.shape)
                # advantage_batch = (
                #     advantage_batch -
                #     advantage_batch.mean()
                # ) / (advantage_batch.std() + 1e-8)
                
                # Update the networks on minibatch of data
                ret = self.update(
                    obs_batch,
                    action_batch,
                    log_prob_batch,
                    advantage_batch,
                )
                pi_losses.append(ret["pi_loss"])
                v_losses.append(ret["value_loss"])
                total_losses.append(ret["total_loss"])
                approx_kls.append(ret["kl_divergence"])
                stds.append(ret["std_dev"])

        print(
            f"Season={Season} --> mean_ep_reward={mean_reward}, pi_loss={np.mean(pi_losses)}, v_loss={np.mean(v_losses)}, total_loss={np.mean(total_losses)}, approx_kl={np.mean(approx_kls)}, avg_std={np.mean(stds)}"
        )

        self.buffer.clear()


pp = PPOPolicy()


for i in range(500):
    pp.training_policy(i)