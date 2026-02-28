import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pylab as plt
from drving_simulator import DrivingSimulator
from model import ActorCritic
from scipy import stats
from lateral_mpc_lib.lat_mpc import LatMpc


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.states = []
        self.actions = []
        self.actions_gt = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.capacity = capacity
        self.advantages = []
        self.returns = []

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.actions_gt.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.actions_gt.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self):
        return len(self.states)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化策略网络
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
        # self.optimizer = optim.Adam([
        #     {'params': [p for n, p in self.policy.named_parameters()
        #                 if 'log_std' not in n]},  # 其他参数有weight_decay
        #     {'params': [self.policy.log_std], 'weight_decay': 0.0}  # log_std无weight_decay
        # ], lr=config['learning_rate'], weight_decay=1e-2)
        # self.optimizer = torch.optim.Adam([
        #         {'params': self.policy.actor.parameters(), 'lr': 1e-4},
        #         {'params': self.policy.critic.parameters(), 'lr': 1e-4},
        #         {'params': self.policy.shared_layers.parameters(), 'lr': 2e-4},
        #         {'params': [self.policy.log_std], 'lr': 5e-4, }  # 3倍学习率
        #     ], weight_decay=1e-2)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=config['learning_rate'],  momentum=0.9, weight_decay=1e-2)
        # 初始化缓冲区
        self.buffer = ReplayBuffer(config['buffer_capacity'])

        # 跟踪训练统计信息
        self.episode_rewards = []
        self.episode_lengths = []

    def compute_advantages(self, last_value, rewards, values, dones, gamma=0.99, gae_lambda=0.85):
        """
        Computes returns and advantage based on generalized advantage estimation.
        """
        N = len(rewards)
        advantages = np.zeros(
            (N, 1),
            dtype=np.float32
        )
        tmp = 0.0
        for k in reversed(range(N)):
            if k==N-1:
                next_non_terminal = 0 if dones[k] in [1, 2] else 1
                next_values =  0.0 if dones[k] in [1, 2] else last_value
            else:
                next_non_terminal = 1
                next_values = values[k+1]

            delta = rewards[k] + gamma * next_non_terminal * next_values - values[k]
            tmp = delta + gamma * gae_lambda * next_non_terminal * tmp

            advantages[k] = tmp
        returns = advantages +  np.array(values).reshape((N,1))

        return advantages.tolist(), returns.tolist()

    # def compute_advantages(self, last_value, rewards, values, dones, gamma=0.95):
    #     """
    #     更加清晰的蒙特卡洛实现，明确处理终止状态
    #     """
    #     returns = []
    #     discounted_reward = last_value
    #     for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
    #         if is_terminal in [1,2]:
    #             discounted_reward = 0
    #         discounted_reward = reward + (gamma * discounted_reward)
    #         returns.insert(0, discounted_reward)

    #     # Normalizing the rewards
    #     returns = torch.tensor(returns, dtype=torch.float32)
    #     # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    #     advantages = returns - torch.tensor(values, dtype=torch.float32)

    #     if torch.isnan(advantages).any() or torch.isnan(returns).any():
    #         print('nan found')
    #     return advantages.cpu().numpy().tolist(), returns.cpu().numpy().tolist()

    def training_model(self, eposide):
        """
        PPO更新步骤
        """
        self.policy.train()

        # 准备数据
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        actions_gt = torch.tensor(np.array(self.buffer.actions_gt), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32).to(self.device)

        advantages = torch.tensor(np.array(self.buffer.advantages), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(self.buffer.returns), dtype=torch.float32).to(self.device)


        # 创建图形，使用4行1列的子图布局
        fig = plt.figure(figsize=(10, 16))

        # === 子图1：动作分布 ===
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.hist(actions.cpu().numpy(), bins=80, color='green', alpha=0.8, label='action distributions')
        ax1.set_xlabel('Action Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('action distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === 子图2：优势函数分布 ===
        ax2 = fig.add_subplot(4, 1, 2)
        ax2.hist(advantages.cpu().numpy(), bins=80, color='green', alpha=0.8, label='advantages distributions')
        ax2.set_xlabel('Advantage Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('adv distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # === 子图3：2D状态分布（用散点图） ===
        ax3 = fig.add_subplot(4, 1, 3)
        x = states[:, 1].cpu().numpy()
        y = states[:, 2].cpu().numpy()

        # 绘制2D直方图
        hb = ax3.hist2d(x, y, bins=80, cmap='Reds', alpha=0.8)
        ax3.set_xlabel('lateral pos error')
        ax3.set_ylabel('lateral vel error')
        ax3.set_title('state distribution heatmap')
        plt.colorbar(hb[3], ax=ax3, label='heat density')

        # === 子图4：状态分量的边缘分布 ===
        ax4 = fig.add_subplot(4, 1, 4)

        # 绘制两个分量的直方图（堆叠显示）
        ax4.hist(x, bins=80, color='blue', alpha=0.5, label='lateral pos distri', density=True)
        ax4.hist(y, bins=80, color='red', alpha=0.5, label='lateral vel distri', density=True)

        ax4.set_xlabel('State Value')
        ax4.set_ylabel('Density')
        ax4.set_title('state distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./dist/{eposide}_distribution.png', dpi=150, bbox_inches='tight')

        # plt.show()
        plt.clf()
        plt.cla()

        # 多轮PPO更新
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.config['ppo_epochs']):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            # 小批量训练
            for start_idx in range(0, len(states), self.config['batch_size']):
                end_idx = start_idx + self.config['batch_size']
                batch_indices = indices[start_idx:end_idx]

                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_actions_gt = actions_gt[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # batch norm at mini batch.
                if batch_advantages.std() > 1e-6:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-6)

                # 评估当前策略
                log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                # values = values.squeeze(-1)

                # 计算概率比
                ratios = torch.exp(log_probs - batch_old_log_probs)
                prob_regular = F.smooth_l1_loss(log_probs.exp(), batch_old_log_probs.exp())
                gt_loss = F.smooth_l1_loss(batch_actions, batch_actions_gt)
                # print(ratios)
                # print('ration max: ', ratios.max(), 'ration min: ', ratios.min(), 'ration std: ', ratios.std(), 'ration mean: ', ratios.mean())
                # 计算PPO损失（Clipped Surrogate Objective）
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.config['clip_epsilon_up'],
                                  1 + self.config['clip_epsilon_down']) * batch_advantages

                # print(torch.isnan(batch_advantages), batch_advantages, rewards[batch_indices])
                policy_loss = -torch.min(surr1, surr2).mean() + 0.1*prob_regular + gt_loss * 0.0

                # 价值函数损失
                value_loss = F.smooth_l1_loss(values, batch_returns)

                # 熵正则化
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (policy_loss
                       + self.config['value_coef'] * value_loss
                       + self.config['entropy_coef'] * entropy_loss)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(), max_norm=self.config['max_grad_norm'])
                self.optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                print('action std:', self.policy.std_factor)

        # 清空缓冲区
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }

    def save_model(self, path):
        """保存模型"""
        # torch.save({
        #     'policy_state_dict': self.policy.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'config': self.config
        # }, path)
        # print(f"Model saved to {path}")
        torch.save(self.policy, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

    def plot_training_progress(self):
        """绘制训练进度"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 奖励曲线
        axes[0].plot(self.episode_rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards')
        axes[0].grid(True)

        # 移动平均奖励
        window_size = 50
        moving_avg = np.convolve(self.episode_rewards,
                                np.ones(window_size)/window_size,
                                mode='valid')
        axes[1].plot(moving_avg)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Moving Avg Reward')
        axes[1].set_title(f'Moving Average Reward (window={window_size})')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        plt.clf()

    def get_rl_std_factor(self, eposide_num, warm_eposide=5000, max_std=2.0, start_std=0.4):
        MAX_EP = 30000
        if eposide_num < warm_eposide:
            return start_std + (max_std - start_std) * eposide_num / warm_eposide
        else:
            return max(1.0 - (eposide_num - warm_eposide)/MAX_EP, 0.25) * max_std

    def run(self, env_name="Pendulum-v1", num_episodes=1000):
        """
        训练主循环
        """
        state_dim = 9
        action_dim = 1

        print(f"Training PPO on {env_name}")
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

        valid_eposide_steps = 0

        # 训练循环
        for episode in range(num_episodes):
            # Reset eposide status info.
            env = DrivingSimulator()
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            one_epside_rewards = []
            one_epside_values = []
            one_epside_dones = []
            one_epside_action = []
            one_epside_action_gt = []
            one_epside_log_prob = []
            one_epside_state = []
            self.policy.eval()

            gt_lat_mpc = LatMpc()

            # Rollout with policy model.
            while True:
                # action sample
                with torch.no_grad():
                    state_tensor = torch.tensor(state.tolist(), dtype=torch.float32).to(self.device)
                    action, log_prob, value = self.policy.act(state_tensor)

                    state_np = state_tensor.cpu().numpy()
                    gt_action = (180/np.pi) * gt_lat_mpc.update(True, state_np[0][7], state_np[0][0], state_np[0][1],  state_np[0][2], 0, 0, 0.0, 0.0)

                # tensor to numpy
                action = action[0].cpu().detach().numpy()[0]
                log_prob = log_prob[0].cpu().detach().numpy()
                value = value[0].cpu().detach().numpy()[0]

                # env step
                next_state, reward, mission_status = env.step(action)
                done = mission_status
                # print(action)
                # print(next_state[0][1:3], episode, mission_status)

                # save reward
                one_epside_rewards.append(reward)
                one_epside_values.append(value)
                one_epside_dones.append(done)
                one_epside_log_prob.append(log_prob)
                one_epside_action.append(action)
                one_epside_action_gt.append(gt_action)
                one_epside_state += next_state.tolist()

                # update eposide state
                state = next_state
                episode_reward += reward
                episode_length += 1
                if done > 0:
                    break
            # Early contine with short eposide length, no meaningful buffer.
            if(len(one_epside_dones) < 15):
                continue

            # Store the eposide buffer status.
            self.buffer.actions += one_epside_action
            self.buffer.actions_gt += one_epside_action_gt
            self.buffer.log_probs += one_epside_log_prob
            self.buffer.dones += one_epside_dones
            self.buffer.values += one_epside_values
            self.buffer.rewards += one_epside_rewards
            self.buffer.states += one_epside_state

            # last state
            with torch.no_grad():
                state_tensor = torch.tensor(state.tolist(), dtype=torch.float32).to(self.device)
                _, _, value = self.policy.act(state_tensor)
                last_value = value[0].cpu().detach().numpy()[0]
            # print('action distribution: ', np.array(one_epside_action).mean(), np.array(one_epside_action).std())
            # print(one_epside_rewards)
            advantages, returns = self.compute_advantages(last_value, one_epside_rewards, one_epside_values, one_epside_dones)

            self.buffer.advantages += advantages
            self.buffer.returns += returns

            # Training policy model while the buffer is full.
            self.policy.set_std_factor(self.get_rl_std_factor(episode))
            # scalar_factor = max(min((episode / (2000)) , 1.0) * 10.0, 1.0)
            self.policy.set_scalar_factor( 10. )
            if len(self.buffer.rewards) >= self.config['buffer_capacity']:
                print('start training: ', len(self.buffer.rewards))
                # print(self.buffer.actions)
                losses = self.training_model(episode)
                print("loss:", losses, 'std_factor', self.policy.std_factor)

            # Record eposide reward and length info.
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Plot running status
            if (episode + 0) % 1 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"one_epside_action_mean: {np.array(one_epside_action).mean():.2f}, "
                      f"one_epside_action_std: {np.array(one_epside_action).std():.2f}, "
                      f"mission_status: {mission_status:.2f}, "
                      f"Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Length: {episode_length}")

            # Save model and eval model.
            if (valid_eposide_steps + 0) % 200 == 0:
                self.save_model(f'rl_model_baseling.pt')
                env.eval_model(self.policy, episode)

            valid_eposide_steps += 1



# 配置参数
config = {
    'learning_rate': 1e-4,
    'gamma': 0.95,
    'gae_lambda': 0.8,
    'clip_epsilon_up': 0.2,
    'clip_epsilon_down': 0.2,
    'ppo_epochs': 20,
    'batch_size': 512,
    'buffer_capacity': 4096,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
}

# 创建并训练智能体
if __name__ == "__main__":
    state_dim = 9
    action_dim = 1
    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim, config)

    # 训练智能体
    agent.run(env_name="Pendulum-v1", num_episodes=50000)

    # 绘制训练进度
    agent.plot_training_progress()

    # 保存模型
    agent.save_model("ppo_model.pth")