# from asyncore import write
import os
import sys
import math
import pickle
# from tensorboard import SummaryWriter
import time
import gym
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import animation


import tinygrad.nn as nn
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters
import numpy as np
from PPOBuffer import PPOBuffer
from drving_simulator_v2 import DrivingSimulator

NUM_STEPS = 2048                    # Number of timesteps data to collect before updating
BATCH_SIZE = 64                     # Batch size of training data
TOTAL_TIMESTEPS = NUM_STEPS * 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # Lambda value for generalized advantage estimation
NUM_EPOCHS = 10                     # Number of epochs to train
LEARNING_RATE = 1e-4


class PI_Network():
    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound) -> None:
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        # self.fc4 = nn.Linear(128, action_dim)
        self.log_std = Tensor([math.log(1.5)], requires_grad=True)
        
    def __call__(self, obs):
        y = self.fc1(obs).tanh()
        y = self.fc2(y).tanh()
        action = self.fc3(y).tanh()
        action = (action + 1)*(self.upper_bound - self.lower_bound)/2 + self.lower_bound
        return action, self.log_std.exp()

class V_Network():
    def __init__(self, obs_dim) -> None:
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
       
    def __call__(self, obs):
        y = self.fc1(obs).tanh()
        y = self.fc2(y).tanh()
        values = self.fc3(y)
        return values
        

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

if __name__ == "__main__":
    # env = gym.make("Pendulum-v1")
    # obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # lower_bound = env.action_space.low
    # upper_bound = env.action_space.high
    env = DrivingSimulator()
    obs_dim = 4
    action_dim = 1
    lower_bound = -10.
    upper_bound = 10.
    
    train_test = "train" if len(sys.argv)==1 else sys.argv[1]
    if train_test=="train":
        # Setup tensorboard summary writer
        if "Log" not in os.listdir("./"):
            os.mkdir("Log")
        # summary_writer = SummaryWriter("Log")

        # Create networks
        pi_network = PI_Network(obs_dim, action_dim, lower_bound, upper_bound)
        v_network = V_Network(obs_dim)
        # pi_network.load_state_dict(torch.load('./saved_network/pi_network.pth') )
        # v_network.load_state_dict(torch.load('./saved_network/v_network.pth'))

        buffer = PPOBuffer(obs_dim, action_dim, NUM_STEPS)

        params = []
        params.extend(get_parameters(pi_network))
        params.extend(get_parameters(v_network))
        opt = nn.optim.Adam(params, LEARNING_RATE)

        @TinyJit
        def get_action_value(obs: Tensor) -> Tuple[Tensor, Tensor]:
            """Sample action based on current policy"""       
            pi_out, std_out = pi_network(obs)
            action = Tensor.randn() * std_out + pi_out
            v_out = v_network(obs)
            log_probs = -0.5 * Tensor(math.log(2 * math.pi)) - std_out.log() - 0.5 * ((action - pi_out) / std_out).square()
            return action.realize(), v_out.realize(), log_probs.realize()

            
        @TinyJit
        @Tensor.train()
        def train_step(obs_batch: Tensor, action_batch: Tensor, log_prob_batch: Tensor, advantage_batch: Tensor, return_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            """
            Performs one step gradient update of policy and value network.
            
            Args:
                obs_batch: (batch_size, obs_dim)
                action_batch: (batch_size, action_dim)
                log_prob_batch: (batch_size, 1)
                advantage_batch: (batch_size, 1)
                return_batch: (batch_size, 1)
            """
            # Ensure all inputs are on the correct device
            # obs_batch = obs_batch#.to(self.device)
            # action_batch = action_batch#.to(self.device)
            # log_prob_batch = log_prob_batch#.to(self.device)
            # advantage_batch = advantage_batch#.to(self.device)
            # return_batch = return_batch#.to(self.device)

            # Get new log probabilities and values
            pi_out, std_out = pi_network(obs_batch)
            new_log_prob  = -0.5 * Tensor(math.log(2 * math.pi)) - std_out.log() - 0.5 * ((action_batch - pi_out) / std_out).square()
            
            # PPO clipping objective
            ratio = (new_log_prob - log_prob_batch).exp()
            clipped_ratio = ratio.clip(1 - 0.2, 1 + 0.2)

            surr1 = ratio * advantage_batch
            surr2 = clipped_ratio * advantage_batch
            
            pi_loss = -surr1.minimum(surr2).mean()
            
            # Value function loss
            value_loss = 0.5 * (v_network(obs_batch) - return_batch).square().mean()
            
            # Total loss
            total_loss = pi_loss + value_loss

            # Gradient step
            opt.zero_grad()
            total_loss.backward()
            # clip_grad_norm_where(params, 0.5)
            opt.step()

            # Calculate approximate KL divergence
            kl_div = ((ratio - 1)- ratio.log()).mean()
            return pi_loss.realize(), value_loss.realize(), total_loss.realize(), kl_div.realize(), std_out.realize()
        
        def save_model():
            """Save model checkpoint"""
            state_dict_policy = get_state_dict(pi_network)
            state_dict_value = get_state_dict(v_network)
            
            with open(f"saved_network/pi_network.pkl", "wb") as f:
                pickle.dump(state_dict_policy, f)
            with open(f"saved_network/v_network.pkl", "wb") as f:
                pickle.dump(state_dict_value, f)

        ep_reward = 0.0
        ep_count = 0
        season_count = 0

        pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
        mean_rewards = []

        obs,_  = env.reset()
        for t in range(TOTAL_TIMESTEPS):
            Tensor.training=False
            pi_out, v_out, log_prob = get_action_value(Tensor(obs, dtype=dtypes.float32))
            
            # to numpy.
            action = pi_out.item()
            values = v_out.item()
            log_prob = log_prob.item()
    
            clipped_action = np.clip(action, lower_bound, upper_bound)
            next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated
            ep_reward += reward

            # Add to buffer
            buffer.record(obs, action, reward, values, log_prob)
            obs = next_obs
            
            # Calculate advantage and returns if it is the end of episode or its time to update
            if done or (t+1) % NUM_STEPS ==0:
                if done:
                    ep_count += 1
                # Value of last time-step
                _, last_value, _ = get_action_value(Tensor(obs, dtype=dtypes.float32))
                last_value = last_value.item()
                # Compute returns and advantage and store in buffer
                buffer.process_trajectory(
                    gamma=GAMMA,
                    gae_lam=GAE_LAM,
                    is_last_terminal=done,
                    last_v=last_value)
                obs, _  = env.reset()

            if (t+1) % NUM_STEPS==0:
                season_count += 1
                # Update for epochs
                Tensor.training = True
                for ep in range(NUM_EPOCHS):
                    batch_data = buffer.get_mini_batch(BATCH_SIZE)
                    num_grads = len(batch_data)

                    # Iterate over minibatch of data
                    for k in range(num_grads):
                        obs_batch = batch_data[k]['obs']
                        action_batch = batch_data[k]['action']
                        log_prob_batch = batch_data[k]['log_prob']
                        advantage_batch = batch_data[k]['advantage']
                        return_batch = batch_data[k]['return']

                        # Normalize advantage
                        advantage_batch = (
                            advantage_batch -
                            np.squeeze(np.mean(advantage_batch, axis=0))
                        ) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)
                        # Convert to torch tensor
                        obs_batch = Tensor(obs_batch, dtype=dtypes.float32)
                        action_batch = Tensor(action_batch, dtype=dtypes.float32)
                        log_prob_batch = Tensor(log_prob_batch, dtype=dtypes.float32)
                        advantage_batch = Tensor(advantage_batch, dtype=dtypes.float32)
                        return_batch = Tensor(return_batch, dtype=dtypes.float32)

                        # Update the networks on minibatch of data
                        pi_loss, value_loss, total_loss, kl_divergence, std_dev = train_step(obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch)
                        pi_losses.append(pi_loss.item())
                        v_losses.append(value_loss.item())
                        total_losses.append(total_loss.item())
                        approx_kls.append(kl_divergence.item())
                        stds.append(std_dev.item())
                buffer.clear()
                # Reset eval model.
                save_model()
                get_action_value.reset()

                mean_ep_reward = ep_reward / ep_count
                ep_reward, ep_count = 0.0, 0

                # summary_writer.add_scalar("misc/ep_reward_mean", np.mean(mean_ep_reward), t)
                # summary_writer.add_scalar("train/pi_loss", np.mean(pi_losses), t)
                # summary_writer.add_scalar("train/v_loss", np.mean(v_losses), t)
                # summary_writer.add_scalar("train/total_loss", np.mean(total_losses), t)
                # summary_writer.add_scalar("train/approx_kl", np.mean(approx_kls), t)
                # summary_writer.add_scalar("train/std", np.mean(stds), t)
                print(f"Season={season_count} --> mean_ep_reward={mean_ep_reward}, pi_loss={np.mean(pi_losses)}, v_loss={np.mean(v_losses)}, total_loss={np.mean(total_losses)}, approx_kl={np.mean(approx_kls)}, avg_std={np.mean(stds)}")

                mean_rewards.append(mean_ep_reward)
                pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
                
                # Save policy and value network
                # torch.save(pi_network.state_dict(), 'saved_network/pi_network.pth')
                # torch.save(v_network.state_dict(), 'saved_network/v_network.pth')
                # policy.save('./saved_network')
                
        # Close summarywriter
        # summary_writer.close()
        
        # Plot episodic reward
        _, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        ax.plot(range(season_count), mean_rewards)
        ax.set_xlabel("season")
        ax.set_ylabel("episodic reward")
        ax.grid(True)
        plt.savefig("saved_images/season_reward.png")

    elif train_test=="eval" or train_test=="test":
        # Function to create gif animation. Taken from: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553 
        def save_frames_as_gif(frames, filename):

            #Mess with this to change frame size
            plt.figure(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100), dpi=300)

            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
            anim.save(filename, writer='imagemagick', fps=120)
        
        # Evaluate trained network
        # Load saved policy network
        pi_network = PI_Network(obs_dim, action_dim, lower_bound, upper_bound)
        pi_network.load_state_dict(torch.load('saved_network/pi_network.pth'))
        obs = env.reset()
        frames = []
        for _ in range(300):
            obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
            action = pi_network(obs_torch).detach().numpy()
            clipped_action = np.clip(action[0], lower_bound, upper_bound)

            frames.append(env.render(mode="rgb_array"))
            obs, reward, done, _ = env.step(clipped_action)
        env.close()
        save_frames_as_gif(frames, filename="saved_images/pendulum_run.gif")

    else:
        print("Please specify whether to train or evaluate!!!")
        sys.exit()