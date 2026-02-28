import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, pi_network, v_network, learning_rate, clip_range, value_coeff, 
                 obs_dim, action_dim, initial_std=1.0, max_grad_norm=0.5, device='cpu'):
        super().__init__()

        self.pi_network = pi_network.to(device)
        self.v_network = v_network.to(device)
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Gaussian policy with learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(self.action_dim, device=device) * torch.log(torch.tensor(initial_std)), 
            requires_grad=True
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        pi_out = self.pi_network(obs)
        dist_out = Normal(pi_out, torch.exp(self.log_std))
        v_out = self.v_network(obs)
        return dist_out, v_out

    @torch.no_grad()
    def get_action(self, obs):
        """Sample action based on current policy"""
        obs_torch = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        dist, values = self.forward(obs_torch)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action[0].cpu().numpy(),
            log_prob.cpu().item(),
            values.cpu().item()
        )

    @torch.no_grad()
    def get_values(self, obs):
        """Return value of the state"""
        obs_torch = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        _, values = self.forward(obs_torch)
        return values.cpu().item()

    def evaluate_action(self, obs_batch, action_batch):
        """Evaluate taken action."""
        dist, values = self.forward(obs_batch)
        log_prob = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)
        return log_prob, values

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
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
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        log_prob_batch = log_prob_batch.to(self.device)
        advantage_batch = advantage_batch.to(self.device)
        return_batch = return_batch.to(self.device)

        # Get new log probabilities and values
        new_log_prob, values = self.evaluate_action(obs_batch, action_batch)

        # PPO clipping objective
        ratio = torch.exp(new_log_prob - log_prob_batch)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        
        # Value function loss
        value_loss = self.value_coeff * torch.mean((values - return_batch) ** 2)
        
        # Total loss
        total_loss = pi_loss + value_loss

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Calculate approximate KL divergence
        with torch.no_grad():
            kl_div = torch.mean(log_prob_batch - new_log_prob)

        return {
            'pi_loss': pi_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_div.item(),
            'std_dev': torch.exp(self.log_std).cpu().detach().numpy(),
            'mean_ratio': ratio.mean().item(),
            'clip_fraction': ((ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)).float().mean().item()
        }

    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'log_std': self.log_std,
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])