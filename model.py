import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 26, hidden_sizes=(512, 256), head_hidden: int = 256):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            _ortho_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            _ortho_init(nn.Linear(hidden_sizes[1], head_hidden)),
            nn.ReLU(),
            _ortho_init(nn.Linear(head_hidden, action_dim), gain=0.01),
        )

        # Critic head
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(hidden_sizes[1], head_hidden)),
            nn.ReLU(),
            _ortho_init(nn.Linear(head_hidden, 1), gain=1.0),
        )

    def forward(self, obs, action_mask):
        """
        Args:
            obs: (batch, obs_dim)
            action_mask: (batch, action_dim) binary mask, 1=legal
        Returns:
            dist: Categorical distribution over legal actions
            value: (batch, 1) state value
        """
        features = self.feature_extractor(obs)
        logits = self.actor(features)

        # Mask illegal actions with -inf
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        dist = Categorical(logits=logits)
        value = self.critic(features)
        return dist, value

    def act(self, obs, action_mask):
        """Single-step action selection (no grad).

        Args:
            obs: (obs_dim,) numpy array
            action_mask: (action_dim,) numpy array
        Returns:
            action: int
            log_prob: float
            value: float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.forward(obs_t, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs, action, action_mask):
        """Evaluate actions for PPO update.

        Args:
            obs: (batch, obs_dim)
            action: (batch,) action indices
            action_mask: (batch, action_dim)
        Returns:
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch,)
        """
        dist, value = self.forward(obs, action_mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)
