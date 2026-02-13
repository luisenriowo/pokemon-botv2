import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import N_CONTINUOUS, N_CATEGORICAL


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    # Categorical index layout (offsets within obs[N_CONTINUOUS:])
    _S_OWN_ACTIVE = 0        # 1 species
    _M_OWN_ACTIVE = 1        # 4 moves
    _A_OWN_ACTIVE = 5        # 1 ability
    _I_OWN_ACTIVE = 6        # 1 item
    _S_OWN_BENCH = 7         # 5 species
    _A_OWN_BENCH = 12        # 5 abilities
    _I_OWN_BENCH = 17        # 5 items
    _S_OPP_ACTIVE = 22       # 1 species
    _A_OPP_ACTIVE = 23       # 1 ability
    _I_OPP_ACTIVE = 24       # 1 item
    _S_OPP_BENCH = 25        # 5 species

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 26,
        hidden_sizes=(512, 256),
        head_hidden: int = 256,
        species_vocab: int = 1500,
        move_vocab: int = 1000,
        ability_vocab: int = 400,
        item_vocab: int = 300,
        species_embed_dim: int = 24,
        move_embed_dim: int = 16,
        ability_embed_dim: int = 12,
        item_embed_dim: int = 12,
    ):
        super().__init__()

        self.n_continuous = N_CONTINUOUS
        self.n_categorical = N_CATEGORICAL

        # Embedding layers (padding_idx=0 → unknown/empty maps to zero vector)
        self.species_emb = nn.Embedding(species_vocab, species_embed_dim, padding_idx=0)
        self.move_emb = nn.Embedding(move_vocab, move_embed_dim, padding_idx=0)
        self.ability_emb = nn.Embedding(ability_vocab, ability_embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(item_vocab, item_embed_dim, padding_idx=0)

        # 12 species + 4 moves + 7 abilities + 7 items
        n_species_slots = 12
        n_move_slots = 4
        n_ability_slots = 7
        n_item_slots = 7
        embed_total = (
            n_species_slots * species_embed_dim
            + n_move_slots * move_embed_dim
            + n_ability_slots * ability_embed_dim
            + n_item_slots * item_embed_dim
        )

        input_dim = self.n_continuous + embed_total

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            _ortho_init(nn.Linear(input_dim, hidden_sizes[0])),
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

    def _embed_categorical(self, cat: torch.Tensor) -> torch.Tensor:
        """Convert categorical index columns to concatenated embedding vectors.

        Args:
            cat: (batch, N_CATEGORICAL) long tensor of identity indices
        Returns:
            (batch, embed_total) float tensor
        """
        species_ids = torch.cat([
            cat[:, self._S_OWN_ACTIVE : self._S_OWN_ACTIVE + 1],
            cat[:, self._S_OWN_BENCH  : self._S_OWN_BENCH + 5],
            cat[:, self._S_OPP_ACTIVE : self._S_OPP_ACTIVE + 1],
            cat[:, self._S_OPP_BENCH  : self._S_OPP_BENCH + 5],
        ], dim=1)  # (batch, 12)

        move_ids = cat[:, self._M_OWN_ACTIVE : self._M_OWN_ACTIVE + 4]  # (batch, 4)

        ability_ids = torch.cat([
            cat[:, self._A_OWN_ACTIVE : self._A_OWN_ACTIVE + 1],
            cat[:, self._A_OWN_BENCH  : self._A_OWN_BENCH + 5],
            cat[:, self._A_OPP_ACTIVE : self._A_OPP_ACTIVE + 1],
        ], dim=1)  # (batch, 7)

        item_ids = torch.cat([
            cat[:, self._I_OWN_ACTIVE : self._I_OWN_ACTIVE + 1],
            cat[:, self._I_OWN_BENCH  : self._I_OWN_BENCH + 5],
            cat[:, self._I_OPP_ACTIVE : self._I_OPP_ACTIVE + 1],
        ], dim=1)  # (batch, 7)

        return torch.cat([
            self.species_emb(species_ids).flatten(1),
            self.move_emb(move_ids).flatten(1),
            self.ability_emb(ability_ids).flatten(1),
            self.item_emb(item_ids).flatten(1),
        ], dim=-1)

    def forward(self, obs, action_mask):
        """
        Args:
            obs: (batch, obs_dim) -- continuous features followed by categorical indices
            action_mask: (batch, action_dim) binary mask, 1=legal
        Returns:
            dist: Categorical distribution over legal actions
            value: (batch, 1) state value
        """
        cont = obs[:, :self.n_continuous]
        cat = obs[:, self.n_continuous:].long()

        embeddings = self._embed_categorical(cat)
        x = torch.cat([cont, embeddings], dim=-1)

        features = self.feature_extractor(x)
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
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

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
