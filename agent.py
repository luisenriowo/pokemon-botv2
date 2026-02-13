import numpy as np
import torch
import torch.nn as nn

from config import Config
from model import ActorCritic


class RolloutBuffer:
    """Stores transitions from a single trajectory for GAE computation."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 26):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, value, reward, done, mask):
        if self.ptr >= self.capacity:
            return
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.masks[self.ptr] = mask
        self.ptr += 1

    @property
    def size(self):
        return self.ptr

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        n = self.ptr
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            self.advantages[t] = gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, n_minibatches: int):
        """Yield random minibatches of indices."""
        n = self.ptr
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_size = n // n_minibatches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield indices[start:end]


class PPOAgent:
    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = ActorCritic(
            obs_dim=config.obs_size,
            action_dim=config.action_space_size,
            hidden_sizes=config.hidden_sizes,
            head_hidden=config.head_hidden,
            species_vocab=config.species_vocab,
            move_vocab=config.move_vocab,
            ability_vocab=config.ability_vocab,
            item_vocab=config.item_vocab,
            species_embed_dim=config.species_embed_dim,
            move_embed_dim=config.move_embed_dim,
            ability_embed_dim=config.ability_embed_dim,
            item_embed_dim=config.item_embed_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

    def create_buffer(self, capacity: int) -> RolloutBuffer:
        return RolloutBuffer(capacity, self.config.obs_size, self.config.action_space_size)

    def act(self, obs: np.ndarray, mask: np.ndarray):
        """Select action using current policy."""
        return self.model.act(obs, mask)

    def update(self, buffers: list[RolloutBuffer], progress: float = 0.0):
        """PPO update from one or more rollout buffers.

        Args:
            buffers: list of RolloutBuffers (one per agent trajectory)
            progress: training progress in [0, 1] for annealing
        Returns:
            dict with loss stats
        """
        cfg = self.config

        # Learning rate annealing
        if cfg.anneal_lr:
            lr = cfg.lr * (1.0 - progress)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        # Concatenate all buffers
        all_obs = np.concatenate([b.obs[:b.ptr] for b in buffers])
        all_actions = np.concatenate([b.actions[:b.ptr] for b in buffers])
        all_log_probs = np.concatenate([b.log_probs[:b.ptr] for b in buffers])
        all_masks = np.concatenate([b.masks[:b.ptr] for b in buffers])
        all_advantages = np.concatenate([b.advantages[:b.ptr] for b in buffers])
        all_returns = np.concatenate([b.returns[:b.ptr] for b in buffers])

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.FloatTensor(all_obs).to(self.device)
        actions_t = torch.LongTensor(all_actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(all_log_probs).to(self.device)
        masks_t = torch.FloatTensor(all_masks).to(self.device)
        advantages_t = torch.FloatTensor(all_advantages).to(self.device)
        returns_t = torch.FloatTensor(all_returns).to(self.device)

        n = len(all_obs)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            batch_size = n // cfg.n_minibatches

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                mb = indices[start:end]

                new_log_probs, entropy, new_values = self.model.evaluate(
                    obs_t[mb], actions_t[mb], masks_t[mb]
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs_t[mb])
                surr1 = ratio * advantages_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages_t[mb]
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_loss = 0.5 * ((new_values - returns_t[mb]) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += ent_loss.item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "v_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
