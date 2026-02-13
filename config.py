from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Environment
    battle_format: str = "gen9randombattle"
    action_space_size: int = 26
    obs_size: int = 682

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    head_hidden: int = 256

    # Embedding vocab sizes
    species_vocab: int = 1500
    move_vocab: int = 1000
    ability_vocab: int = 400
    item_vocab: int = 300

    # Embedding dimensions
    species_embed_dim: int = 24
    move_embed_dim: int = 16
    ability_embed_dim: int = 12
    item_embed_dim: int = 12

    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.9999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4

    # Rollout
    rollout_steps: int = 1024

    # Training
    total_timesteps: int = 10_000_000
    eval_freq: int = 100_000
    save_freq: int = 50_000
    eval_battles: int = 100

    # Self-play
    opponent_update_freq: int = 200_000
    opponent_pool_size: int = 5
    use_opponent_pool: bool = False

    # MCTS (Phase 3)
    mcts_rollouts: int = 500
    mcts_c_puct: float = 1.5
    mcts_time_limit: float = 8.0

    # Paths
    model_save_dir: str = "models"
    log_dir: str = "logs"
    server_port: int = 8000

    # Annealing
    anneal_lr: bool = True
