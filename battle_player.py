import numpy as np
import torch

from poke_env import Player
from poke_env.environment.singles_env import SinglesEnv

from config import Config
from environment import OBS_SIZE, embed_battle_standalone
from model import ActorCritic
from utils import compute_action_mask, load_checkpoint
from mcts import mcts_action
from mcts_simple import simple_mcts_action
from mcts_improved import improved_mcts_action


class TrainedRLPlayer(Player):
    """A poke-env Player that uses a trained ActorCritic model."""

    def __init__(
        self,
        model_path: str,
        config: Config = None,
        deterministic: bool = True,
        use_mcts: bool = False,
        mcts_mode: str = 'improved',  # 'simple', 'improved', or 'full' (Node.js)
        mcts_rollouts: int = 200,
        mcts_verbose: bool = False,
        # Improvement flags (only for 'improved' mode)
        mcts_smart_opponent: bool = True,
        mcts_value_bootstrap: bool = True,
        mcts_move_ordering: bool = True,
        mcts_adaptive_rollouts: bool = False,  # Ablation study showed this hurts performance
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config or Config()
        self.deterministic = deterministic
        self.use_mcts = use_mcts
        self.mcts_mode = mcts_mode
        self.mcts_rollouts = mcts_rollouts
        self.mcts_verbose = mcts_verbose
        self.mcts_smart_opponent = mcts_smart_opponent
        self.mcts_value_bootstrap = mcts_value_bootstrap
        self.mcts_move_ordering = mcts_move_ordering
        self.mcts_adaptive_rollouts = mcts_adaptive_rollouts

        self.model = ActorCritic(
            obs_dim=OBS_SIZE,
            action_dim=self.config.action_space_size,
            hidden_sizes=self.config.hidden_sizes,
            head_hidden=self.config.head_hidden,
            species_vocab=self.config.species_vocab,
            move_vocab=self.config.move_vocab,
            ability_vocab=self.config.ability_vocab,
            item_vocab=self.config.item_vocab,
            species_embed_dim=self.config.species_embed_dim,
            move_embed_dim=self.config.move_embed_dim,
            ability_embed_dim=self.config.ability_embed_dim,
            item_embed_dim=self.config.item_embed_dim,
        )
        load_checkpoint(model_path, self.model)
        self.model.eval()

    def choose_move(self, battle):
        obs = embed_battle_standalone(battle)
        mask = compute_action_mask(battle)

        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.FloatTensor(mask).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.model(obs_t, mask_t)

            if self.use_mcts:
                # Use MCTS with NN priors and value
                priors = dist.probs.squeeze(0).numpy()
                value_estimate = value.item()

                if self.mcts_mode == 'simple':
                    # Simplified Python-only MCTS (baseline)
                    action = simple_mcts_action(
                        battle,
                        priors,
                        value_estimate,
                        mask,
                        n_rollouts=self.mcts_rollouts,
                    )
                elif self.mcts_mode == 'improved':
                    # Improved Python MCTS with enhancements
                    action = improved_mcts_action(
                        battle,
                        priors,
                        value_estimate,
                        mask,
                        n_rollouts=self.mcts_rollouts,
                        use_smart_opponent=self.mcts_smart_opponent,
                        use_value_bootstrap=self.mcts_value_bootstrap,
                        use_move_ordering=self.mcts_move_ordering,
                        use_adaptive_rollouts=self.mcts_adaptive_rollouts,
                    )
                else:
                    # Full MCTS with Node.js Showdown simulator
                    action = mcts_action(
                        battle,
                        priors,
                        value_estimate,
                        mask,
                        verbose=self.mcts_verbose,
                    )
            else:
                # Direct policy sampling
                if self.deterministic:
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()

        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
