import numpy as np
import torch

from poke_env import Player
from poke_env.environment.singles_env import SinglesEnv

from config import Config
from environment import OBS_SIZE, embed_battle_standalone
from model import ActorCritic
from utils import compute_action_mask, load_checkpoint


class TrainedRLPlayer(Player):
    """A poke-env Player that uses a trained ActorCritic model."""

    def __init__(
        self,
        model_path: str,
        config: Config = None,
        deterministic: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config or Config()
        self.deterministic = deterministic

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
            if self.deterministic:
                action = dist.probs.argmax(dim=-1).item()
            else:
                action = dist.sample().item()

        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
