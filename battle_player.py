import numpy as np
import torch

from poke_env import Player

from poke_env.environment.singles_env import SinglesEnv

from config import Config
from environment import OBS_SIZE
from model import ActorCritic
from utils import compute_action_mask, load_checkpoint

# Import all encoding helpers from environment
from environment import (
    _encode_weather, _encode_terrain, _encode_side_conditions,
    _encode_active_pokemon, _encode_move, _encode_own_bench_pokemon,
    _encode_opp_bench_pokemon, DIM_MOVE,
)
from poke_env.battle.field import Field


def embed_battle_standalone(battle) -> np.ndarray:
    """Encode battle state without needing a Gen9Env instance."""
    parts = []

    # 1. Weather (8)
    parts.append(_encode_weather(battle))

    # 2. Terrain (5)
    parts.append(_encode_terrain(battle))

    # 3. Side conditions: own (10) + opponent (10)
    parts.append(_encode_side_conditions(battle.side_conditions))
    parts.append(_encode_side_conditions(battle.opponent_side_conditions))

    # 4. Trick room (1)
    parts.append(np.array(
        [float(Field.TRICK_ROOM in battle.fields)],
        dtype=np.float32,
    ))

    # 5. Gimmicks (6)
    parts.append(np.array([
        float(battle.can_tera),
        float(battle.can_mega_evolve),
        float(battle.can_z_move),
        float(battle.can_dynamax),
        float(battle.used_tera or False),
        float(battle.opponent_used_tera or False),
    ], dtype=np.float32))

    # 6. Meta info (4)
    own_fainted = sum(1 for p in battle.team.values() if p.fainted)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
    parts.append(np.array([
        float(battle.turn) / 100.0,
        float(own_fainted) / 6.0,
        float(opp_fainted) / 6.0,
        float(battle.force_switch),
    ], dtype=np.float32))

    # 7. Own active pokemon (78)
    parts.append(_encode_active_pokemon(battle.active_pokemon))

    # 8. Own moves (38 × 4 = 152)
    own_moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
    opp_active = battle.opponent_active_pokemon
    for i in range(4):
        move = own_moves[i] if i < len(own_moves) else None
        parts.append(_encode_move(move, opp_active))

    # 9. Opponent active pokemon (78)
    parts.append(_encode_active_pokemon(opp_active))

    # 10. Own bench (33 × 5 = 165)
    own_bench = [p for p in battle.team.values() if p != battle.active_pokemon]
    for i in range(5):
        mon = own_bench[i] if i < len(own_bench) else None
        parts.append(_encode_own_bench_pokemon(mon))

    # 11. Opponent bench (27 × 5 = 135)
    opp_bench = [p for p in battle.opponent_team.values() if p != opp_active]
    for i in range(5):
        mon = opp_bench[i] if i < len(opp_bench) else None
        parts.append(_encode_opp_bench_pokemon(mon))

    return np.concatenate(parts)


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
