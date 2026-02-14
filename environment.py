import functools
import re
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium.spaces import Box

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment.singles_env import SinglesEnv

from config import Config
from utils import compute_action_mask

# ── Type mappings ──────────────────────────────────────────────────────────────

TYPE_LIST = [
    PokemonType.BUG, PokemonType.DARK, PokemonType.DRAGON, PokemonType.ELECTRIC,
    PokemonType.FAIRY, PokemonType.FIGHTING, PokemonType.FIRE, PokemonType.FLYING,
    PokemonType.GHOST, PokemonType.GRASS, PokemonType.GROUND, PokemonType.ICE,
    PokemonType.NORMAL, PokemonType.POISON, PokemonType.PSYCHIC, PokemonType.ROCK,
    PokemonType.STEEL, PokemonType.WATER,
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_LIST)}
N_TYPES = len(TYPE_LIST)  # 18

STATUS_LIST = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUS_LIST)}
N_STATUS = len(STATUS_LIST)  # 6

WEATHER_LIST = [
    Weather.DESOLATELAND, Weather.DELTASTREAM, Weather.HAIL,
    Weather.PRIMORDIALSEA, Weather.RAINDANCE, Weather.SANDSTORM,
    Weather.SNOWSCAPE, Weather.SUNNYDAY,
]
WEATHER_TO_IDX = {w: i for i, w in enumerate(WEATHER_LIST)}
N_WEATHER = len(WEATHER_LIST)  # 8

TERRAIN_LIST = [
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
]
TERRAIN_TO_IDX = {t: i for i, t in enumerate(TERRAIN_LIST)}
N_TERRAIN_ONEHOT = 5  # includes "no terrain" at idx 0

SIDE_COND_KEYS = [
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
    SideCondition.TAILWIND,
    SideCondition.SAFEGUARD,
    SideCondition.MIST,
]
N_SIDE_COND = len(SIDE_COND_KEYS)  # 10

BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]

# ── Dimension constants ───────────────────────────────────────────────────────

DIM_WEATHER = N_WEATHER                    # 8
DIM_TERRAIN = N_TERRAIN_ONEHOT             # 5
DIM_SIDE = N_SIDE_COND                     # 10  (×2 sides = 20)
DIM_TRICK_ROOM = 1                         # 1
DIM_GIMMICK = 6                            # 6
DIM_META = 4                               # 4
DIM_ACTIVE = N_TYPES * 2 + 6 + 1 + N_STATUS + 7 + N_TYPES + 4  # 78
DIM_MOVE = 1 + 1 + N_TYPES + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 7 + 1 + 1  # 38
DIM_OWN_BENCH = 1 + 1 + N_STATUS + N_TYPES + 1 + 6  # 33
DIM_OPP_BENCH = 1 + 1 + 1 + N_TYPES + N_STATUS      # 27

# ── Categorical identity features ─────────────────────────────────────────────
# Layout in obs[N_CONTINUOUS:]:
#   [0]      own_active species
#   [1:5]    own_active moves (4)
#   [5]      own_active ability
#   [6]      own_active item
#   [7:12]   own_bench species (5)
#   [12:17]  own_bench abilities (5)
#   [17:22]  own_bench items (5)
#   [22]     opp_active species
#   [23]     opp_active ability
#   [24]     opp_active item
#   [25:30]  opp_bench species (5)
N_CATEGORICAL = 30

# ── Collision-free lookup tables (built from poke-env GenData) ────────────────

_gen9 = GenData.from_gen(9)

# Species: pokedex keys are already in to_id_str format
_SPECIES_LIST = sorted(_gen9.pokedex.keys())
SPECIES_TO_ID = {name: i + 1 for i, name in enumerate(_SPECIES_LIST)}
SPECIES_VOCAB = len(_SPECIES_LIST) + 1  # 1550

# Moves: move keys are already in to_id_str format
_MOVES_LIST = sorted(_gen9.moves.keys())
MOVE_TO_ID = {name: i + 1 for i, name in enumerate(_MOVES_LIST)}
MOVE_VOCAB = len(_MOVES_LIST) + 1  # 953

# Abilities: extract from pokedex, convert to to_id_str format
_abilities = set()
for _sd in _gen9.pokedex.values():
    for _ab in _sd.get("abilities", {}).values():
        _abilities.add(to_id_str(_ab))
_ABILITIES_LIST = sorted(_abilities)
ABILITY_TO_ID = {name: i + 1 for i, name in enumerate(_ABILITIES_LIST)}
ABILITY_VOCAB = len(_ABILITIES_LIST) + 1  # 315

# Items: parse from pokemon-showdown data (keys are already to_id_str format)
_items_ts = Path(__file__).parent / "pokemon-showdown" / "data" / "items.ts"
if _items_ts.exists():
    _ITEMS_LIST = sorted(re.findall(r"^\t(\w+): \{", _items_ts.read_text(encoding="utf-8"), re.MULTILINE))
    ITEM_TO_ID = {name: i + 1 for i, name in enumerate(_ITEMS_LIST)}
    ITEM_VOCAB = len(_ITEMS_LIST) + 1  # 584
else:
    ITEM_TO_ID = None
    ITEM_VOCAB = 600  # fallback

N_CONTINUOUS = (
    DIM_WEATHER
    + DIM_TERRAIN
    + DIM_SIDE * 2
    + DIM_TRICK_ROOM
    + DIM_GIMMICK
    + DIM_META
    + DIM_ACTIVE          # own active
    + DIM_MOVE * 4        # own moves
    + DIM_ACTIVE          # opp active
    + DIM_OWN_BENCH * 5   # own bench
    + DIM_OPP_BENCH * 5   # opp bench
)  # = 652

OBS_SIZE = N_CONTINUOUS + N_CATEGORICAL  # = 682


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _species_id(name: str) -> int:
    """Lookup species index. Returns 0 for unknown/empty."""
    return SPECIES_TO_ID.get(name, 0) if name else 0

def _move_id(name: str) -> int:
    """Lookup move index. Returns 0 for unknown/empty."""
    return MOVE_TO_ID.get(name, 0) if name else 0

def _ability_id(name: str) -> int:
    """Lookup ability index. Returns 0 for unknown/empty."""
    return ABILITY_TO_ID.get(name, 0) if name else 0

def _item_id(name: str) -> int:
    """Lookup item index. Falls back to hash if lookup table unavailable."""
    if not name:
        return 0
    if ITEM_TO_ID is not None:
        return ITEM_TO_ID.get(name, 0)
    h = 5381
    for c in name:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
    return (h % (ITEM_VOCAB - 1)) + 1

def _type_onehot(ptype: Optional[PokemonType]) -> np.ndarray:
    vec = np.zeros(N_TYPES, dtype=np.float32)
    if ptype is not None and ptype in TYPE_TO_IDX:
        vec[TYPE_TO_IDX[ptype]] = 1.0
    return vec


def _status_onehot(status: Optional[Status]) -> np.ndarray:
    vec = np.zeros(N_STATUS, dtype=np.float32)
    if status is not None and status in STATUS_TO_IDX:
        vec[STATUS_TO_IDX[status]] = 1.0
    return vec


def _get_accuracy(move: Move) -> float:
    acc = move.accuracy
    if isinstance(acc, bool):
        return 1.0
    if acc is None:
        return 0.0
    return float(acc) / 100.0 if acc > 1.0 else float(acc)


def _encode_weather(battle: AbstractBattle) -> np.ndarray:
    vec = np.zeros(DIM_WEATHER, dtype=np.float32)
    for w in battle.weather:
        if w in WEATHER_TO_IDX:
            vec[WEATHER_TO_IDX[w]] = 1.0
    return vec


def _encode_terrain(battle: AbstractBattle) -> np.ndarray:
    vec = np.zeros(DIM_TERRAIN, dtype=np.float32)
    found = False
    for f in battle.fields:
        if f in TERRAIN_TO_IDX:
            vec[TERRAIN_TO_IDX[f] + 1] = 1.0  # +1 because idx 0 = "no terrain"
            found = True
    if not found:
        vec[0] = 1.0
    return vec


def _encode_side_conditions(side_conds: dict) -> np.ndarray:
    vec = np.zeros(DIM_SIDE, dtype=np.float32)
    for i, sc in enumerate(SIDE_COND_KEYS):
        if sc in side_conds:
            val = side_conds[sc]
            if sc == SideCondition.SPIKES:
                vec[i] = float(val) / 3.0
            elif sc == SideCondition.TOXIC_SPIKES:
                vec[i] = float(val) / 2.0
            else:
                vec[i] = 1.0
    return vec


def _encode_active_pokemon(pokemon: Optional[Pokemon]) -> np.ndarray:
    vec = np.zeros(DIM_ACTIVE, dtype=np.float32)
    if pokemon is None:
        return vec

    idx = 0

    # Types (36): type_1 one-hot (18) + type_2 one-hot (18)
    vec[idx:idx + N_TYPES] = _type_onehot(pokemon.type_1)
    idx += N_TYPES
    vec[idx:idx + N_TYPES] = _type_onehot(pokemon.type_2)
    idx += N_TYPES

    # Base stats / 255 (6)
    base = pokemon.base_stats
    for key in STAT_KEYS:
        vec[idx] = float(base.get(key, 0)) / 255.0
        idx += 1

    # HP fraction (1)
    vec[idx] = pokemon.current_hp_fraction
    idx += 1

    # Status one-hot (6)
    vec[idx:idx + N_STATUS] = _status_onehot(pokemon.status)
    idx += N_STATUS

    # Boosts / 6 (7)
    boosts = pokemon.boosts
    for key in BOOST_KEYS:
        vec[idx] = float(boosts.get(key, 0)) / 6.0
        idx += 1

    # Tera type one-hot (18)
    vec[idx:idx + N_TYPES] = _type_onehot(pokemon.tera_type)
    idx += N_TYPES

    # Flags (4): is_terastallized, first_turn, must_recharge, preparing
    vec[idx] = float(pokemon.is_terastallized)
    vec[idx + 1] = float(pokemon.first_turn)
    vec[idx + 2] = float(pokemon.must_recharge)
    vec[idx + 3] = float(pokemon.preparing)
    idx += 4

    return vec


def _encode_move(move: Optional[Move], opponent: Optional[Pokemon]) -> np.ndarray:
    vec = np.zeros(DIM_MOVE, dtype=np.float32)
    if move is None or move.is_empty:
        return vec

    idx = 0

    # Base power / 200 (1)
    vec[idx] = float(move.base_power) / 200.0
    idx += 1

    # Accuracy (1)
    vec[idx] = _get_accuracy(move)
    idx += 1

    # Type one-hot (18)
    vec[idx:idx + N_TYPES] = _type_onehot(move.type)
    idx += N_TYPES

    # Category one-hot (3): PHYSICAL, SPECIAL, STATUS
    if move.category == MoveCategory.PHYSICAL:
        vec[idx] = 1.0
    elif move.category == MoveCategory.SPECIAL:
        vec[idx + 1] = 1.0
    elif move.category == MoveCategory.STATUS:
        vec[idx + 2] = 1.0
    idx += 3

    # Priority / 5 (1)
    vec[idx] = float(move.priority) / 5.0
    idx += 1

    # Type effectiveness / 4 (1)
    if opponent is not None:
        try:
            vec[idx] = float(opponent.damage_multiplier(move)) / 4.0
        except Exception:
            vec[idx] = 0.25
    else:
        vec[idx] = 0.25
    idx += 1

    # PP fraction (1)
    if move.max_pp > 0:
        vec[idx] = float(move.current_pp) / float(move.max_pp)
    idx += 1

    # Drain (1)
    vec[idx] = float(move.drain)
    idx += 1

    # Heal (1)
    vec[idx] = float(move.heal)
    idx += 1

    # Recoil (1)
    vec[idx] = float(move.recoil)
    idx += 1

    # Status inflicted: one-hot (6) + has_secondary (1) = 7
    if move.status is not None:
        status_vec = _status_onehot(move.status)
        vec[idx:idx + N_STATUS] = status_vec
    idx += N_STATUS
    vec[idx] = float(move.secondary is not None or move.status is not None)
    idx += 1

    # Force switch (1)
    vec[idx] = float(move.force_switch)
    idx += 1

    # Self boost (1)
    has_boost = (
        move.self_boost is not None
        and any(v > 0 for v in move.self_boost.values())
    )
    vec[idx] = float(has_boost)
    idx += 1

    return vec


def _encode_own_bench_pokemon(pokemon: Optional[Pokemon]) -> np.ndarray:
    vec = np.zeros(DIM_OWN_BENCH, dtype=np.float32)
    if pokemon is None:
        return vec

    idx = 0

    # HP fraction (1)
    vec[idx] = pokemon.current_hp_fraction
    idx += 1

    # Fainted (1)
    vec[idx] = float(pokemon.fainted)
    idx += 1

    # Status one-hot (6)
    vec[idx:idx + N_STATUS] = _status_onehot(pokemon.status)
    idx += N_STATUS

    # Types (18): use type_1 only for bench (compact), but plan says 18
    # Actually we encode both type slots into a single 18-dim multi-hot
    for t in pokemon.types:
        if t in TYPE_TO_IDX:
            vec[idx + TYPE_TO_IDX[t]] = 1.0
    idx += N_TYPES

    # Switchable (1): not fainted and not active
    vec[idx] = float(not pokemon.fainted and not pokemon.active)
    idx += 1

    # Base stats / 255 (6)
    base = pokemon.base_stats
    for key in STAT_KEYS:
        vec[idx] = float(base.get(key, 0)) / 255.0
        idx += 1

    return vec


def _encode_opp_bench_pokemon(pokemon: Optional[Pokemon]) -> np.ndarray:
    vec = np.zeros(DIM_OPP_BENCH, dtype=np.float32)
    if pokemon is None:
        return vec

    idx = 0

    # HP fraction (1)
    vec[idx] = pokemon.current_hp_fraction
    idx += 1

    # Fainted (1)
    vec[idx] = float(pokemon.fainted)
    idx += 1

    # Revealed (1)
    vec[idx] = float(pokemon.revealed)
    idx += 1

    # Types (18)
    for t in pokemon.types:
        if t in TYPE_TO_IDX:
            vec[idx + TYPE_TO_IDX[t]] = 1.0
    idx += N_TYPES

    # Status one-hot (6)
    vec[idx:idx + N_STATUS] = _status_onehot(pokemon.status)
    idx += N_STATUS

    return vec


# ── Environment ───────────────────────────────────────────────────────────────

def embed_battle_standalone(battle: AbstractBattle) -> np.ndarray:
    """Encode full battle state into a flat vector (shared by Gen9Env and TrainedRLPlayer)."""
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
        float(getattr(battle, 'used_tera', False) or False),
        float(getattr(battle, 'opponent_used_tera', False) or False),
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
    own_bench = [
        p for p in battle.team.values()
        if p != battle.active_pokemon
    ]
    for i in range(5):
        mon = own_bench[i] if i < len(own_bench) else None
        parts.append(_encode_own_bench_pokemon(mon))

    # 11. Opponent bench (27 × 5 = 135)
    opp_bench = [
        p for p in battle.opponent_team.values()
        if p != opp_active
    ]
    for i in range(5):
        mon = opp_bench[i] if i < len(opp_bench) else None
        parts.append(_encode_opp_bench_pokemon(mon))

    # 12. Categorical identity indices (N_CATEGORICAL = 30)
    cat = np.zeros(N_CATEGORICAL, dtype=np.float32)
    active = battle.active_pokemon
    if active:
        cat[0] = _species_id(active.species)
        active_moves = list(active.moves.values())
        for i in range(min(4, len(active_moves))):
            cat[1 + i] = _move_id(active_moves[i].id)
        if active.ability:
            cat[5] = _ability_id(active.ability)
        if active.item:
            cat[6] = _item_id(active.item)
    for i, mon in enumerate(own_bench[:5]):
        cat[7 + i] = _species_id(mon.species)
        if mon.ability:
            cat[12 + i] = _ability_id(mon.ability)
        if mon.item:
            cat[17 + i] = _item_id(mon.item)
    if opp_active:
        cat[22] = _species_id(opp_active.species)
        if opp_active.ability:
            cat[23] = _ability_id(opp_active.ability)
        if opp_active.item:
            cat[24] = _item_id(opp_active.item)
    for i, mon in enumerate(opp_bench[:5]):
        cat[25 + i] = _species_id(mon.species)
    parts.append(cat)

    obs = np.concatenate(parts)
    assert obs.shape == (OBS_SIZE,), f"Expected {OBS_SIZE}, got {obs.shape[0]}"
    return obs


class Gen9Env(SinglesEnv):
    def __init__(self, config: Config, **kwargs):
        super().__init__(
            battle_format=config.battle_format,
            strict=False,
            **kwargs,
        )
        self.config = config
        self.current_masks = {}
        self.current_battles = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-2.0, high=2.0, shape=(OBS_SIZE,), dtype=np.float32)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        obs = embed_battle_standalone(battle)

        # Store mask and battle reference for training loop access
        agent_name = battle.player_username
        self.current_masks[agent_name] = compute_action_mask(battle)
        self.current_battles[agent_name] = battle

        return obs

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Sparse terminal reward: +1 win, -1 loss, 0 otherwise."""
        if battle.won:
            return 1.0
        elif battle.lost:
            return -1.0
        return 0.0
