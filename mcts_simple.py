"""
Simplified Python-only MCTS for Pokemon battles.

This is a lightweight MCTS implementation that uses heuristic-based simulation
instead of Showdown's full simulator. Good enough to validate if MCTS helps
before investing in full Showdown integration.

Simplifications:
- Damage calculation uses type effectiveness + base power + stats (no abilities/items)
- No complex mechanics (weather, terrain, abilities, items)
- Assumes perfect speed calculation
- Simplified status/boost effects
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.data import GenData

# Suppress harmless division warnings from np.where
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


@dataclass
class SimplePokemon:
    """Simplified Pokemon state for fast simulation."""
    species: str
    hp: float  # 0.0 to 1.0
    max_hp: int
    atk: int
    defense: int
    spa: int
    spd: int
    spe: int
    types: Tuple[str, ...]
    moves: List[str]
    status: Optional[str] = None
    boosts: Dict[str, int] = None
    fainted: bool = False

    def __post_init__(self):
        if self.boosts is None:
            self.boosts = {}


@dataclass
class SimpleState:
    """Simplified battle state."""
    own_team: List[SimplePokemon]
    opp_team: List[SimplePokemon]
    own_active: int  # index into own_team
    opp_active: int  # index into opp_team
    turn: int = 0

    def clone(self) -> 'SimpleState':
        """Deep copy of state."""
        import copy
        return copy.deepcopy(self)

    def is_terminal(self) -> bool:
        """Check if battle is over."""
        own_alive = any(not m.fainted for m in self.own_team)
        opp_alive = any(not m.fainted for m in self.opp_team)
        return not own_alive or not opp_alive

    def get_reward(self) -> float:
        """Get reward for terminal state (from our perspective)."""
        own_alive = sum(1 for m in self.own_team if not m.fainted)
        opp_alive = sum(1 for m in self.opp_team if not m.fainted)

        if own_alive == 0:
            return -1.0  # We lost
        if opp_alive == 0:
            return 1.0   # We won

        # Partial reward based on HP advantage
        own_hp = sum(m.hp for m in self.own_team if not m.fainted)
        opp_hp = sum(m.hp for m in self.opp_team if not m.fainted)
        return np.tanh((own_hp - opp_hp) / 3.0)  # Normalized to [-1, 1]


class SimplifiedBattleSimulator:
    """Fast heuristic-based battle simulator."""

    def __init__(self):
        self.gen_data = GenData.from_gen(9)
        # type_chart is already available: type_chart[attacker_type][defender_type] = multiplier

    def get_type_effectiveness(self, move_type: str, defender_types: Tuple[str, ...]) -> float:
        """Calculate type effectiveness multiplier."""
        multiplier = 1.0
        move_type_upper = move_type.upper()

        for def_type in defender_types:
            def_type_upper = def_type.upper()
            if move_type_upper in self.gen_data.type_chart and def_type_upper in self.gen_data.type_chart[move_type_upper]:
                multiplier *= self.gen_data.type_chart[move_type_upper][def_type_upper]

        return multiplier

    def calculate_damage(self, attacker: SimplePokemon, defender: SimplePokemon,
                        move_id: str) -> float:
        """
        Calculate damage as fraction of defender's HP.

        Simplified damage formula:
        damage = (base_power * attack * type_eff * boosts) / (defense * 50)
        """
        # Get move data
        try:
            move_data = self.gen_data.moves[move_id]
        except KeyError:
            return 0.0  # Unknown move

        base_power = move_data.get('basePower', 0)
        if base_power == 0:
            return 0.0  # Status move

        category = move_data.get('category', 'Status')
        move_type = move_data.get('type', 'Normal')

        # Determine attack/defense stats
        if category == 'Physical':
            attack = attacker.atk
            defense = defender.defense
            boost_atk = attacker.boosts.get('atk', 0)
            boost_def = defender.boosts.get('def', 0)
        elif category == 'Special':
            attack = attacker.spa
            defense = defender.spd
            boost_atk = attacker.boosts.get('spa', 0)
            boost_def = defender.boosts.get('spd', 0)
        else:
            return 0.0  # Status move

        # Apply boosts
        boost_multiplier = max(2, 2 + boost_atk) / max(2, 2 - boost_def)

        # Type effectiveness
        type_eff = self.get_type_effectiveness(move_type, defender.types)

        # STAB (Same Type Attack Bonus)
        stab = 1.5 if move_type.upper() in [t.upper() for t in attacker.types] else 1.0

        # Simplified damage calculation (approximating level 50 formula)
        # Real formula: ((2*level/5 + 2) * base_power * attack / defense / 50 + 2) * modifiers
        # Simplified: (base_power * attack * type_eff * stab * boosts) / (defense * 2)
        damage = (base_power * attack * type_eff * stab * boost_multiplier) / (defense * 2)

        # Random factor (0.85 - 1.0)
        damage *= np.random.uniform(0.85, 1.0)

        # Return as fraction (damage is absolute HP, normalize by max HP)
        damage_fraction = damage / defender.max_hp
        return min(1.0, damage_fraction)

    def simulate_turn(self, state: SimpleState, own_action: int, opp_action: int) -> SimpleState:
        """
        Simulate one turn and return new state.

        Actions: 0-5 = switch to pokemon 0-5, 6-9 = use move 0-3
        (ignores mega/z-move/dmax/tera for simplicity)
        """
        new_state = state.clone()

        own_mon = new_state.own_team[new_state.own_active]
        opp_mon = new_state.opp_team[new_state.opp_active]

        # Process switches first
        if own_action < 6:  # Switch
            switch_idx = own_action
            if switch_idx < len(new_state.own_team) and not new_state.own_team[switch_idx].fainted:
                new_state.own_active = switch_idx
                own_mon = new_state.own_team[switch_idx]

        if opp_action < 6:  # Switch
            switch_idx = opp_action
            if switch_idx < len(new_state.opp_team) and not new_state.opp_team[switch_idx].fainted:
                new_state.opp_active = switch_idx
                opp_mon = new_state.opp_team[switch_idx]

        # Determine move order (simplified: just use speed)
        own_goes_first = own_mon.spe >= opp_mon.spe

        # Execute moves in order
        if own_action >= 6 and opp_action >= 6:  # Both use moves
            if own_goes_first:
                self._execute_move(new_state, True, own_action)
                if not new_state.is_terminal():
                    self._execute_move(new_state, False, opp_action)
            else:
                self._execute_move(new_state, False, opp_action)
                if not new_state.is_terminal():
                    self._execute_move(new_state, True, own_action)
        elif own_action >= 6:  # Only we use a move
            self._execute_move(new_state, True, own_action)
        elif opp_action >= 6:  # Only opponent uses a move
            self._execute_move(new_state, False, opp_action)

        new_state.turn += 1
        return new_state

    def _execute_move(self, state: SimpleState, is_our_move: bool, action: int):
        """Execute a move action."""
        if is_our_move:
            attacker = state.own_team[state.own_active]
            defender = state.opp_team[state.opp_active]
        else:
            attacker = state.opp_team[state.opp_active]
            defender = state.own_team[state.own_active]

        if attacker.fainted or defender.fainted:
            return

        # Get move index (6-9 -> 0-3)
        move_idx = action - 6
        if move_idx >= len(attacker.moves):
            return

        move_id = attacker.moves[move_idx]

        # Calculate and apply damage
        damage_fraction = self.calculate_damage(attacker, defender, move_id)
        defender.hp = max(0.0, defender.hp - damage_fraction)

        if defender.hp == 0.0:
            defender.fainted = True


class SimpleMCTS:
    """Simplified MCTS using heuristic simulation."""

    def __init__(self, n_rollouts: int = 200, c_puct: float = 1.0):
        self.n_rollouts = n_rollouts
        self.c_puct = c_puct
        self.simulator = SimplifiedBattleSimulator()

    def search(self, battle, priors: np.ndarray,
               value: float, action_mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Run MCTS search.

        Args:
            battle: AbstractBattle or SimpleState
            priors: NN policy priors
            value: NN value estimate
            action_mask: Legal actions

        Returns:
            (best_action, visit_counts, q_values)
        """
        # Convert to simple state if needed
        if isinstance(battle, SimpleState):
            state = battle
        else:
            state = self._battle_to_state(battle)

        # Initialize statistics
        visit_counts = np.zeros(26, dtype=np.float32)
        total_value = np.zeros(26, dtype=np.float32)

        # Run rollouts
        for _ in range(self.n_rollouts):
            action = self._select_action(priors, visit_counts, total_value, action_mask)
            reward = self._rollout(state, action, action_mask)

            # Update statistics
            visit_counts[action] += 1
            total_value[action] += reward

        # Compute Q-values
        q_values = np.where(visit_counts > 0, total_value / visit_counts, 0.0)

        # Select best action (highest visit count among legal actions)
        legal_visits = np.where(action_mask > 0, visit_counts, -np.inf)
        best_action = int(np.argmax(legal_visits))

        return best_action, visit_counts, q_values

    def _select_action(self, priors: np.ndarray, visits: np.ndarray,
                      total_value: np.ndarray, mask: np.ndarray) -> int:
        """Select action using PUCT."""
        q_values = np.where(visits > 0, total_value / visits, 0.0)
        sqrt_total = np.sqrt(np.sum(visits) + 1)
        u_values = self.c_puct * priors * sqrt_total / (1 + visits)

        # PUCT score
        scores = q_values + u_values

        # Mask illegal actions
        scores = np.where(mask > 0, scores, -np.inf)

        return int(np.argmax(scores))

    def _rollout(self, state: SimpleState, first_action: int, action_mask: np.ndarray) -> float:
        """
        Rollout from state using first_action, then random policy.

        Returns reward from our perspective.
        """
        sim_state = state.clone()

        # First move with our chosen action
        opp_action = self._sample_opponent_action(sim_state)
        sim_state = self.simulator.simulate_turn(sim_state, first_action, opp_action)

        # Random rollout until terminal (max 10 turns to avoid infinite loops)
        for _ in range(10):
            if sim_state.is_terminal():
                break

            own_action = self._sample_random_action(sim_state, is_own=True)
            opp_action = self._sample_opponent_action(sim_state)
            sim_state = self.simulator.simulate_turn(sim_state, own_action, opp_action)

        return sim_state.get_reward()

    def _sample_random_action(self, state: SimpleState, is_own: bool) -> int:
        """Sample random legal action."""
        if is_own:
            active_mon = state.own_team[state.own_active]
            team = state.own_team
        else:
            active_mon = state.opp_team[state.opp_active]
            team = state.opp_team

        actions = []

        # Moves (up to 4)
        for i in range(min(4, len(active_mon.moves))):
            actions.append(6 + i)

        # Switches (alive teammates)
        for i, mon in enumerate(team):
            if not mon.fainted and i != (state.own_active if is_own else state.opp_active):
                actions.append(i)

        return np.random.choice(actions) if actions else 6  # Default to first move

    def _sample_opponent_action(self, state: SimpleState) -> int:
        """Sample opponent action (could use heuristics here)."""
        return self._sample_random_action(state, is_own=False)

    def _battle_to_state(self, battle: AbstractBattle) -> SimpleState:
        """Convert poke-env battle to SimpleState."""
        own_team = []
        for mon in battle.team.values():
            stats = mon.base_stats
            own_team.append(SimplePokemon(
                species=mon.species,
                hp=mon.current_hp_fraction,
                max_hp=mon.max_hp or 100,
                atk=stats.get('atk', 100),
                defense=stats.get('def', 100),
                spa=stats.get('spa', 100),
                spd=stats.get('spd', 100),
                spe=stats.get('spe', 100),
                types=tuple(t.name for t in mon.types),
                moves=[m.id for m in mon.moves.values()],
                status=mon.status.name if mon.status else None,
                boosts=dict(mon.boosts),
                fainted=mon.fainted,
            ))

        opp_team = []
        for mon in battle.opponent_team.values():
            # Use revealed info or estimates
            stats = mon.base_stats
            opp_team.append(SimplePokemon(
                species=mon.species,
                hp=mon.current_hp_fraction if mon.revealed else 1.0,
                max_hp=mon.max_hp or 100,
                atk=stats.get('atk', 100),
                defense=stats.get('def', 100),
                spa=stats.get('spa', 100),
                spd=stats.get('spd', 100),
                spe=stats.get('spe', 100),
                types=tuple(t.name for t in mon.types),
                moves=[m.id for m in mon.moves.values()] if mon.moves else ['tackle'],
                status=mon.status.name if mon.status else None,
                boosts=dict(mon.boosts) if mon.boosts else {},
                fainted=mon.fainted,
            ))

        # Find active indices
        own_active = next((i for i, m in enumerate(battle.team.values()) if m.active), 0)
        opp_active = next((i for i, m in enumerate(battle.opponent_team.values()) if m.active), 0)

        return SimpleState(
            own_team=own_team,
            opp_team=opp_team,
            own_active=own_active,
            opp_active=opp_active,
            turn=battle.turn,
        )


# Singleton instance
_simple_mcts: Optional[SimpleMCTS] = None


def get_simple_mcts(n_rollouts: int = 200) -> SimpleMCTS:
    """Get or create global SimpleMCTS instance."""
    global _simple_mcts
    if _simple_mcts is None:
        _simple_mcts = SimpleMCTS(n_rollouts=n_rollouts)
    return _simple_mcts


def simple_mcts_action(battle: AbstractBattle, priors: np.ndarray,
                       value: float, action_mask: np.ndarray,
                       n_rollouts: int = 200) -> int:
    """
    Get MCTS-enhanced action using simplified simulator.

    Args:
        battle: Current battle
        priors: NN policy (26-dim)
        value: NN value estimate
        action_mask: Legal actions (26-dim)
        n_rollouts: Number of MCTS rollouts

    Returns:
        Best action index
    """
    mcts = get_simple_mcts(n_rollouts=n_rollouts)
    best_action, _, _ = mcts.search(battle, priors, value, action_mask)
    return best_action
