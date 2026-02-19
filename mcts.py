"""
MCTS Integration for Pokemon RL

Manages subprocess communication with mcts_service.js for MCTS-enhanced action selection.
"""

import json
import subprocess
import numpy as np
import atexit
from typing import Optional, Tuple

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition


class MCTSEngine:
    """Wrapper for Node.js MCTS service."""

    def __init__(self, service_path: str = "mcts_service.js", verbose: bool = False):
        """
        Initialize MCTS engine.

        Args:
            service_path: Path to mcts_service.js
            verbose: Enable debug logging
        """
        self.service_path = service_path
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        self._start_service()

    def _start_service(self):
        """Start the Node.js MCTS service."""
        try:
            self.process = subprocess.Popen(
                ['node', self.service_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            atexit.register(self.shutdown)
            if self.verbose:
                print(f"[MCTS] Started service: {self.service_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to start MCTS service: {e}")

    def shutdown(self):
        """Shutdown the MCTS service."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            if self.verbose:
                print("[MCTS] Service terminated")

    def search(
        self,
        battle: AbstractBattle,
        priors: np.ndarray,
        value: float,
        action_mask: np.ndarray,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Run MCTS search to find best action.

        Args:
            battle: Current battle state
            priors: Neural network policy (26-dim softmax)
            value: Neural network value estimate
            action_mask: Legal actions (26-dim binary)

        Returns:
            (best_action, visit_counts, q_values)
        """
        # Serialize battle state
        state = self._serialize_battle(battle)

        # Build request
        request = {
            "state": state,
            "priors": priors.tolist(),
            "value": float(value),
            "action_mask": action_mask.tolist(),
        }

        # Send to service
        if not self.process or self.process.poll() is not None:
            # Check if process crashed
            if self.process:
                stderr_output = self.process.stderr.read()
                raise RuntimeError(f"MCTS service crashed. stderr: {stderr_output}")
            raise RuntimeError("MCTS service is not running")

        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                # Check for stderr output
                stderr_output = ""
                if self.process.stderr:
                    import select
                    import sys
                    # Non-blocking read (platform-specific)
                    if sys.platform == 'win32':
                        # Windows doesn't support select on pipes, just try to read
                        try:
                            import msvcrt
                            if msvcrt.kbhit():
                                stderr_output = self.process.stderr.read()
                        except:
                            pass
                    else:
                        # Unix-like systems
                        import select
                        if select.select([self.process.stderr], [], [], 0)[0]:
                            stderr_output = self.process.stderr.read()

                raise RuntimeError(f"MCTS service returned empty response. stderr: {stderr_output}")

            response = json.loads(response_line)

            if "error" in response:
                raise RuntimeError(f"MCTS service error: {response['error']}")

            best_action = response["bestAction"]
            visits = np.array(response["visits"], dtype=np.float32)
            q_values = np.array(response["qValues"], dtype=np.float32)

            if self.verbose:
                print(f"[MCTS] Best action: {best_action}, Visits: {visits[best_action]:.0f}")

            return best_action, visits, q_values

        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCTS service returned invalid JSON: {response_line}. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"MCTS search failed: {e}")

    def _serialize_battle(self, battle: AbstractBattle) -> dict:
        """Convert poke-env Battle to serializable state."""
        own_team = []
        for mon in battle.team.values():
            own_team.append({
                "species": mon.species,
                "level": mon.level,
                "moves": [m.id for m in mon.moves.values()],
                "ability": mon.ability,
                "item": mon.item,
                "hp": int(mon.current_hp_fraction * mon.max_hp) if mon.max_hp else 0,
                "maxhp": mon.max_hp or 0,
                "status": mon.status.name if mon.status else None,
                "boosts": dict(mon.boosts),
                "active": mon.active,
                "fainted": mon.fainted,
                "gender": mon.gender.name if mon.gender else "",
            })

        opp_team = []
        for mon in battle.opponent_team.values():
            # Include all known info, even if incomplete
            moves = [m.id for m in mon.moves.values()] if mon.moves else []
            opp_team.append({
                "species": mon.species,
                "level": mon.level,
                "moves": moves,
                "ability": mon.ability,  # None if not revealed
                "item": mon.item,        # None if not revealed
                "hp_fraction": mon.current_hp_fraction,
                "status": mon.status.name if mon.status else None,
                "boosts": dict(mon.boosts) if mon.boosts else {},
                "active": mon.active,
                "fainted": mon.fainted,
                "revealed": mon.revealed,
                "gender": mon.gender.name if mon.gender else "",
            })

        # Extract field conditions
        field = {
            "weather": next((w.name for w in battle.weather), None),
            "terrain": None,
            "trick_room": Field.TRICK_ROOM in battle.fields,
        }

        # Check for terrain
        for f in battle.fields:
            if f in [Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
                     Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN]:
                field["terrain"] = f.name
                break

        # Side conditions
        side_conditions = {sc.name: v for sc, v in battle.side_conditions.items()}
        opp_side_conditions = {sc.name: v for sc, v in battle.opponent_side_conditions.items()}

        return {
            "own_team": own_team,
            "opp_team": opp_team,
            "opp_team_size": 6,  # gen9randombattle always 6
            "field": field,
            "side_conditions": side_conditions,
            "opp_side_conditions": opp_side_conditions,
            "turn": battle.turn,
        }


# Singleton instance
_mcts_engine: Optional[MCTSEngine] = None


def get_mcts_engine(verbose: bool = False) -> MCTSEngine:
    """Get or create the global MCTS engine instance."""
    global _mcts_engine
    if _mcts_engine is None:
        _mcts_engine = MCTSEngine(verbose=verbose)
    return _mcts_engine


def mcts_action(
    battle: AbstractBattle,
    priors: np.ndarray,
    value: float,
    action_mask: np.ndarray,
    verbose: bool = False,
) -> int:
    """
    Get MCTS-enhanced action.

    Args:
        battle: Current battle state
        priors: NN policy priors (26-dim)
        value: NN value estimate
        action_mask: Legal actions (26-dim)
        verbose: Debug logging

    Returns:
        Best action index (0-25)
    """
    engine = get_mcts_engine(verbose=verbose)
    best_action, _, _ = engine.search(battle, priors, value, action_mask)
    return best_action
