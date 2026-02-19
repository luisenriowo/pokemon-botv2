"""
Simple test for MCTS service standalone functionality.

Tests basic IPC communication and state serialization.
"""

import json
import numpy as np
from mcts import MCTSEngine


def create_dummy_state():
    """Create a minimal valid battle state for testing."""
    return {
        "own_team": [
            {
                "species": "Pikachu",
                "level": 50,
                "moves": ["thunderbolt", "quickattack", "irontail", "thunderwave"],
                "ability": "static",
                "item": "lightball",
                "hp": 95,
                "maxhp": 95,
                "status": None,
                "boosts": {},
                "active": True,
                "fainted": False,
                "gender": "M",
            },
            {
                "species": "Charizard",
                "level": 50,
                "moves": ["flamethrower", "airslash", "dragonpulse", "roost"],
                "ability": "blaze",
                "item": "leftovers",
                "hp": 153,
                "maxhp": 153,
                "status": None,
                "boosts": {},
                "active": False,
                "fainted": False,
                "gender": "M",
            },
        ],
        "opp_team": [
            {
                "species": "Venusaur",
                "level": 50,
                "moves": ["gigadrain", "sludgebomb"],
                "ability": None,  # Unknown
                "item": None,     # Unknown
                "hp_fraction": 0.75,
                "status": None,
                "boosts": {},
                "active": True,
                "fainted": False,
                "revealed": True,
                "gender": "F",
            },
        ],
        "opp_team_size": 6,
        "field": {
            "weather": None,
            "terrain": None,
            "trick_room": False,
        },
        "side_conditions": {},
        "opp_side_conditions": {},
        "turn": 3,
    }


def test_basic_search():
    """Test basic MCTS search functionality."""
    print("Testing MCTS service...")
    print("-" * 60)

    # Create engine
    engine = MCTSEngine(verbose=True)

    # Create dummy inputs
    state = create_dummy_state()
    priors = np.ones(26, dtype=np.float32) / 26  # Uniform priors
    value = 0.0
    action_mask = np.array([0]*6 + [1]*4 + [0]*16, dtype=np.float32)  # Only moves 6-9 legal

    print("\nState:")
    print(f"  Own active: {state['own_team'][0]['species']}")
    print(f"  Opp active: {state['opp_team'][0]['species']}")
    print(f"  Turn: {state['turn']}")
    print(f"\nLegal actions: {np.where(action_mask > 0)[0].tolist()}")

    # Run search
    print("\nRunning MCTS search...")
    try:
        best_action, visits, q_values = engine.search(state, priors, value, action_mask)

        print("\nResults:")
        print(f"  Best action: {best_action}")
        print(f"  Total visits: {visits.sum():.0f}")
        print(f"  Top 3 actions by visits:")
        top_indices = np.argsort(visits)[-3:][::-1]
        for idx in top_indices:
            if visits[idx] > 0:
                print(f"    Action {idx}: {visits[idx]:.0f} visits, Q={q_values[idx]:.3f}")

        print("\n✓ MCTS service test passed!")
        return True

    except Exception as e:
        print(f"\n✗ MCTS service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        engine.shutdown()


if __name__ == "__main__":
    success = test_basic_search()
    exit(0 if success else 1)
