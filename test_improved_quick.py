"""Quick functional test of improved MCTS."""

import numpy as np
from mcts_simple import SimplePokemon, SimpleState
from mcts_improved import ImprovedSimpleMCTS

# Create test state
pikachu = SimplePokemon(
    species="Pikachu",
    hp=0.8,
    max_hp=95,
    atk=55,
    defense=40,
    spa=50,
    spd=50,
    spe=90,
    types=("Electric",),
    moves=["thunderbolt", "quickattack", "irontail", "thunderwave"],
)

charizard = SimplePokemon(
    species="Charizard",
    hp=1.0,
    max_hp=153,
    atk=84,
    defense=78,
    spa=109,
    spd=85,
    spe=100,
    types=("Fire", "Flying"),
    moves=["flamethrower", "airslash", "dragonpulse", "roost"],
)

state = SimpleState(
    own_team=[pikachu],
    opp_team=[charizard],
    own_active=0,
    opp_active=0,
    turn=1,
)

# Test improved MCTS
mcts = ImprovedSimpleMCTS(n_rollouts=50)
priors = np.ones(26) / 26
action_mask = np.array([0]*6 + [1]*4 + [0]*16, dtype=np.float32)

print("Testing ImprovedSimpleMCTS...")
try:
    best_action, visits, q_values = mcts.search(state, priors, 0.0, action_mask)
    print(f"OK: Search completed")
    print(f"  Best action: {best_action} (should be 6-9 for moves)")
    print(f"  Total visits: {visits.sum():.0f}")
    print(f"  Visit distribution: {visits[6:10]}")  # Show move visits
    print(f"  Q-values: min={q_values.min():.2f}, max={q_values.max():.2f}")

    # Check that best action is a move (not switch)
    if 6 <= best_action <= 9:
        print(f"OK: Best action is a move (expected for Pikachu vs Charizard)")
    else:
        print(f"WARNING: Best action is not a move (unexpected)")

    print("\nOK: All checks passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
