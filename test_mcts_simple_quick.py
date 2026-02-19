"""Quick test to verify simplified MCTS works correctly."""

import numpy as np
from mcts_simple import SimpleMCTS, SimplePokemon, SimpleState

def test_basic_simulation():
    """Test that basic simulation doesn't crash."""
    print("Test 1: Basic simulation")

    # Create simple state
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

    mcts = SimpleMCTS(n_rollouts=50)

    # Create dummy inputs
    priors = np.ones(26, dtype=np.float32) / 26
    action_mask = np.array([0]*6 + [1]*4 + [0]*16, dtype=np.float32)  # Only moves legal

    try:
        best_action, visits, q_values = mcts.search(
            state,  # Pass SimpleState directly for testing
            priors,
            0.0,
            action_mask,
        )

        print(f"  OK: Simulation completed")
        print(f"  Best action: {best_action} (should be 6-9)")
        print(f"  Total visits: {visits.sum():.0f}")
        print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")

        # Verify action is legal
        assert 6 <= best_action <= 9, f"Best action {best_action} not in move range!"
        assert visits.sum() == 50, f"Wrong number of visits: {visits.sum()}"

        print(f"  OK: All assertions passed\n")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_damage_calculation():
    """Test type effectiveness and damage calculation."""
    print("Test 2: Damage calculation")

    mcts = SimpleMCTS()

    # Electric vs Water (super effective)
    pikachu = SimplePokemon(
        species="Pikachu",
        hp=1.0,
        max_hp=95,
        atk=55,
        defense=40,
        spa=50,
        spd=50,
        spe=90,
        types=("Electric",),
        moves=["thunderbolt"],
    )

    blastoise = SimplePokemon(
        species="Blastoise",
        hp=1.0,
        max_hp=158,
        atk=83,
        defense=100,
        spa=85,
        spd=105,
        spe=78,
        types=("Water",),
        moves=["surf"],
    )

    damage = mcts.simulator.calculate_damage(pikachu, blastoise, "thunderbolt")
    print(f"  Pikachu Thunderbolt vs Blastoise: {damage:.2%} HP")
    assert damage > 0.10, f"Should do significant damage (super effective), got {damage:.2%}"
    assert damage < 0.50, f"Shouldn't OHKO a resistant target, got {damage:.2%}"
    print(f"  OK: Type effectiveness working\n")

    return True


def test_terminal_detection():
    """Test terminal state detection."""
    print("Test 3: Terminal state detection")

    fainted_mon = SimplePokemon(
        species="Pikachu",
        hp=0.0,
        max_hp=95,
        atk=55,
        defense=40,
        spa=50,
        spd=50,
        spe=90,
        types=("Electric",),
        moves=["thunderbolt"],
        fainted=True,
    )

    alive_mon = SimplePokemon(
        species="Charizard",
        hp=0.5,
        max_hp=153,
        atk=84,
        defense=78,
        spa=109,
        spd=85,
        spe=100,
        types=("Fire", "Flying"),
        moves=["flamethrower"],
        fainted=False,
    )

    # Test: all fainted = terminal
    state1 = SimpleState(
        own_team=[fainted_mon],
        opp_team=[alive_mon],
        own_active=0,
        opp_active=0,
    )

    assert state1.is_terminal(), "Should be terminal when we're all fainted"
    reward = state1.get_reward()
    assert reward < 0, "Should have negative reward when we lost"
    print(f"  OK: Terminal detection working (reward={reward:.2f})")

    # Test: both alive = not terminal
    state2 = SimpleState(
        own_team=[alive_mon],
        opp_team=[alive_mon],
        own_active=0,
        opp_active=0,
    )

    assert not state2.is_terminal(), "Should not be terminal when both alive"
    print(f"  OK: Non-terminal detection working\n")

    return True


if __name__ == "__main__":
    print("="*60)
    print("Simplified MCTS Quick Test")
    print("="*60 + "\n")

    tests = [
        test_damage_calculation,
        test_terminal_detection,
        test_basic_simulation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"ERROR: Test crashed: {e}\n")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("="*60)
    print("Results")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\nOK: All tests passed! Simplified MCTS is ready to use.")
        print("\nNext step: Test with actual model")
        print("  python test_mcts.py --model models/latest.pt --battles 20")
    else:
        print("\nERROR: Some tests failed. Check errors above.")
