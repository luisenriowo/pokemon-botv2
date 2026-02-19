#!/usr/bin/env python
"""
Verify MCTS setup is complete and working.

Checks:
1. Node.js is installed
2. Showdown simulator is present
3. MCTS service compiles
4. Python dependencies are available
5. Basic IPC communication works
"""

import subprocess
import sys
import os
from pathlib import Path


def check_node():
    """Verify Node.js is installed."""
    print("Checking Node.js installation...")
    try:
        result = subprocess.run(
            ['node', '--version'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ Node.js {version}")
            return True
        else:
            print(f"  ✗ Node.js not found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_showdown():
    """Verify Showdown simulator is present."""
    print("\nChecking Showdown simulator...")
    showdown_path = Path("pokemon-showdown/.sim-dist/sim/battle.js")
    if showdown_path.exists():
        print(f"  ✓ Showdown simulator found at {showdown_path}")
        return True
    else:
        print(f"  ✗ Showdown simulator not found at {showdown_path}")
        print("    Run: cd pokemon-showdown && npm install && npm run build")
        return False


def check_mcts_service():
    """Verify MCTS service exists and has no syntax errors."""
    print("\nChecking MCTS service...")
    service_path = Path("mcts_service.js")
    if not service_path.exists():
        print(f"  ✗ {service_path} not found")
        return False

    try:
        # Just check if Node can parse the file (syntax check)
        result = subprocess.run(
            ['node', '--check', str(service_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"  ✓ {service_path} syntax OK")
            return True
        else:
            print(f"  ✗ Syntax error in {service_path}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_python_deps():
    """Verify Python dependencies."""
    print("\nChecking Python dependencies...")
    deps = ['numpy', 'torch', 'poke_env']
    all_ok = True

    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} not installed")
            all_ok = False

    return all_ok


def check_mcts_import():
    """Verify mcts.py can be imported."""
    print("\nChecking MCTS Python module...")
    try:
        import mcts
        print("  ✓ mcts.py imports successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error importing mcts.py: {e}")
        return False


def check_ipc():
    """Test basic IPC communication."""
    print("\nTesting MCTS IPC communication...")
    try:
        # Import after checking dependencies
        import numpy as np
        from mcts import MCTSEngine

        # Create minimal test state
        state = {
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
            ],
            "opp_team": [
                {
                    "species": "Charizard",
                    "level": 50,
                    "moves": ["flamethrower"],
                    "ability": None,
                    "item": None,
                    "hp_fraction": 1.0,
                    "status": None,
                    "boosts": {},
                    "active": True,
                    "fainted": False,
                    "revealed": True,
                    "gender": "M",
                },
            ],
            "opp_team_size": 6,
            "field": {"weather": None, "terrain": None, "trick_room": False},
            "side_conditions": {},
            "opp_side_conditions": {},
            "turn": 1,
        }

        priors = np.ones(26, dtype=np.float32) / 26
        value = 0.0
        action_mask = np.array([0]*6 + [1]*4 + [0]*16, dtype=np.float32)

        # Create engine and test search
        engine = MCTSEngine(verbose=False)
        try:
            best_action, visits, q_values = engine.search(state, priors, value, action_mask)
            print(f"  ✓ IPC test passed (action={best_action}, visits={visits.sum():.0f})")
            return True
        finally:
            engine.shutdown()

    except Exception as e:
        print(f"  ✗ IPC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("MCTS Setup Verification")
    print("="*60)

    checks = [
        ("Node.js", check_node),
        ("Showdown", check_showdown),
        ("MCTS Service", check_mcts_service),
        ("Python Deps", check_python_deps),
        ("MCTS Import", check_mcts_import),
        ("IPC Communication", check_ipc),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")

    print()
    if all_pass:
        print("✓ All checks passed! MCTS is ready to use.")
        print()
        print("Next steps:")
        print("  1. Re-train with fixed curriculum:")
        print("     python train.py --run A")
        print()
        print("  2. Test MCTS when policy base is strong:")
        print("     python test_mcts.py --model models/best.pt --battles 50")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
