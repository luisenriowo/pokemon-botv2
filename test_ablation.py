"""
Ablation study to identify which improvement is hurting performance.

Tests each improvement individually and in combination.
"""

import argparse
import asyncio

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    SimpleHeuristicsPlayer,
)

from battle_player import TrainedRLPlayer
from config import Config


async def test_config(model_path: str, config: Config, n_battles: int,
                     tag: str, name: str, **mcts_kwargs):
    """Test a single configuration."""
    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=True,
        **mcts_kwargs,
        account_configuration=AccountConfiguration(f"Test_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"Opp_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"\nTesting {name} ({n_battles} battles)...")
    await player.battle_against(opponent, n_battles=n_battles)

    wins = player.n_won_battles
    winrate = wins / n_battles

    print(f"  {name}: {wins}/{n_battles} wins ({winrate:.1%})")

    return {
        "name": name,
        "wins": wins,
        "losses": player.n_lost_battles,
        "winrate": winrate,
    }


async def main():
    parser = argparse.ArgumentParser(description="MCTS Ablation Study")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--battles", type=int, default=100,
                       help="Battles per configuration")
    args = parser.parse_args()

    config = Config()

    print("="*70)
    print("MCTS Improvements Ablation Study")
    print("="*70)
    print("\nTesting each improvement individually to find the culprit...")

    results = []

    # Baseline: Policy only
    results.append(await test_config(
        args.model, config, args.battles, "pol", "Policy",
        use_mcts=False
    ))

    # Baseline: Simple MCTS
    results.append(await test_config(
        args.model, config, args.battles, "simple", "Simple MCTS",
        use_mcts=True,
        mcts_mode='simple',
        mcts_rollouts=200,
    ))

    # Test each improvement individually (all others OFF)
    print("\n" + "="*70)
    print("Individual Improvements (one at a time)")
    print("="*70)

    results.append(await test_config(
        args.model, config, args.battles, "smart", "Smart Opponent ONLY",
        use_mcts=True,
        mcts_mode='improved',
        mcts_rollouts=200,
        mcts_smart_opponent=True,
        mcts_value_bootstrap=False,
        mcts_move_ordering=False,
        mcts_adaptive_rollouts=False,
    ))

    results.append(await test_config(
        args.model, config, args.battles, "bootstrap", "Value Bootstrap ONLY",
        use_mcts=True,
        mcts_mode='improved',
        mcts_rollouts=200,
        mcts_smart_opponent=False,
        mcts_value_bootstrap=True,
        mcts_move_ordering=False,
        mcts_adaptive_rollouts=False,
    ))

    results.append(await test_config(
        args.model, config, args.battles, "ordering", "Move Ordering ONLY",
        use_mcts=True,
        mcts_mode='improved',
        mcts_rollouts=200,
        mcts_smart_opponent=False,
        mcts_value_bootstrap=False,
        mcts_move_ordering=True,
        mcts_adaptive_rollouts=False,
    ))

    results.append(await test_config(
        args.model, config, args.battles, "adaptive", "Adaptive Rollouts ONLY",
        use_mcts=True,
        mcts_mode='improved',
        mcts_rollouts=200,
        mcts_smart_opponent=False,
        mcts_value_bootstrap=False,
        mcts_move_ordering=False,
        mcts_adaptive_rollouts=True,
    ))

    # All improvements
    results.append(await test_config(
        args.model, config, args.battles, "all", "All Improvements",
        use_mcts=True,
        mcts_mode='improved',
        mcts_rollouts=200,
        mcts_smart_opponent=True,
        mcts_value_bootstrap=True,
        mcts_move_ordering=True,
        mcts_adaptive_rollouts=True,
    ))

    # Summary
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'Configuration':<30} {'Win Rate':>12} {'vs Policy':>12} {'vs Simple':>12}")
    print("-"*70)

    policy_wr = results[0]['winrate']
    simple_wr = results[1]['winrate']

    for r in results:
        vs_policy = r['winrate'] - policy_wr
        vs_simple = r['winrate'] - simple_wr

        vs_policy_str = f"{vs_policy:+.1%}" if vs_policy != 0 else "-"
        vs_simple_str = f"{vs_simple:+.1%}" if vs_simple != 0 else "-"

        print(f"{r['name']:<30} {r['winrate']:>11.1%} {vs_policy_str:>12} {vs_simple_str:>12}")

    print()

    # Identify culprits
    print("Analysis:")
    print("-" * 70)

    individual_results = results[2:6]  # Smart, Bootstrap, Ordering, Adaptive
    improvement_names = [
        "Smart Opponent",
        "Value Bootstrap",
        "Move Ordering",
        "Adaptive Rollouts"
    ]

    for i, (r, name) in enumerate(zip(individual_results, improvement_names)):
        diff_vs_simple = r['winrate'] - simple_wr
        if diff_vs_simple < -0.02:  # More than 2% worse
            print(f"  CULPRIT: {name} hurts by {diff_vs_simple:.1%}")
        elif diff_vs_simple > 0.02:  # More than 2% better
            print(f"  GOOD: {name} helps by {diff_vs_simple:+.1%}")
        else:
            print(f"  NEUTRAL: {name} ({diff_vs_simple:+.1%})")

    print()


if __name__ == "__main__":
    asyncio.run(main())
