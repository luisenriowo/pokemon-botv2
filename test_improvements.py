"""
Quick test to compare MCTS versions.

Tests:
- Policy only (baseline)
- Simple MCTS (baseline MCTS)
- Improved MCTS (with all enhancements)

Expected improvement: ~5-9% for improved over simple MCTS
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


async def test_configuration(model_path: str, config: Config, n_battles: int,
                             tag: str, use_mcts: bool, mcts_mode: str = None):
    """Test a single configuration."""
    mode_name = "Policy" if not use_mcts else f"MCTS-{mcts_mode}"

    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=True,
        use_mcts=use_mcts,
        mcts_mode=mcts_mode or 'simple',
        mcts_rollouts=200,
        account_configuration=AccountConfiguration(f"Test_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"Opp_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"\nTesting {mode_name} ({n_battles} battles)...")
    await player.battle_against(opponent, n_battles=n_battles)

    wins = player.n_won_battles
    losses = player.n_lost_battles
    winrate = wins / n_battles

    print(f"  {mode_name}: {wins}/{n_battles} wins ({winrate:.1%})")

    return {
        "mode": mode_name,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
    }


async def main():
    parser = argparse.ArgumentParser(description="Compare MCTS improvements")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--battles", type=int, default=50,
                       help="Battles per configuration")
    args = parser.parse_args()

    config = Config()

    print("="*70)
    print("MCTS Improvements Comparison Test")
    print("="*70)

    results = []

    # 1. Baseline: Policy only
    results.append(await test_configuration(
        args.model, config, args.battles, "pol", use_mcts=False
    ))

    # 2. Simple MCTS (baseline)
    results.append(await test_configuration(
        args.model, config, args.battles, "simple", use_mcts=True, mcts_mode="simple"
    ))

    # 3. Improved MCTS (with all enhancements)
    results.append(await test_configuration(
        args.model, config, args.battles, "improved", use_mcts=True, mcts_mode="improved"
    ))

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Mode':<20} {'Wins':>8} {'Losses':>8} {'Win Rate':>12} {'vs Policy':>12}")
    print("-"*70)

    policy_wr = results[0]['winrate']
    for r in results:
        improvement = r['winrate'] - policy_wr
        improvement_str = f"+{improvement:.1%}" if improvement > 0 else f"{improvement:.1%}"
        print(f"{r['mode']:<20} {r['wins']:>8} {r['losses']:>8} {r['winrate']:>11.1%} "
              f"{improvement_str:>12}")

    print()

    # Specific comparison: Simple vs Improved
    simple_wr = results[1]['winrate']
    improved_wr = results[2]['winrate']
    mcts_improvement = improved_wr - simple_wr
    mcts_improvement_pct = (mcts_improvement / max(simple_wr, 0.01)) * 100

    print("Key Comparisons:")
    print(f"  Simple MCTS improvement:   {results[1]['winrate'] - policy_wr:+.1%} "
          f"({((results[1]['winrate'] - policy_wr) / max(policy_wr, 0.01) * 100):+.1f}% relative)")
    print(f"  Improved MCTS improvement: {improved_wr - policy_wr:+.1%} "
          f"({((improved_wr - policy_wr) / max(policy_wr, 0.01) * 100):+.1f}% relative)")
    print(f"  Improvements add:          {mcts_improvement:+.1%} "
          f"({mcts_improvement_pct:+.1f}% over simple MCTS)")
    print()

    if mcts_improvement >= 0.03:
        print("SUCCESS: Improvements are working well! (+3% or more)")
    elif mcts_improvement >= 0.01:
        print("GOOD: Improvements are helping (+1-3%)")
    else:
        print("NOTE: Improvements are small (<1%) - may need more battles for statistical significance")


if __name__ == "__main__":
    asyncio.run(main())
