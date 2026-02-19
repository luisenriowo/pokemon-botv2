"""
Test MCTS integration.

Compares policy-only vs MCTS-enhanced performance against SimpleHeuristicsPlayer.
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


async def evaluate_player(
    model_path: str,
    use_mcts: bool,
    mcts_mode: str,
    mcts_rollouts: int,
    n_battles: int,
    config: Config,
    tag: str,
):
    """Evaluate a player configuration."""
    if use_mcts:
        mode = f"MCTS-{mcts_mode}" if use_mcts else "Policy"
    else:
        mode = "Policy"

    rl_player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=True,
        use_mcts=use_mcts,
        mcts_mode=mcts_mode,
        mcts_rollouts=mcts_rollouts,
        mcts_verbose=False,
        account_configuration=AccountConfiguration(f"RL_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"Heur_{tag}", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"\n{'='*60}")
    print(f"Evaluating {mode} mode vs SimpleHeuristicsPlayer")
    print(f"Battles: {n_battles}")
    print(f"{'='*60}\n")

    await rl_player.battle_against(opponent, n_battles=n_battles)

    wins = rl_player.n_won_battles
    losses = rl_player.n_lost_battles
    ties = rl_player.n_tied_battles
    winrate = wins / max(n_battles, 1)

    print(f"\n{mode} Results:")
    print(f"  Wins:     {wins}")
    print(f"  Losses:   {losses}")
    print(f"  Ties:     {ties}")
    print(f"  Win rate: {winrate:.1%}")
    print()

    return {"mode": mode, "wins": wins, "losses": losses, "ties": ties, "winrate": winrate}


async def main():
    parser = argparse.ArgumentParser(description="Test MCTS integration")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--battles", type=int, default=50, help="Battles per configuration")
    parser.add_argument("--mcts-mode", type=str, default="improved",
                       choices=["simple", "improved", "full"],
                       help="MCTS mode: 'simple' (baseline), 'improved' (with enhancements), or 'full' (Node.js Showdown)")
    parser.add_argument("--rollouts", type=int, default=200, help="MCTS rollouts per action")
    args = parser.parse_args()

    config = Config()

    # Test both configurations
    results = []

    # 1. Policy-only baseline
    policy_result = await evaluate_player(
        args.model,
        use_mcts=False,
        mcts_mode=args.mcts_mode,
        mcts_rollouts=args.rollouts,
        n_battles=args.battles,
        config=config,
        tag="pol",
    )
    results.append(policy_result)

    # 2. MCTS-enhanced
    mcts_result = await evaluate_player(
        args.model,
        use_mcts=True,
        mcts_mode=args.mcts_mode,
        mcts_rollouts=args.rollouts,
        n_battles=args.battles,
        config=config,
        tag="mcts",
    )
    results.append(mcts_result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<15} {'Win Rate':>12} {'Wins':>8} {'Losses':>8} {'Ties':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['mode']:<15} {r['winrate']:>11.1%} {r['wins']:>8} {r['losses']:>8} {r['ties']:>8}")
    print()

    # Calculate improvement
    policy_wr = results[0]['winrate']
    mcts_wr = results[1]['winrate']
    improvement = mcts_wr - policy_wr
    improvement_pct = (improvement / max(policy_wr, 0.01)) * 100

    print(f"MCTS Improvement: {improvement:+.1%} ({improvement_pct:+.1f}% relative)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
