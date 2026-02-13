import argparse
import asyncio

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)

from battle_player import TrainedRLPlayer
from config import Config


OPPONENTS = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


async def evaluate_against(
    model_path: str,
    opponent_type: str,
    n_battles: int,
    config: Config,
    deterministic: bool = True,
):
    """Evaluate trained model against a baseline opponent."""
    rl_player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=deterministic,
        account_configuration=AccountConfiguration("RLAgent", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    opp_cls = OPPONENTS[opponent_type]
    opponent = opp_cls(
        account_configuration=AccountConfiguration("Opponent", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"Evaluating vs {opponent_type} for {n_battles} battles...")
    await rl_player.battle_against(opponent, n_battles=n_battles)

    wins = rl_player.n_won_battles
    losses = rl_player.n_lost_battles
    ties = rl_player.n_tied_battles
    winrate = wins / max(n_battles, 1)

    print(f"Results vs {opponent_type}:")
    print(f"  Wins: {wins}  Losses: {losses}  Ties: {ties}")
    print(f"  Win rate: {winrate:.1%}")

    return {"wins": wins, "losses": losses, "ties": ties, "winrate": winrate}


async def full_evaluation(model_path: str, n_battles: int, config: Config):
    """Evaluate against all baseline opponents."""
    results = {}
    for opp_name in OPPONENTS:
        results[opp_name] = await evaluate_against(
            model_path, opp_name, n_battles, config
        )
        print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--battles", type=int, default=200, help="Number of battles per opponent")
    parser.add_argument(
        "--opponent",
        type=str,
        default="all",
        choices=["all", "random", "max_power", "heuristic"],
        help="Opponent type",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    args = parser.parse_args()

    config = Config()

    if args.opponent == "all":
        asyncio.run(full_evaluation(args.model, args.battles, config))
    else:
        asyncio.run(evaluate_against(
            args.model, args.opponent, args.battles, config,
            deterministic=not args.stochastic,
        ))


if __name__ == "__main__":
    main()
