import argparse
import asyncio

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from battle_player import TrainedRLPlayer
from config import Config


async def local_accept(model_path: str, config: Config, deterministic: bool):
    """Accept challenges on local Showdown server."""
    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=deterministic,
        account_configuration=AccountConfiguration("AntiMateo", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"Waiting for challenges on local server as 'AntiMateo'...")
    print(f"Format: {config.battle_format}")
    print(f"Mode: {'deterministic' if deterministic else 'stochastic'}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            await player.accept_challenges(None, 1)
            print(f"Battle finished. W/L/T: {player.n_won_battles}/{player.n_lost_battles}/{player.n_tied_battles}")
    except KeyboardInterrupt:
        print("\nStopped.")


async def local_challenge(model_path: str, opponent: str, config: Config, deterministic: bool, n_games: int):
    """Challenge a specific user on local server."""
    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=deterministic,
        account_configuration=AccountConfiguration("AntiMateo", None),
        battle_format=config.battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"Challenging '{opponent}' on local server...")
    for i in range(n_games):
        await player.send_challenges(opponent, 1)
        print(f"  Game {i+1}/{n_games}: W/L/T = {player.n_won_battles}/{player.n_lost_battles}/{player.n_tied_battles}")

    print(f"\nFinal: {player.n_won_battles}W / {player.n_lost_battles}L / {player.n_tied_battles}T")


async def online_ladder(model_path: str, username: str, password: str, config: Config, deterministic: bool, n_games: int):
    """Play on the official Showdown ladder."""
    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=deterministic,
        account_configuration=AccountConfiguration(username, password),
        battle_format=config.battle_format,
        server_configuration=ShowdownServerConfiguration,
    )

    print(f"Playing ladder on Showdown as '{username}'...")
    print(f"Format: {config.battle_format}")
    print(f"Games: {n_games}")
    print()

    await player.ladder(n_games)

    print(f"\nLadder results: {player.n_won_battles}W / {player.n_lost_battles}L / {player.n_tied_battles}T")


async def online_challenge(model_path: str, opponent: str, username: str, password: str, config: Config, deterministic: bool, n_games: int):
    """Challenge a specific user on official Showdown."""
    player = TrainedRLPlayer(
        model_path=model_path,
        config=config,
        deterministic=deterministic,
        account_configuration=AccountConfiguration(username, password),
        battle_format=config.battle_format,
        server_configuration=ShowdownServerConfiguration,
    )

    print(f"Challenging '{opponent}' on Showdown as '{username}'...")
    for i in range(n_games):
        await player.send_challenges(opponent, 1)
        print(f"  Game {i+1}/{n_games}: W/L/T = {player.n_won_battles}/{player.n_lost_battles}/{player.n_tied_battles}")

    print(f"\nFinal: {player.n_won_battles}W / {player.n_lost_battles}L / {player.n_tied_battles}T")


def main():
    parser = argparse.ArgumentParser(description="Play Pokemon battles with trained model")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Local subcommand
    local = subparsers.add_parser("local", help="Play on local Showdown server")
    local_sub = local.add_subparsers(dest="action", required=True)

    accept = local_sub.add_parser("accept", help="Accept incoming challenges")
    accept.add_argument("--model", type=str, required=True)
    accept.add_argument("--stochastic", action="store_true")

    challenge = local_sub.add_parser("challenge", help="Challenge a specific user")
    challenge.add_argument("opponent", type=str)
    challenge.add_argument("--model", type=str, required=True)
    challenge.add_argument("--games", type=int, default=1)
    challenge.add_argument("--stochastic", action="store_true")

    # Online subcommand
    online = subparsers.add_parser("online", help="Play on official Showdown")
    online_sub = online.add_subparsers(dest="action", required=True)

    ladder = online_sub.add_parser("ladder", help="Play on ladder")
    ladder.add_argument("--model", type=str, required=True)
    ladder.add_argument("--username", type=str, required=True)
    ladder.add_argument("--password", type=str, required=True)
    ladder.add_argument("--games", type=int, default=50)
    ladder.add_argument("--stochastic", action="store_true")

    online_chal = online_sub.add_parser("challenge", help="Challenge a specific user")
    online_chal.add_argument("opponent", type=str)
    online_chal.add_argument("--model", type=str, required=True)
    online_chal.add_argument("--username", type=str, required=True)
    online_chal.add_argument("--password", type=str, required=True)
    online_chal.add_argument("--games", type=int, default=1)
    online_chal.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()
    config = Config()
    det = not getattr(args, "stochastic", False)

    if args.mode == "local":
        if args.action == "accept":
            asyncio.run(local_accept(args.model, config, det))
        elif args.action == "challenge":
            asyncio.run(local_challenge(args.model, args.opponent, config, det, args.games))
    elif args.mode == "online":
        if args.action == "ladder":
            asyncio.run(online_ladder(args.model, args.username, args.password, config, det, args.games))
        elif args.action == "challenge":
            asyncio.run(online_challenge(args.model, args.opponent, args.username, args.password, config, det, args.games))


if __name__ == "__main__":
    main()
