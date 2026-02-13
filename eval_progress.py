"""Quick evaluation of multiple checkpoints to assess learning progress."""
import asyncio
import sys

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from battle_player import TrainedRLPlayer
from config import Config

CHECKPOINTS = [
    ("50K", "models/ppo_50176.pt"),
    ("1.5M", "models/ppo_1500160.pt"),
    ("3.2M", "models/ppo_3200000.pt"),
    ("5M", "models/ppo_4950016.pt"),
    ("6.5M", "models/ppo_6500352.pt"),
    ("best", "models/best.pt"),
]

N_BATTLES = 50
_counter = 0


async def eval_one(label: str, model_path: str, config: Config):
    global _counter

    results = {}
    for opp_name, opp_cls in [("random", RandomPlayer), ("heuristic", SimpleHeuristicsPlayer)]:
        _counter += 1
        tag = _counter

        rl = TrainedRLPlayer(
            model_path=model_path,
            config=config,
            deterministic=True,
            account_configuration=AccountConfiguration(f"E{tag}", None),
            battle_format=config.battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
        opp = opp_cls(
            account_configuration=AccountConfiguration(f"O{tag}", None),
            battle_format=config.battle_format,
            server_configuration=LocalhostServerConfiguration,
        )

        await rl.battle_against(opp, n_battles=N_BATTLES)
        wr = rl.n_won_battles / N_BATTLES
        results[opp_name] = wr

    return results


async def main():
    config = Config()

    print(f"{'Checkpoint':<12} {'vs Random':>12} {'vs Heuristic':>14}")
    print("-" * 40)

    for label, path in CHECKPOINTS:
        try:
            res = await eval_one(label, path, config)
            print(f"{label:<12} {res['random']:>11.1%} {res['heuristic']:>13.1%}")
        except Exception as e:
            print(f"{label:<12} ERROR: {e}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
