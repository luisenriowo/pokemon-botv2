import argparse
import asyncio
import copy
import os
import random
import sys
import time

import numpy as np
import torch

from poke_env import AccountConfiguration, LocalhostServerConfiguration, SimpleHeuristicsPlayer

from agent import PPOAgent
from battle_player import TrainedRLPlayer
from config import Config
from environment import Gen9Env, OBS_SIZE
from model import ActorCritic
from utils import compute_action_mask, save_checkpoint, load_checkpoint, setup_logging


_eval_counter = 0


def evaluate_vs_heuristic(model_path: str, config: Config, n_battles: int = 50) -> float:
    """Evaluate the current model against SimpleHeuristicsPlayer.

    Runs in a fresh event loop (safe to call from sync training code).
    Returns win rate as a float in [0, 1].
    """
    global _eval_counter
    _eval_counter += 1
    tag = _eval_counter

    async def _eval():
        rl_player = TrainedRLPlayer(
            model_path=model_path,
            config=config,
            deterministic=True,
            account_configuration=AccountConfiguration(f"Eval{tag}", None),
            battle_format=config.battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(f"Heur{tag}", None),
            battle_format=config.battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
        await rl_player.battle_against(opponent, n_battles=n_battles)
        return rl_player.n_won_battles / max(n_battles, 1)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_eval())
    finally:
        loop.close()


def make_env(config: Config, p1_name: str = "PPOPlayer1", p2_name: str = "PPOPlayer2"):
    """Create a Gen9Env for self-play training."""
    env = Gen9Env(
        config=config,
        account_configuration1=AccountConfiguration(p1_name, None),
        account_configuration2=AccountConfiguration(p2_name, None),
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
    )
    return env


def collect_rollout(env, agent: PPOAgent, config: Config, obs_dict: dict,
                     opponent_model: ActorCritic = None):
    """Collect rollout_steps env steps, returning per-agent buffers and latest obs.

    Each env step produces transitions for all active agents.
    We store them in separate buffers (one per agent) for correct GAE.

    If opponent_model is provided, player 2 uses that frozen model for action
    selection and only player 1's buffer is used for training.
    """
    agents_list = env.possible_agents
    p1, p2 = agents_list[0], agents_list[1]

    # When using opponent pool, only train on p1 data
    train_agents = {p1} if opponent_model else set(agents_list)
    buffers = {name: agent.create_buffer(config.rollout_steps) for name in train_agents}

    total_rewards = {name: 0.0 for name in agents_list}
    episodes_done = 0
    episode_rewards = []

    for _ in range(config.rollout_steps):
        if not env.agents:
            obs_dict, _ = env.reset()
            continue

        actions = {}
        step_data = {}

        for ag in env.agents:
            obs = obs_dict[ag]
            mask = env.current_masks.get(ag, compute_action_mask(env.current_battles[ag]))

            if ag == p2 and opponent_model is not None:
                action, log_prob, value = opponent_model.act(obs, mask)
            else:
                action, log_prob, value = agent.act(obs, mask)

            actions[ag] = np.int64(action)
            step_data[ag] = (obs, action, log_prob, value, mask)

        # poke-env can mark the battle finished between obs and step (race condition)
        try:
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
        except (AssertionError, Exception):
            episodes_done += 1
            for ag in train_agents:
                episode_rewards.append(total_rewards[ag])
                total_rewards[ag] = 0.0
            obs_dict, _ = env.reset()
            continue

        for ag, (obs, action, log_prob, value, mask) in step_data.items():
            if ag not in train_agents:
                continue
            reward = rewards.get(ag, 0.0)
            done = terminated.get(ag, False) or truncated.get(ag, False)
            buffers[ag].add(obs, action, log_prob, value, reward, done, mask)
            total_rewards[ag] += reward

        all_done = not env.agents or all(
            terminated.get(ag, False) or truncated.get(ag, False)
            for ag in agents_list
        )

        if all_done:
            episodes_done += 1
            for ag in train_agents:
                episode_rewards.append(total_rewards[ag])
                total_rewards[ag] = 0.0
            obs_dict, _ = env.reset()
        else:
            obs_dict = next_obs

    # Compute last values for bootstrapping
    for ag in train_agents:
        if ag in obs_dict and ag in env.agents:
            mask = env.current_masks.get(ag, np.ones(config.action_space_size, dtype=np.float32))
            _, _, last_value = agent.act(obs_dict[ag], mask)
        else:
            last_value = 0.0
        buffers[ag].compute_gae(last_value, config.gamma, config.gae_lambda)

    return buffers, obs_dict, episodes_done, episode_rewards


def train(config: Config, resume_path: str = None, run_name: str = "A"):
    """Main self-play training loop."""
    # Update obs_size from environment
    config.obs_size = OBS_SIZE

    # Separate dirs per run so parallel training doesn't collide
    model_dir = os.path.join(config.model_save_dir, run_name)
    log_dir = os.path.join(config.log_dir, run_name)

    ppo = PPOAgent(config)
    device = ppo.device
    writer = setup_logging(log_dir)

    start_timestep = 0
    if resume_path and os.path.exists(resume_path):
        start_timestep = load_checkpoint(resume_path, ppo.model, ppo.optimizer)
        print(f"Resumed from {resume_path} at timestep {start_timestep}")

    # Opponent pool for pool-based self-play
    opponent_pool = []

    # Unique player names per run to avoid Showdown name collisions
    env = make_env(config, p1_name=f"P1{run_name}", p2_name=f"P2{run_name}")

    timestep = start_timestep
    iteration = 0
    best_winrate = 0.0

    print(f"Training PPO agent for gen9randombattle (run={run_name})")
    print(f"  Device: {device}")
    print(f"  Obs size: {config.obs_size}")
    print(f"  Action space: {config.action_space_size}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Rollout steps: {config.rollout_steps}")
    print(f"  Reward: sparse terminal (+1/-1/0)")
    print()

    # Frozen opponent model for pool-based self-play
    opponent_model = None
    if config.use_opponent_pool:
        opponent_model = ActorCritic(
            obs_dim=config.obs_size,
            action_dim=config.action_space_size,
            hidden_sizes=config.hidden_sizes,
            head_hidden=config.head_hidden,
            species_vocab=config.species_vocab,
            move_vocab=config.move_vocab,
            ability_vocab=config.ability_vocab,
            item_vocab=config.item_vocab,
            species_embed_dim=config.species_embed_dim,
            move_embed_dim=config.move_embed_dim,
            ability_embed_dim=config.ability_embed_dim,
            item_embed_dim=config.item_embed_dim,
        ).to(device)
        opponent_model.eval()

    # Start first battle right before entering the loop (minimize idle time)
    obs_dict, _ = env.reset()

    while timestep < config.total_timesteps:
        t0 = time.time()
        progress = timestep / config.total_timesteps

        # Select opponent from pool (or None → self-play with current model)
        active_opponent = None
        if config.use_opponent_pool and opponent_pool and opponent_model is not None:
            snapshot = random.choice(opponent_pool)
            opponent_model.load_state_dict(snapshot)
            active_opponent = opponent_model

        # Collect rollout
        buffers, obs_dict, episodes_done, episode_rewards = collect_rollout(
            env, ppo, config, obs_dict, opponent_model=active_opponent
        )

        # Count transitions collected
        n_transitions = sum(b.size for b in buffers.values())
        timestep += n_transitions

        # PPO update
        active_buffers = [b for b in buffers.values() if b.size > 0]
        if active_buffers:
            stats = ppo.update(active_buffers, progress=progress)
        else:
            stats = {"pg_loss": 0, "v_loss": 0, "entropy": 0}

        elapsed = time.time() - t0
        fps = n_transitions / max(elapsed, 1e-6)

        # Logging
        writer.add_scalar("train/pg_loss", stats["pg_loss"], timestep)
        writer.add_scalar("train/v_loss", stats["v_loss"], timestep)
        writer.add_scalar("train/entropy", stats["entropy"], timestep)
        writer.add_scalar("train/fps", fps, timestep)
        writer.add_scalar("train/episodes", episodes_done, timestep)

        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            writer.add_scalar("train/mean_episode_reward", mean_reward, timestep)
            win_count = sum(1 for r in episode_rewards if r > 0)
            winrate = win_count / len(episode_rewards)
            writer.add_scalar("train/winrate", winrate, timestep)

        iteration += 1
        if iteration % 1 == 0:
            ep_info = ""
            if episode_rewards:
                wins = sum(1 for r in episode_rewards if r > 0)
                ep_info = f" | ep={episodes_done} wins={wins}/{len(episode_rewards)}"
            print(
                f"[{timestep:>9d}/{config.total_timesteps}] "
                f"pg={stats['pg_loss']:.4f} v={stats['v_loss']:.4f} "
                f"ent={stats['entropy']:.4f} fps={fps:.0f}{ep_info}"
            )

        # Save checkpoint
        if timestep % config.save_freq < n_transitions:
            path = os.path.join(model_dir, f"ppo_{timestep}.pt")
            save_checkpoint(ppo.model, ppo.optimizer, timestep, path)

            latest_path = os.path.join(model_dir, "latest.pt")
            save_checkpoint(ppo.model, ppo.optimizer, timestep, latest_path)

        # Periodic evaluation vs SimpleHeuristicsPlayer
        if timestep % config.eval_freq < n_transitions:
            eval_path = os.path.join(model_dir, "latest.pt")
            if os.path.exists(eval_path):
                try:
                    winrate = evaluate_vs_heuristic(
                        eval_path, config, n_battles=config.eval_battles
                    )
                    writer.add_scalar("eval/vs_heuristic_winrate", winrate, timestep)
                    print(f"  [Eval] vs Heuristic: {winrate:.1%} ({config.eval_battles} battles)")
                    if winrate > best_winrate:
                        best_winrate = winrate
                        best_path = os.path.join(model_dir, "best.pt")
                        save_checkpoint(ppo.model, ppo.optimizer, timestep, best_path)
                        print(f"  [Eval] New best model saved ({winrate:.1%})")
                except Exception as e:
                    print(f"  [Eval] Failed: {e}")

        # Update opponent pool
        if timestep % config.opponent_update_freq < n_transitions:
            snapshot = copy.deepcopy(ppo.model.state_dict())
            if len(opponent_pool) >= config.opponent_pool_size:
                opponent_pool.pop(0)
            opponent_pool.append(snapshot)
            print(f"  Opponent pool updated (size={len(opponent_pool)})")

    # Final save
    final_path = os.path.join(model_dir, "final.pt")
    save_checkpoint(ppo.model, ppo.optimizer, timestep, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    env.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for gen9randombattle")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--port", type=int, default=8000, help="Showdown server port")
    parser.add_argument("--name", type=str, default="A", help="Run name (unique per parallel run)")
    args = parser.parse_args()

    config = Config()
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.port:
        config.server_port = args.port

    train(config, resume_path=args.resume, run_name=args.name)


if __name__ == "__main__":
    main()
