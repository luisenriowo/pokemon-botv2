import argparse
import asyncio
import copy
import os
import sys
import time

import numpy as np
import torch

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from agent import PPOAgent
from config import Config
from environment import Gen9Env, OBS_SIZE
from utils import compute_action_mask, save_checkpoint, load_checkpoint, setup_logging


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


def collect_rollout(env, agent: PPOAgent, config: Config, obs_dict: dict):
    """Collect rollout_steps env steps, returning per-agent buffers and latest obs.

    Each env step produces transitions for all active agents.
    We store them in separate buffers (one per agent) for correct GAE.
    """
    buffers = {name: agent.create_buffer(config.rollout_steps) for name in env.possible_agents}

    total_rewards = {name: 0.0 for name in env.possible_agents}
    episodes_done = 0
    episode_rewards = []

    for _ in range(config.rollout_steps):
        actions = {}
        step_data = {}

        for ag in env.agents:
            obs = obs_dict[ag]
            mask = env.current_masks.get(ag, compute_action_mask(env.current_battles[ag]))
            action, log_prob, value = agent.act(obs, mask)
            actions[ag] = np.int64(action)
            step_data[ag] = (obs, action, log_prob, value, mask)

        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        for ag, (obs, action, log_prob, value, mask) in step_data.items():
            reward = rewards.get(ag, 0.0)
            done = terminated.get(ag, False) or truncated.get(ag, False)
            buffers[ag].add(obs, action, log_prob, value, reward, done, mask)
            total_rewards[ag] += reward

        all_done = all(
            terminated.get(ag, False) or truncated.get(ag, False)
            for ag in env.possible_agents
        )

        if all_done:
            episodes_done += 1
            for ag in env.possible_agents:
                episode_rewards.append(total_rewards[ag])
                total_rewards[ag] = 0.0
            obs_dict, _ = env.reset()
        else:
            obs_dict = next_obs

    # Compute last values for bootstrapping
    for ag in env.possible_agents:
        if ag in obs_dict and ag in env.agents:
            mask = env.current_masks.get(ag, np.ones(config.action_space_size, dtype=np.float32))
            _, _, last_value = agent.act(obs_dict[ag], mask)
        else:
            last_value = 0.0
        buffers[ag].compute_gae(last_value, config.gamma, config.gae_lambda)

    return buffers, obs_dict, episodes_done, episode_rewards


def train(config: Config, resume_path: str = None):
    """Main self-play training loop."""
    # Update obs_size from environment
    config.obs_size = OBS_SIZE

    ppo = PPOAgent(config)
    writer = setup_logging(config.log_dir)

    start_timestep = 0
    if resume_path and os.path.exists(resume_path):
        start_timestep = load_checkpoint(resume_path, ppo.model, ppo.optimizer)
        print(f"Resumed from {resume_path} at timestep {start_timestep}")

    # Opponent pool for pool-based self-play
    opponent_pool = []

    env = make_env(config)
    obs_dict, _ = env.reset()

    timestep = start_timestep
    iteration = 0
    best_winrate = 0.0

    print(f"Training PPO agent for gen9randombattle")
    print(f"  Obs size: {config.obs_size}")
    print(f"  Action space: {config.action_space_size}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Rollout steps: {config.rollout_steps}")
    print(f"  Reward: sparse terminal (+1/-1/0)")
    print()

    while timestep < config.total_timesteps:
        t0 = time.time()
        progress = timestep / config.total_timesteps

        # Collect rollout
        buffers, obs_dict, episodes_done, episode_rewards = collect_rollout(
            env, ppo, config, obs_dict
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
            path = os.path.join(config.model_save_dir, f"ppo_{timestep}.pt")
            save_checkpoint(ppo.model, ppo.optimizer, timestep, path)

            latest_path = os.path.join(config.model_save_dir, "latest.pt")
            save_checkpoint(ppo.model, ppo.optimizer, timestep, latest_path)

        # Update opponent pool
        if timestep % config.opponent_update_freq < n_transitions:
            snapshot = copy.deepcopy(ppo.model.state_dict())
            if len(opponent_pool) >= config.opponent_pool_size:
                opponent_pool.pop(0)
            opponent_pool.append(snapshot)
            print(f"  Opponent pool updated (size={len(opponent_pool)})")

    # Final save
    final_path = os.path.join(config.model_save_dir, "final.pt")
    save_checkpoint(ppo.model, ppo.optimizer, timestep, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    env.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for gen9randombattle")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--port", type=int, default=8000, help="Showdown server port")
    args = parser.parse_args()

    config = Config()
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.port:
        config.server_port = args.port

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
