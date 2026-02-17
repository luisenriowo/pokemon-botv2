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
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType

from agent import PPOAgent
from battle_player import TrainedRLPlayer
from config import Config
from environment import Gen9Env, OBS_SIZE, SPECIES_VOCAB, MOVE_VOCAB, ABILITY_VOCAB, ITEM_VOCAB
from model import ActorCritic
from utils import compute_action_mask, save_checkpoint, load_checkpoint, setup_logging


_eval_counter = 0


def evaluate_vs_heuristic(model_path: str, config: Config, n_battles: int = 50,
                          run_name: str = "A") -> float:
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
            account_configuration=AccountConfiguration(f"Ev{run_name}{tag}", None),
            battle_format=config.battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(f"He{run_name}{tag}", None),
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


def _max_power_action(battle, mask):
    """MaxBasePowerPlayer logic: pick the highest base power legal move."""
    # poke-env action space: 0-5=switch, 6-9=move, 10-13=mega, 14-17=zmove, 18-21=dmax, 22-25=tera
    best_action = -1
    best_power = -1
    for i, move in enumerate(battle.available_moves[:4]):
        action_idx = 6 + i  # regular moves start at index 6
        if action_idx < 26 and mask[action_idx] > 0 and move.base_power > best_power:
            best_power = move.base_power
            best_action = action_idx
    if best_action >= 0:
        return best_action
    legal = np.where(mask > 0)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


def _stat_estimation(mon, stat: str) -> float:
    """Estimate effective stat value accounting for boosts."""
    base = mon.base_stats.get(stat, 100)
    boost = mon.boosts.get(stat, 0)
    if boost > 0:
        return base * (2 + boost) / 2
    elif boost < 0:
        return base * 2 / (2 - boost)
    return float(base)


_SWITCH_OUT_MATCHUP_THRESHOLD = -2


def _estimate_matchup(mon, opponent) -> float:
    """Estimate type-based matchup score (positive = mon has advantage).

    Replicates SimpleHeuristicsPlayer._estimate_matchup logic:
    offensive - defensive + speed_bonus + hp_bonus
    """
    # Offensive: best effectiveness of our STAB types vs opponent
    offensive = 1.0
    for t in mon.types:
        if t is not None and t != PokemonType.THREE_QUESTION_MARKS:
            try:
                eff = opponent.damage_multiplier(t)
                offensive = max(offensive, eff)
            except Exception:
                pass

    # Defensive: best effectiveness of opponent's STAB types vs us
    defensive = 1.0
    for t in opponent.types:
        if t is not None and t != PokemonType.THREE_QUESTION_MARKS:
            try:
                eff = mon.damage_multiplier(t)
                defensive = max(defensive, eff)
            except Exception:
                pass

    # Speed tier bonus
    speed_bonus = 0.1 if mon.base_stats.get("spe", 0) > opponent.base_stats.get("spe", 0) else -0.1

    # HP advantage
    hp_bonus = (mon.current_hp_fraction - opponent.current_hp_fraction) * 0.4

    return offensive - defensive + speed_bonus + hp_bonus


def _should_switch_out(battle) -> bool:
    """Decide if we should switch out (mirrors SimpleHeuristicsPlayer logic)."""
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    if not active or not opp or not battle.available_switches:
        return False

    # Check if any switch-in has a positive matchup
    has_good_switch = any(
        _estimate_matchup(m, opp) > 0 for m in battle.available_switches
    )
    if not has_good_switch:
        return False

    # Switch on severe stat drops
    boosts = active.boosts
    if boosts.get("def", 0) <= -3 or boosts.get("spd", 0) <= -3:
        return True
    if boosts.get("atk", 0) <= -3 and active.base_stats.get("atk", 0) >= active.base_stats.get("spa", 0):
        return True
    if boosts.get("spa", 0) <= -3 and active.base_stats.get("spa", 0) >= active.base_stats.get("atk", 0):
        return True

    # Switch on bad type matchup
    if _estimate_matchup(active, opp) < _SWITCH_OUT_MATCHUP_THRESHOLD:
        return True

    return False


def _switch_action_index(battle, switch_mon) -> int:
    """Get the action index (0-5) for switching to a given Pokemon."""
    team_list = list(battle.team.values())
    for idx, mon in enumerate(team_list):
        if mon.base_species == switch_mon.base_species:
            return idx
    return -1


def _heuristic_action(battle, mask):
    """Heuristic opponent matching SimpleHeuristicsPlayer's key logic:
    switching on bad matchups, stat-aware damage estimation, accuracy.

    poke-env action space: 0-5=switch, 6-9=move, 10-13=mega, 14-17=zmove, 18-21=dmax, 22-25=tera
    """
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    if not active or not opp:
        legal = np.where(mask > 0)[0]
        return int(np.random.choice(legal)) if len(legal) > 0 else 0

    # 1. Check if we should switch out
    if _should_switch_out(battle):
        best_switch_action = -1
        best_switch_score = -float('inf')
        for switch_mon in battle.available_switches:
            action_idx = _switch_action_index(battle, switch_mon)
            if action_idx < 0 or action_idx >= 6 or mask[action_idx] == 0:
                continue
            score = _estimate_matchup(switch_mon, opp)
            if score > best_switch_score:
                best_switch_score = score
                best_switch_action = action_idx
        if best_switch_action >= 0:
            return best_switch_action

    # 2. Score moves with stat-aware damage estimation
    phys_ratio = _stat_estimation(active, "atk") / max(_stat_estimation(opp, "def"), 1)
    spec_ratio = _stat_estimation(active, "spa") / max(_stat_estimation(opp, "spd"), 1)

    best_action = -1
    best_score = -1.0

    for i, move in enumerate(battle.available_moves[:4]):
        action_idx = 6 + i  # regular moves start at index 6
        if action_idx >= 26 or mask[action_idx] == 0:
            continue

        bp = float(move.base_power)
        if bp == 0:
            score = 30.0  # status moves get a small base score
        else:
            score = bp
            # STAB
            if move.type in active.types:
                score *= 1.5
            # Type effectiveness
            try:
                score *= opp.damage_multiplier(move)
            except Exception:
                pass
            # Stat ratio
            if move.category == MoveCategory.PHYSICAL:
                score *= phys_ratio
            elif move.category == MoveCategory.SPECIAL:
                score *= spec_ratio
            # Accuracy
            acc = move.accuracy
            if isinstance(acc, bool):
                acc = 1.0
            elif acc is not None and acc > 1.0:
                acc = acc / 100.0
            else:
                acc = acc if acc else 1.0
            score *= acc

        if score > best_score:
            best_score = score
            best_action = action_idx

    if best_action >= 0:
        return best_action

    legal = np.where(mask > 0)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


def collect_rollout(env, agent: PPOAgent, config: Config, obs_dict: dict,
                     opponent_model: ActorCritic = None,
                     curriculum_fn=None):
    """Collect rollout_steps env steps, returning per-agent buffers and latest obs.

    Each env step produces transitions for all active agents.
    We store them in separate buffers (one per agent) for correct GAE.

    If opponent_model is provided, player 2 uses that frozen model for action
    selection and only player 1's buffer is used for training.
    If curriculum_fn is provided, player 2 uses that function (takes battle, mask)
    and only player 1's buffer is used for training.
    """
    agents_list = env.possible_agents
    p1, p2 = agents_list[0], agents_list[1]

    # When using curriculum or opponent pool, only train on p1 data
    train_agents = {p1} if (opponent_model or curriculum_fn) else set(agents_list)
    buffers = {name: agent.create_buffer(config.rollout_steps) for name in train_agents}

    total_rewards = {name: 0.0 for name in agents_list}
    episodes_done = 0
    episode_rewards = []
    episode_wins = 0

    for _ in range(config.rollout_steps):
        if not env.agents:
            obs_dict, _ = env.reset()
            continue

        actions = {}
        step_data = {}

        for ag in env.agents:
            obs = obs_dict[ag]
            mask = env.current_masks.get(ag, compute_action_mask(env.current_battles[ag]))

            if ag == p2 and curriculum_fn is not None:
                battle = env.current_battles[ag]
                action = curriculum_fn(battle, mask)
                log_prob, value = 0.0, 0.0
            elif ag == p2 and opponent_model is not None:
                action, log_prob, value = opponent_model.act(obs, mask)
            else:
                action, log_prob, value = agent.act(obs, mask)

            actions[ag] = np.int64(action)
            step_data[ag] = (obs, action, log_prob, value, mask)

        # poke-env can mark the battle finished between obs and step (race condition)
        try:
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
        except (AssertionError, Exception):
            # Don't count crashed episodes in win stats (outcome unknown)
            for ag in train_agents:
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
            # Check actual battle outcome (more reliable than reward sign)
            p1_battle = env.current_battles.get(p1)
            if p1_battle is not None and p1_battle.won:
                episode_wins += 1
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

    return buffers, obs_dict, episodes_done, episode_rewards, episode_wins


def train(config: Config, resume_path: str = None, run_name: str = "A"):
    """Main self-play training loop."""
    # Update sizes from environment (authoritative source)
    config.obs_size = OBS_SIZE
    config.species_vocab = SPECIES_VOCAB
    config.move_vocab = MOVE_VOCAB
    config.ability_vocab = ABILITY_VOCAB
    config.item_vocab = ITEM_VOCAB

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
    print(f"  Reward: dense delta (fainted={config.fainted_value}, hp={config.hp_value})")
    print(f"  Curriculum: MaxPower<{config.curriculum_phase1_steps/1e6:.0f}M, "
          f"Heuristic<{config.curriculum_phase2_steps/1e6:.0f}M, Self-play after")
    print(f"  Entropy: {config.entropy_coef} -> {config.entropy_coef_end}")
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

    prev_phase = None

    while timestep < config.total_timesteps:
        t0 = time.time()
        progress = timestep / config.total_timesteps

        # Entropy annealing (linear decay)
        entropy_coef = config.entropy_coef + (config.entropy_coef_end - config.entropy_coef) * progress

        # Curriculum phase selection
        curriculum_fn = None
        active_opponent = None
        if timestep < config.curriculum_phase1_steps:
            phase = "MaxPower"
            curriculum_fn = _max_power_action
        elif timestep < config.curriculum_phase2_steps:
            phase = "Heuristic"
            curriculum_fn = _heuristic_action
        else:
            phase = "Self-play"
            # Select opponent from pool (or None → self-play with current model)
            if config.use_opponent_pool and opponent_pool and opponent_model is not None:
                snapshot = random.choice(opponent_pool)
                opponent_model.load_state_dict(snapshot)
                active_opponent = opponent_model

        if phase != prev_phase:
            print(f"\n>>> Curriculum phase: {phase} (at {timestep:,} steps)\n")
            prev_phase = phase

        # Collect rollout
        buffers, obs_dict, episodes_done, episode_rewards, episode_wins = collect_rollout(
            env, ppo, config, obs_dict,
            opponent_model=active_opponent,
            curriculum_fn=curriculum_fn,
        )

        # Count transitions collected
        n_transitions = sum(b.size for b in buffers.values())
        timestep += n_transitions

        # PPO update
        active_buffers = [b for b in buffers.values() if b.size > 0]
        if active_buffers:
            stats = ppo.update(active_buffers, progress=progress, entropy_coef=entropy_coef)
        else:
            stats = {"pg_loss": 0, "v_loss": 0, "entropy": 0}

        elapsed = time.time() - t0
        fps = n_transitions / max(elapsed, 1e-6)

        # Logging
        writer.add_scalar("train/pg_loss", stats["pg_loss"], timestep)
        writer.add_scalar("train/v_loss", stats["v_loss"], timestep)
        writer.add_scalar("train/entropy", stats["entropy"], timestep)
        writer.add_scalar("train/entropy_coef", entropy_coef, timestep)
        writer.add_scalar("train/fps", fps, timestep)
        writer.add_scalar("train/episodes", episodes_done, timestep)

        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            writer.add_scalar("train/mean_episode_reward", mean_reward, timestep)
            winrate = episode_wins / max(episodes_done, 1)
            writer.add_scalar("train/winrate", winrate, timestep)

        iteration += 1
        if iteration % 1 == 0:
            ep_info = ""
            if episodes_done > 0:
                ep_info = f" | ep={episodes_done} wins={episode_wins}/{episodes_done}"
            print(
                f"[{timestep:>9d}/{config.total_timesteps}] ({phase}) "
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
                        eval_path, config, n_battles=config.eval_battles,
                        run_name=run_name,
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
