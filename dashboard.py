"""
Training Dashboard for AntiMateo RL Agent.

Reads training CSV logs and generates comprehensive plots.

Usage:
    # Ver run específico
    python dashboard.py --run A

    # Comparar múltiples runs
    python dashboard.py --runs A B C

    # Ver en vivo (actualiza cada 30s)
    python dashboard.py --run A --live

    # Guardar como imagen
    python dashboard.py --run A --save
"""

import argparse
import os
import time
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# Style
plt.style.use('dark_background')
COLORS = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b']
PHASE_COLORS = {
    'MaxPower': '#ffd43b',
    'Heuristic': '#51cf66',
    'Self-play': '#cc5de8',
}


def load_run_data(run_name: str, log_base: str = "logs") -> dict:
    """Load CSV data for a run. Returns dict with train and eval DataFrames."""
    csv_path = os.path.join(log_base, run_name, "training_stats.csv")

    if not os.path.exists(csv_path):
        # Try to find TensorBoard events instead
        tb_dir = os.path.join(log_base, run_name)
        event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
        if event_files:
            return load_from_tensorboard(event_files[0], run_name)
        raise FileNotFoundError(f"No data found for run '{run_name}' at {csv_path}")

    df = pd.read_csv(csv_path)

    # Split into training rows and eval rows
    train_df = df[~df['phase'].str.contains('eval', na=False)].copy()
    eval_df = df[df['phase'].str.contains('eval', na=False)].copy()
    eval_df = eval_df[eval_df['eval_winrate'].notna() & (eval_df['eval_winrate'] != '')].copy()
    eval_df['eval_winrate'] = pd.to_numeric(eval_df['eval_winrate'], errors='coerce')

    # Clean train data
    for col in ['pg_loss', 'v_loss', 'entropy', 'fps', 'winrate', 'mean_reward', 'entropy_coef']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    return {
        'name': run_name,
        'train': train_df,
        'eval': eval_df,
    }


def load_from_tensorboard(event_file: str, run_name: str) -> dict:
    """Load data from TensorBoard event file as fallback."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame([
                {'timestep': e.step, 'value': e.value}
                for e in events
            ])

        # Build train DataFrame
        train_data = {}
        for tag, df in data.items():
            key = tag.split('/')[-1]
            train_data[key] = df.set_index('timestep')['value']

        train_df = pd.DataFrame(train_data).reset_index().rename(columns={'index': 'timestep'})
        train_df['phase'] = 'unknown'

        # Build eval DataFrame
        eval_df = pd.DataFrame()
        if 'eval/vs_heuristic_winrate' in data:
            eval_df = data['eval/vs_heuristic_winrate'].rename(columns={'value': 'eval_winrate'})

        return {'name': run_name, 'train': train_df, 'eval': eval_df}

    except ImportError:
        raise ImportError("Install tensorboard: pip install tensorboard")


def smooth(values, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def format_steps(x, pos):
    """Format x-axis as 'XM' or 'Xk'."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}k'
    return str(int(x))


def plot_phase_background(ax, train_df, phase_col='phase'):
    """Add colored background for curriculum phases."""
    if phase_col not in train_df.columns or train_df.empty:
        return

    phases = train_df[phase_col].values
    timesteps = train_df['timestep'].values
    if len(timesteps) == 0:
        return

    # Find phase boundaries
    current_phase = phases[0]
    phase_start = timesteps[0]

    ylim = ax.get_ylim()
    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            color = PHASE_COLORS.get(current_phase, '#333333')
            ax.axvspan(phase_start, timesteps[i], alpha=0.06, color=color, zorder=0)
            current_phase = phases[i]
            phase_start = timesteps[i]

    # Last phase
    color = PHASE_COLORS.get(current_phase, '#333333')
    ax.axvspan(phase_start, timesteps[-1], alpha=0.06, color=color, zorder=0)


def create_dashboard(runs_data: list, save_path: str = None):
    """Create multi-panel training dashboard."""
    n_runs = len(runs_data)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax_winrate  = fig.add_subplot(gs[0, :2])  # Wide: train winrate
    ax_eval     = fig.add_subplot(gs[0, 2])   # Eval winrate
    ax_policy   = fig.add_subplot(gs[1, 0])   # Policy loss
    ax_value    = fig.add_subplot(gs[1, 1])   # Value loss
    ax_entropy  = fig.add_subplot(gs[1, 2])   # Entropy
    ax_reward   = fig.add_subplot(gs[2, 0])   # Mean reward
    ax_fps      = fig.add_subplot(gs[2, 1])   # FPS
    ax_summary  = fig.add_subplot(gs[2, 2])   # Summary table

    all_axes = [ax_winrate, ax_eval, ax_policy, ax_value,
                ax_entropy, ax_reward, ax_fps]

    for ax in all_axes:
        ax.set_facecolor('#16213e')
        ax.grid(True, alpha=0.15, color='white')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        ax.xaxis.set_major_formatter(FuncFormatter(format_steps))

    # ────────────────────────────────────────────
    # Plot 1: Training Win Rate (main plot)
    # ────────────────────────────────────────────
    ax_winrate.set_title('Training Win Rate', color='white', pad=8, fontsize=11)
    ax_winrate.set_ylabel('Win Rate', color='#aaaaaa', fontsize=9)
    ax_winrate.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    for i, run in enumerate(runs_data):
        color = COLORS[i % len(COLORS)]
        df = run['train']
        if df.empty or 'winrate' not in df.columns:
            continue

        valid = df[df['winrate'].notna()]
        if valid.empty:
            continue

        ts = valid['timestep'].values
        wr = valid['winrate'].values

        # Raw (transparent)
        ax_winrate.plot(ts, wr, alpha=0.15, color=color, linewidth=0.7)

        # Smoothed
        if len(wr) >= 10:
            sm = smooth(wr, window=15)
            ts_sm = ts[7:-7] if len(ts) > 14 else ts[:len(sm)]
            ax_winrate.plot(ts_sm, sm, color=color, linewidth=2,
                          label=f'Run {run["name"]}')
        else:
            ax_winrate.plot(ts, wr, color=color, linewidth=2,
                          label=f'Run {run["name"]}')

        # Phase backgrounds (first run only)
        if i == 0 and 'phase' in valid.columns:
            plot_phase_background(ax_winrate, valid)

    ax_winrate.axhline(0.5, color='white', linestyle='--', alpha=0.3, linewidth=1,
                      label='50% reference')
    if n_runs > 1:
        ax_winrate.legend(fontsize=8, loc='upper left',
                         facecolor='#1a1a2e', edgecolor='#333355')

    # Phase legend
    phase_patches = [matplotlib.patches.Patch(
        color=color, alpha=0.5, label=phase
    ) for phase, color in PHASE_COLORS.items()]
    ax_winrate.legend(handles=phase_patches + ax_winrate.get_legend_handles_labels()[0],
                     fontsize=7, loc='upper left',
                     facecolor='#1a1a2e', edgecolor='#333355')

    # ────────────────────────────────────────────
    # Plot 2: Evaluation Win Rate
    # ────────────────────────────────────────────
    ax_eval.set_title('Eval vs Heuristic', color='white', pad=8, fontsize=11)
    ax_eval.set_ylabel('Win Rate', color='#aaaaaa', fontsize=9)
    ax_eval.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    for i, run in enumerate(runs_data):
        color = COLORS[i % len(COLORS)]
        df = run['eval']
        if df.empty or 'eval_winrate' not in df.columns:
            continue

        valid = df[df['eval_winrate'].notna()]
        if valid.empty:
            continue

        ts = valid['timestep'].values
        wr = valid['eval_winrate'].values

        ax_eval.plot(ts, wr, 'o-', color=color, linewidth=1.5,
                    markersize=4, label=f'Run {run["name"]}')

        # Annotate best
        if len(wr) > 0:
            best_idx = np.argmax(wr)
            ax_eval.annotate(
                f'{wr[best_idx]:.0%}',
                (ts[best_idx], wr[best_idx]),
                textcoords='offset points', xytext=(5, 5),
                color=color, fontsize=7,
            )

    ax_eval.axhline(0.5, color='white', linestyle='--', alpha=0.3, linewidth=1)
    if n_runs > 1:
        ax_eval.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333355')

    # ────────────────────────────────────────────
    # Plots 3-7: Loss / Entropy / Reward / FPS
    # ────────────────────────────────────────────
    metrics = [
        (ax_policy, 'pg_loss', 'Policy Loss', 'Loss'),
        (ax_value, 'v_loss', 'Value Loss', 'Loss'),
        (ax_entropy, 'entropy', 'Entropy', 'Entropy'),
        (ax_reward, 'mean_reward', 'Mean Episode Reward', 'Reward'),
        (ax_fps, 'fps', 'FPS (Speed)', 'Steps/sec'),
    ]

    for ax, col, title, ylabel in metrics:
        ax.set_title(title, color='white', pad=6, fontsize=10)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)

        for i, run in enumerate(runs_data):
            color = COLORS[i % len(COLORS)]
            df = run['train']
            if df.empty or col not in df.columns:
                continue

            valid = df[df[col].notna()]
            if valid.empty:
                continue

            ts = valid['timestep'].values
            vals = valid[col].values

            ax.plot(ts, vals, alpha=0.15, color=color, linewidth=0.7)
            if len(vals) >= 10:
                sm = smooth(vals, window=15)
                ts_sm = ts[7:-7] if len(ts) > 14 else ts[:len(sm)]
                ax.plot(ts_sm, sm, color=color, linewidth=1.8,
                       label=f'Run {run["name"]}')
            else:
                ax.plot(ts, vals, color=color, linewidth=1.8,
                       label=f'Run {run["name"]}')

        if n_runs > 1:
            ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333355')

    # ────────────────────────────────────────────
    # Plot 8: Summary Table
    # ────────────────────────────────────────────
    ax_summary.axis('off')
    ax_summary.set_title('Run Summary', color='white', pad=8, fontsize=11)

    table_data = []
    col_labels = ['Run', 'Steps', 'Best Eval', 'Last WR', 'Hours']

    for run in runs_data:
        train = run['train']
        evl = run['eval']

        total_steps = train['timestep'].max() if not train.empty else 0
        last_wr = train['winrate'].dropna().iloc[-1] if not train.empty and 'winrate' in train else 0
        best_eval = evl['eval_winrate'].max() if not evl.empty and 'eval_winrate' in evl else 0
        hours = train['elapsed_hours'].max() if 'elapsed_hours' in train and not train.empty else 0

        table_data.append([
            run['name'],
            format_steps(total_steps, None),
            f'{best_eval:.1%}',
            f'{last_wr:.1%}',
            f'{hours:.1f}h',
        ])

    if table_data:
        table = ax_summary.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style table
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#16213e' if row > 0 else '#0f3460')
            cell.set_edgecolor('#333355')
            cell.set_text_props(color='white' if row > 0 else '#00d4ff')

    # Main title
    run_names = ', '.join(r['name'] for r in runs_data)
    fig.suptitle(
        f'AntiMateo Training Dashboard  —  Run{"s" if n_runs > 1 else ""}: {run_names}',
        color='white', fontsize=14, fontweight='bold', y=0.97,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Dashboard saved to {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Training Dashboard')
    parser.add_argument('--run', type=str, help='Single run name (e.g. A)')
    parser.add_argument('--runs', type=str, nargs='+', help='Multiple runs to compare (e.g. A B C)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Base log directory')
    parser.add_argument('--save', action='store_true', help='Save to PNG instead of displaying')
    parser.add_argument('--live', action='store_true', help='Refresh every 30 seconds')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval in seconds')
    args = parser.parse_args()

    run_names = args.runs or ([args.run] if args.run else None)
    if not run_names:
        # Auto-detect all runs
        log_base = args.log_dir
        if os.path.exists(log_base):
            run_names = [
                d for d in os.listdir(log_base)
                if os.path.isdir(os.path.join(log_base, d))
            ]
        if not run_names:
            print(f"No runs found in '{log_base}'. Specify --run or --runs.")
            return

    print(f"Loading runs: {run_names}")

    if args.live:
        print(f"Live mode: refreshing every {args.interval}s. Press Ctrl+C to stop.")
        matplotlib.use('TkAgg')
        plt.ion()
        while True:
            try:
                runs_data = []
                for name in run_names:
                    try:
                        runs_data.append(load_run_data(name, args.log_dir))
                    except FileNotFoundError as e:
                        print(f"  Warning: {e}")

                if not runs_data:
                    print("No data found. Waiting...")
                    time.sleep(args.interval)
                    continue

                plt.close('all')
                create_dashboard(runs_data)
                plt.pause(0.1)
                print(f"  Updated at {time.strftime('%H:%M:%S')}. Next in {args.interval}s...")
                time.sleep(args.interval)

            except KeyboardInterrupt:
                print("\nStopped live mode.")
                break
    else:
        runs_data = []
        for name in run_names:
            try:
                data = load_run_data(name, args.log_dir)
                runs_data.append(data)
                train_rows = len(data['train'])
                eval_rows = len(data['eval'])
                print(f"  Run {name}: {train_rows} training rows, {eval_rows} eval rows")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

        if not runs_data:
            print("No data found.")
            return

        save_path = None
        if args.save:
            save_path = f"dashboard_{'_'.join(run_names)}.png"

        create_dashboard(runs_data, save_path=save_path)


if __name__ == "__main__":
    main()
