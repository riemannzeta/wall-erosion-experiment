"""
Plotting for Wall Erosion Experiment Results
=============================================

Reads wall_erosion_summary.json and produces:
  1. Per-position MAE curves — the main figure showing wall softening
  2. Wall ratio bar chart across conditions
  3. Erosion fraction vs lambda — tests proportionality prediction

Usage:
    python plot_wall_erosion.py --results results/wall_erosion/wall_erosion_summary.json
    python plot_wall_erosion.py --results results/wall_erosion/wall_erosion_summary.json --output figures/
"""

import json
import argparse
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("matplotlib required: pip install matplotlib")
    raise


MECHANISM_COLORS = {
    'none': '#888888',
    'entropy': '#2196F3',
    'distill': '#E91E63',
    'smooth': '#4CAF50',
    'classify': '#FF9800',
}

MECHANISM_LABELS = {
    'none': 'No subsidy',
    'entropy': 'A: Entropy reg.',
    'distill': 'B: Distillation',
    'smooth': 'C: Smoothness',
    'classify': 'D: Aux classifier',
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def group_results(results):
    """Group results by (mechanism, control, lambda), averaging over seeds."""
    groups = {}
    for r in results:
        mech = r.get('mechanism', 'none')
        ctrl = r.get('control', False)
        lam = r.get('subsidy_lambda', 0.0)

        # Handle baseline_full specially
        cond = r.get('condition', '')
        if 'baseline_full' in cond:
            key = ('full', False, 0.0)
        else:
            key = (mech, ctrl, lam)

        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    return groups


def plot_per_position_mae(results, output_dir, loss_horizon=5):
    """Figure 1: Per-position MAE curves for each condition."""
    groups = group_results(results)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot baseline full and horizon first
    for key, runs in sorted(groups.items()):
        mech, ctrl, lam = key

        # Average per-position MAE across seeds
        all_positions = set()
        for r in runs:
            all_positions.update(int(k) for k in r['per_position'].keys())

        positions = sorted(all_positions)
        mae_means = []
        mae_stds = []
        for t in positions:
            vals = [r['per_position'][str(t)]['mae_mean']
                    for r in runs if str(t) in r['per_position']]
            mae_means.append(np.mean(vals))
            mae_stds.append(np.std(vals))

        color = MECHANISM_COLORS.get(mech, '#888888')
        linestyle = '--' if ctrl else '-'
        alpha = 0.5 if ctrl else 1.0
        linewidth = 2.5 if not ctrl else 1.5

        if mech == 'full':
            label = 'Full supervision (upper bound)'
            color = '#000000'
            linestyle = ':'
            linewidth = 2
            alpha = 0.7
        elif mech == 'none':
            label = f'Baseline horizon (K={loss_horizon})'
            linewidth = 2.5
        elif ctrl:
            label = f'{MECHANISM_LABELS.get(mech, mech)} control (λ={lam})'
        else:
            label = f'{MECHANISM_LABELS.get(mech, mech)} (λ={lam})'

        ax.plot(positions, mae_means, color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, label=label, marker='o',
                markersize=3)

        if len(runs) > 1:
            mae_means = np.array(mae_means)
            mae_stds = np.array(mae_stds)
            ax.fill_between(positions, mae_means - mae_stds,
                            mae_means + mae_stds, color=color, alpha=0.1)

    # Mark the wall boundary
    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle='--',
               alpha=0.4, linewidth=1.5, label=f'Loss horizon (K={loss_horizon})')

    ax.set_xlabel('Position t', fontsize=13)
    ax.set_ylabel('MAE (bits)', fontsize=13)
    ax.set_title('Wall Erosion: Per-Position MAE vs Bayesian Optimal', fontsize=14)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_xticks(range(1, 16))
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'wall_erosion_per_position.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_wall_ratio_bar(results, output_dir, loss_horizon=5):
    """Figure 2: Wall ratio bar chart across conditions."""
    groups = group_results(results)

    conditions = []
    wr_means = []
    wr_stds = []
    colors = []

    for key in sorted(groups.keys()):
        mech, ctrl, lam = key
        runs = groups[key]

        wrs = [r['wall_metrics']['wall_ratio'] for r in runs]

        if mech == 'full':
            label = 'Full supervision'
        elif mech == 'none':
            label = f'Baseline (K={loss_horizon})'
        elif ctrl:
            label = f'{mech} ctrl λ={lam}'
        else:
            label = f'{mech} λ={lam}'

        conditions.append(label)
        wr_means.append(np.mean(wrs))
        wr_stds.append(np.std(wrs))
        colors.append(MECHANISM_COLORS.get(mech, '#888888'))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = range(len(conditions))
    bars = ax.bar(x, wr_means, yerr=wr_stds, color=colors, alpha=0.8,
                  capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Wall Ratio (untrained/trained MAE)', fontsize=12)
    ax.set_title('Wall Ratio Across Conditions', fontsize=14)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
               label='No wall (WR=1)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(output_dir, 'wall_erosion_wall_ratio.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_erosion_vs_lambda(results, output_dir):
    """Figure 3: Erosion fraction vs lambda — tests proportionality."""
    groups = group_results(results)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for mech in ['entropy', 'distill', 'smooth', 'classify']:
        lambdas = []
        ef_means = []
        ef_stds = []

        for key, runs in groups.items():
            m, ctrl, lam = key
            if m == mech and not ctrl and lam > 0:
                efs = [r['erosion_fraction'] for r in runs]
                lambdas.append(lam)
                ef_means.append(np.mean(efs))
                ef_stds.append(np.std(efs))

        if not lambdas:
            continue

        order = np.argsort(lambdas)
        lambdas = [lambdas[i] for i in order]
        ef_means = [ef_means[i] for i in order]
        ef_stds = [ef_stds[i] for i in order]

        color = MECHANISM_COLORS[mech]
        label = MECHANISM_LABELS[mech]
        ax.errorbar(lambdas, ef_means, yerr=ef_stds, color=color,
                    marker='o', linewidth=2, capsize=5, label=label,
                    markersize=8)

        # Also plot control as X marker
        for key, runs in groups.items():
            m, ctrl, lam = key
            if m == mech and ctrl:
                efs = [r['erosion_fraction'] for r in runs]
                ax.scatter([lam], [np.mean(efs)], color=color, marker='x',
                           s=150, linewidths=3, zorder=5,
                           label=f'{label} control')

    ax.set_xlabel('Subsidy strength (λ)', fontsize=13)
    ax.set_ylabel('Erosion Fraction (0=wall, 1=no wall)', fontsize=13)
    ax.set_title('Wall Erosion vs Subsidy Strength', fontsize=14)
    ax.set_xscale('log')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.3, label='Full wall')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='No wall')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    path = os.path.join(output_dir, 'wall_erosion_vs_lambda.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot wall erosion experiment results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to wall_erosion_summary.json')
    parser.add_argument('--output', type=str, default='figures/wall_erosion',
                        help='Output directory for figures')
    parser.add_argument('--loss_horizon', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = load_results(args.results)
    print(f"Loaded {len(results)} result entries")

    plot_per_position_mae(results, args.output, args.loss_horizon)
    plot_wall_ratio_bar(results, args.output, args.loss_horizon)
    plot_erosion_vs_lambda(results, args.output)

    print("\nAll figures saved.")


if __name__ == '__main__':
    main()
