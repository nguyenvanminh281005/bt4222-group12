"""
Multi-seed comparison of Original NRMS vs. Gated Two-Branch NRMS (no training).

Loads the seed 42/43/44 checkpoints produced by train_and_evaluate.py and reports, on the Day 7 test split:
  1. Overall and per-history-bin metrics (mean ± std over seeds) for
     NRMS, NRMS + hard popularity fallback, and Gated NRMS (+ gate weight alpha per bin).
  2. Paired per-impression significance tests on seed-averaged scores
     (Wilcoxon signed-rank + paired bootstrap 95% CI of the mean difference),
     plus the sign of the mean difference in each individual seed.

Usage (from the project root):
    python models/evaluate_seeds.py
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import torch

from train_and_evaluate import (
    NRMS, GatedNRMS, evaluate_model, prepare_data, set_seed, HIST_BINS, HIST_LABELS,
)

METRICS = ['auc', 'mrr', 'ndcg5', 'ndcg10']
METRIC_NAMES = {'auc': 'AUC', 'mrr': 'MRR', 'ndcg5': 'NDCG@5', 'ndcg10': 'NDCG@10'}
SEEDS = [42, 43, 44]
VARIANTS = ['nrms', 'fallback', 'gated']
VARIANT_NAMES = {'nrms': 'Original NRMS', 'fallback': 'NRMS + hard fallback', 'gated': 'Gated NRMS'}

# Reference categorical palette, slots 1-3 (validated all-pairs for CVD); text stays in neutral ink.
COLORS = {'nrms': '#2a78d6', 'fallback': '#eb6834', 'gated': '#1baf7a'}
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def bootstrap_ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(a, b, per_seed_a, per_seed_b, mask):
    """a, b: seed-averaged per-impression scores; per_seed_*: [n_seeds, N]. Tests mean(b - a) on mask."""
    diff = b[mask] - a[mask]
    nz = diff[diff != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    lo, hi = bootstrap_ci(diff)
    seed_diffs = [float((sb[mask] - sa[mask]).mean()) for sa, sb in zip(per_seed_a, per_seed_b)]
    return {'n': int(mask.sum()), 'mean_diff': float(diff.mean()), 'ci95': [lo, hi], 'wilcoxon_p': p,
            'per_seed_diff': seed_diffs}


def plot_summary(summary, figures_dir):
    x = np.arange(len(HIST_LABELS))
    # 3. Figure: NDCG@5 by history bin (3 variants, mean ± std over seeds) + gate weight alpha by bin
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    offset = 0.22
    for k, v in enumerate(VARIANTS):
        means = [summary['groups'][b][v]['ndcg5']['mean'] for b in HIST_LABELS]
        stds = [summary['groups'][b][v]['ndcg5']['std'] for b in HIST_LABELS]
        ax.errorbar(x + (k - 1) * offset, means, yerr=stds, fmt='o', markersize=8, capsize=3, color=COLORS[v],
                    ecolor=COLORS[v], elinewidth=2, markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=VARIANT_NAMES[v])
    for k, v in [(0, 'nrms'), (2, 'gated')]:
        m = summary['groups'][HIST_LABELS[0]][v]['ndcg5']['mean']
        ax.text(x[0] + (k - 1) * offset + 0.08, m, f"{m:.3f}", ha='left', va='center', fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nN={summary['groups'][b]['count']:,}" for b in HIST_LABELS], fontsize=9.5, color=INK_2)
    ax.set_ylim(0.21, 0.37)
    ax.set_xlim(-0.5, len(HIST_LABELS) - 0.5)
    ax.set_ylabel('NDCG@5 (mean ± std, 3 seeds)', color=INK_2)
    ax.set_title('NDCG@5 by user history length (Day 7 test)', fontsize=12.5, fontweight='bold', color=INK, loc='left')
    ax.legend(frameon=False, loc='lower right', fontsize=9.5)

    ax = axes[1]
    am = [summary['groups'][b]['gated']['alpha']['mean'] for b in HIST_LABELS]
    asd = [summary['groups'][b]['gated']['alpha']['std'] for b in HIST_LABELS]
    ax.bar(x, am, 0.6, yerr=asd, capsize=3, color=COLORS['gated'], error_kw={'ecolor': INK_2, 'elinewidth': 1})
    for i, m in enumerate(am):
        ax.text(x[i], m + asd[i] + 0.02, f"{m:.2f}", ha='center', fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace(' (', '\n(') for b in HIST_LABELS], fontsize=9.5, color=INK_2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('History-branch weight α', color=INK_2)
    ax.set_title('Gated NRMS: learned weight α (1 − α → popularity)', fontsize=12.5, fontweight='bold', color=INK, loc='left')

    for ax in axes:
        ax.grid(True, axis='y', color=GRID, linewidth=0.8)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2)
    plt.tight_layout()
    out_fig = os.path.join(figures_dir, 'seed_comparison_by_history.png')
    plt.savefig(out_fig, dpi=300, facecolor='#fcfcfb')
    plt.close()
    print(f"Saved {out_fig}")



def main():
    parser = argparse.ArgumentParser(description="Multi-seed NRMS vs. Gated NRMS comparison")
    parser.add_argument('--gated_tag', type=str, default='gated', help="Tag of the gated run (checkpoint prefix)")
    parser.add_argument('--nrms_tag', type=str, default='nrms', help="Tag of the NRMS run (checkpoint prefix)")
    parser.add_argument('--plot_only', action='store_true', help="Redraw the figure from the saved summary JSON")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "models":
        data_dir, models_dir = "../data", "."
    else:
        data_dir, models_dir = "data", "models"
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    figures_dir = os.path.join(models_dir, "figures")
    if args.plot_only:
        with open(os.path.join(figures_dir, 'seed_comparison_summary.json')) as f:
            plot_summary(json.load(f), figures_dir)
        return

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_data(data_dir, device)
    test_records = data.test_records
    hist_len = np.array([r['history_len'] for r in test_records])
    bins = pd.cut(hist_len, bins=HIST_BINS, labels=HIST_LABELS)
    cold = (hist_len == 0)

    def make(cls):
        return cls(data.pretrained_embeddings, data.news_title_lookup, d_model=256, n_heads=16, query_dim=200,
                   dropout=0.2, pop_sets=data.pop_sets).to(device)

    # scores[variant][metric] -> [n_seeds, N]; alpha[variant] -> [n_seeds, N]
    scores = {v: {m: [] for m in METRICS} for v in VARIANTS}
    alpha = {v: [] for v in VARIANTS}
    for seed in SEEDS:
        for cls, tag, runs in [(NRMS, args.nrms_tag, [('nrms', False), ('fallback', True)]),
                               (GatedNRMS, args.gated_tag, [('gated', False)])]:
            ckpt = os.path.join(checkpoints_dir, f'{tag}_best_seed{seed}.pt')
            model = make(cls)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            for variant, fb in runs:
                res = evaluate_model(model, test_records, data.news_title_lookup, device, cold_fallback=fb)
                det = pd.DataFrame(res['detailed'])
                for m in METRICS:
                    scores[variant][m].append(det[m].to_numpy())
                alpha[variant].append(det['alpha'].to_numpy())
                print(f"seed {seed} | {VARIANT_NAMES[variant]:<22} | " +
                      " | ".join(f"{METRIC_NAMES[m]} {res[m]:.4f}" for m in METRICS))
            del model
            torch.cuda.empty_cache()
    scores = {v: {m: np.stack(scores[v][m]) for m in METRICS} for v in VARIANTS}
    alpha = {v: np.stack(alpha[v]) for v in VARIANTS}

    # 1. Mean ± std over seeds, overall and per history bin
    def seed_stats(arr, mask):
        per_seed = arr[:, mask].mean(axis=1)
        return {'mean': float(per_seed.mean()), 'std': float(per_seed.std()), 'per_seed': per_seed.tolist()}

    groups = {'Overall': np.ones(len(hist_len), dtype=bool), 'Warm (>=1)': ~cold}
    groups.update({b: (bins == b) for b in HIST_LABELS})
    summary = {'seeds': SEEDS, 'groups': {}}
    for g, mask in groups.items():
        summary['groups'][g] = {'count': int(mask.sum())}
        for v in VARIANTS:
            entry = {m: seed_stats(scores[v][m], mask) for m in METRICS}
            if v == 'gated':
                entry['alpha'] = seed_stats(alpha[v], mask)
            summary['groups'][g][v] = entry

    print("\n" + "=" * 100)
    print("MEAN ± STD OVER SEEDS 42/43/44 (Day 7 test)")
    print("=" * 100)
    for g in groups:
        s = summary['groups'][g]
        print(f"\n{g} (N={s['count']})")
        for v in VARIANTS:
            row = " | ".join(f"{METRIC_NAMES[m]} {s[v][m]['mean']:.4f}±{s[v][m]['std']:.4f}" for m in METRICS)
            extra = f" | alpha {s[v]['alpha']['mean']:.3f}±{s[v]['alpha']['std']:.3f}" if v == 'gated' else ""
            print(f"  {VARIANT_NAMES[v]:<22} {row}{extra}")

    # 2. Paired significance tests on seed-averaged per-impression scores
    comparisons = [('nrms', 'gated'), ('fallback', 'gated'), ('nrms', 'fallback')]
    test_groups = {'Overall': groups['Overall'], 'Cold Start (0)': cold, 'Warm (>=1)': ~cold}
    summary['paired_tests'] = {}
    print("\n" + "=" * 100)
    print("PAIRED TESTS: mean(B - A) per impression, seed-averaged | 95% bootstrap CI | Wilcoxon p | per-seed mean diff")
    print("=" * 100)
    for a, b in comparisons:
        key = f'{b}_vs_{a}'
        summary['paired_tests'][key] = {}
        for g, mask in test_groups.items():
            summary['paired_tests'][key][g] = {}
            for m in METRICS:
                r = paired_test(scores[a][m].mean(axis=0), scores[b][m].mean(axis=0), scores[a][m], scores[b][m], mask)
                summary['paired_tests'][key][g][m] = r
                print(f"{VARIANT_NAMES[b]:>20} vs {VARIANT_NAMES[a]:<20} | {g:<15} (N={r['n']:>5}) | {METRIC_NAMES[m]:<7} "
                      f"Δ={r['mean_diff']:+.4f} CI[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p={r['wilcoxon_p']:.2e} "
                      f"seeds=[{', '.join(f'{d:+.4f}' for d in r['per_seed_diff'])}]")

    out_json = os.path.join(figures_dir, 'seed_comparison_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    plot_summary(summary, figures_dir)

if __name__ == '__main__':
    main()
