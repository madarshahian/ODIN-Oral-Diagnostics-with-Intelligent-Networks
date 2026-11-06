#!/usr/bin/env python3
"""
============================================================================
AGGREGATE 5-FOLD CROSS-VALIDATION RESULTS
============================================================================
Scans logs/fold_{k}/*/artifacts/run_metadata.json (+ test_predictions.pkl),
computes mean ± std across folds for:
    • Test accuracy
    • Macro / weighted F1
    • Per-class F1 (healthy / lesion / mrc)
and — by pooling every fold's out-of-fold predictions — produces a single
"pooled" confusion matrix and classification report covering ALL patients
(each evaluated exactly once). That pooled view is the honest, dataset-wide
result to headline in the paper, alongside the per-fold mean ± std.

Outputs (under output_figures/kfold_summary/), all in PDF + PNG + SVG:
    kfold_metrics_table.csv          — per-fold rows + mean/std rows (all metrics)
    kfold_pooled_confusion.pdf       — confusion matrix over all OOF predictions
    kfold_per_class_metrics.pdf      — Precision/Recall/Specificity/F1, mean±std
    kfold_per_fold_overview.pdf      — accuracy + macro-F1 per fold (consistency)
    kfold_pooled_roc.pdf             — pooled ROC (AUC) over all OOF probabilities
    kfold_pooled_pr.pdf              — pooled PR (AUPRC) over all OOF probabilities
    kfold_roc_byclass_folds.pdf      — per-class ROC: 5 fold curves + bold pooled
    kfold_pr_byclass_folds.pdf       — per-class PR:  5 fold curves + bold pooled
    kfold_roc_pooled_vs_folds.pdf    — 2-panel: pooled (left) vs all fold curves (right)
    kfold_pr_pooled_vs_folds.pdf     — PR counterpart of the above
    kfold_roc_mean_band.pdf          — per-class ROC: faint folds + mean ±1 std + pooled
    kfold_pr_mean_band.pdf           — PR counterpart of the above
    kfold_reliability_diagram.pdf    — pooled reliability curve + ECE (calibration)
    kfold_f1_heatmap.pdf             — fold × class F1 grid with mean ± std (suppl.)
    kfold_forest_summary.pdf         — forest plot: mean ± 95% CI per metric
    kfold_tsne_representative.pdf     — t-SNE of the representative fold (acc≈mean)
    kfold_tsne_grid.pdf              — 5-panel grid of every fold's t-SNE (suppl.)
    kfold_patient_level_pooled.pdf   — pooled patient-level rescue chart + per-fold
                                       outcome summary (needs 03_post_analysis.py
                                       --patient run per fold first)
    kfold_dataset_composition.pdf    — pooled test composition (full cohort) +
                                       per-fold train/val/test split sizes (needs
                                       03_post_analysis.py --dataset run per fold)
    kfold_summary_report.txt         — headline mean±std + pooled ECE + report

Note: t-SNE is NOT pooled — each fold trains a distinct model with its own
feature space, so embeddings are not comparable across folds. We therefore (a)
surface the representative fold's t-SNE as the main figure and (b) provide a
grid of all folds for transparency. Grad-CAM is likewise per-fold; pick
illustrative examples from any fold's output_figures/fold_k/evaluation/.

Usage:
    python 03_aggregate_kfold_results.py
    python 03_aggregate_kfold_results.py --n-folds 5 --logs-dir logs
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--logs-dir', type=str, default='logs')
parser.add_argument('--out-dir', type=str, default='output_figures/kfold_summary')
args = parser.parse_args()

LOGS_DIR = Path(args.logs_dir)
OUT_DIR  = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL_CMAP = LinearSegmentedColormap.from_list(
    'journal_blue',
    ['#FFFFFF', '#E8F1F5', '#B8D4E3', '#7FA9C7', '#3D6A8A', '#1A3D5C'], N=256)

# ── Manuscript figure style (consistent with training/post-analysis scripts) ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 10, 'legend.frameon': True,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'grid.alpha': 0.3,
})

# Per-class palette — colorblind-safe, consistent with the training script.
CLASS_PALETTE   = {'healthy': '#2166AC', 'lesion': '#B2182B', 'mrc': '#1B7837'}
# B&W-safe redundant encodings (per class, by index)
CLASS_LINESTYLES = ['-', '--', ':']
CLASS_MARKERS    = ['o', 's', '^']
CHANCE_COLOR     = '#757575'

def _disp(c):
    return 'MRC' if c.lower() == 'mrc' else c.capitalize()

def _cls_color(c, idx=0):
    return CLASS_PALETTE.get(c.lower(), ['#3D6A8A', '#C77F3D', '#8A3D6A'][idx % 3])

def _save(fig, stem):
    """Save a figure as PDF, PNG and SVG at 300 DPI with white background."""
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(OUT_DIR / f"{stem}.{fmt}", format=fmt, dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {stem}.pdf / .png / .svg")


# ── Locate each fold's latest run artifacts ────────────────────────────────────
def find_fold_artifacts(fold_k):
    """Return (metadata_dict, predictions_dict) for the latest run of fold_k."""
    fold_dir = LOGS_DIR / f"fold_{fold_k}"
    if not fold_dir.exists():
        return None, None
    # Each run is logs/fold_k/run_YYYYMMDD_HHMMSS/artifacts/
    runs = sorted([d for d in fold_dir.glob("run_*") if d.is_dir()])
    if not runs:
        return None, None
    latest = runs[-1]                      # most recent run for this fold
    meta_path = latest / "artifacts" / "run_metadata.json"
    pred_path = latest / "artifacts" / "test_predictions.pkl"
    meta = json.load(open(meta_path)) if meta_path.exists() else None
    preds = pickle.load(open(pred_path, 'rb')) if pred_path.exists() else None
    return meta, preds


# ── Gather ─────────────────────────────────────────────────────────────────────
rows = []
pooled_y_true, pooled_y_pred = [], []
pooled_proba = []
per_fold_eval = {}   # k -> {'y_true': np.array, 'proba': np.array}  for per-fold ROC/PR
class_names = None

print(f"Scanning {LOGS_DIR}/fold_1..{args.n_folds} ...")
for k in range(1, args.n_folds + 1):
    meta, preds = find_fold_artifacts(k)
    if meta is None:
        print(f"  ⚠ fold_{k}: no run_metadata.json found — skipping")
        continue

    if class_names is None and meta.get('class_names'):
        class_names = list(meta['class_names'])

    per_class = meta.get('test_f1_per_class', {})
    row = {
        'fold': k,
        'test_accuracy': meta.get('test_accuracy', np.nan),
        'f1_macro': meta.get('test_f1_macro', np.nan),
        'f1_weighted': meta.get('test_f1_weighted', np.nan),
        'epochs_trained': meta.get('epochs_trained', np.nan),
        'n_test_pairs': meta.get('num_test_pairs', np.nan),
    }
    for cls, v in per_class.items():
        row[f'f1_{cls}'] = v

    # Compute per-class precision / recall / specificity / F1 for THIS fold,
    # directly from its predictions (so mean±std across folds is available for
    # all four metrics, not just F1). Specificity = TN / (TN + FP) per class.
    if preds is not None and class_names is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        yt_f = np.array(preds['y_true'])
        yp_f = np.array(preds['y_pred'])
        for ci, cls in enumerate(class_names):
            tn = int(np.sum((yt_f != ci) & (yp_f != ci)))
            fp = int(np.sum((yt_f != ci) & (yp_f == ci)))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            row[f'precision_{cls}'] = float(precision_score(
                yt_f, yp_f, labels=[ci], average='macro', zero_division=0))
            row[f'recall_{cls}'] = float(recall_score(
                yt_f, yp_f, labels=[ci], average='macro', zero_division=0))
            row[f'specificity_{cls}'] = float(spec)
            # F1 from predictions too (overrides metadata if class order matches)
            row[f'f1_{cls}'] = float(f1_score(
                yt_f, yp_f, labels=[ci], average='macro', zero_division=0))

        # Recompute fold-level aggregates from predictions so they never depend
        # on optional metadata keys (older training runs did not write
        # 'test_f1_macro', which previously surfaced as macro-F1 = NaN).
        from sklearn.metrics import accuracy_score
        row['f1_macro']     = float(f1_score(yt_f, yp_f, average='macro',
                                             zero_division=0))
        row['f1_weighted']  = float(f1_score(yt_f, yp_f, average='weighted',
                                             zero_division=0))
        row['test_accuracy'] = float(accuracy_score(yt_f, yp_f))

    rows.append(row)

    if preds is not None:
        pooled_y_true.extend(preds['y_true'])
        pooled_y_pred.extend(preds['y_pred'])
        if 'y_pred_proba' in preds:
            pooled_proba.extend(preds['y_pred_proba'])
            # Keep this fold's labels+proba separate for per-fold ROC/PR overlays
            per_fold_eval[k] = {
                'y_true': np.asarray(preds['y_true']),
                'proba':  np.asarray(preds['y_pred_proba']),
            }
        if class_names is None:
            class_names = list(preds.get('class_names', []))

    acc = row['test_accuracy']
    print(f"  ✓ fold_{k}: acc={acc:.4f}  macro-F1={row['f1_macro']:.4f}  "
          f"epochs={row['epochs_trained']}")

if not rows:
    print("✗ No fold results found. Did training write logs/fold_k/*/artifacts/?")
    raise SystemExit(1)

if class_names is None:
    class_names = ['healthy', 'lesion', 'mrc']


# ── Per-fold metrics table with mean ± std ─────────────────────────────────────
import csv

metric_keys = ['test_accuracy', 'f1_macro', 'f1_weighted'] + \
              [f'{m}_{c}' for c in class_names
               for m in ('precision', 'recall', 'specificity', 'f1')]

def col(key):
    return np.array([r[key] for r in rows if key in r and not np.isnan(r.get(key, np.nan))])

summary_lines = []
summary_lines.append("=" * 68)
summary_lines.append(f"{args.n_folds}-FOLD CROSS-VALIDATION SUMMARY")
summary_lines.append("=" * 68)
summary_lines.append(f"Folds with results: {len(rows)} / {args.n_folds}")
summary_lines.append("")
summary_lines.append(f"{'Metric':<20s} {'Mean':>8s} {'Std':>8s}   Per-fold values")
summary_lines.append("-" * 68)

mean_std = {}
for key in metric_keys:
    vals = col(key)
    if len(vals) == 0:
        continue
    m, s = float(np.mean(vals)), float(np.std(vals))
    mean_std[key] = (m, s)
    per_fold = "  ".join(f"{v:.3f}" for v in vals)
    label = key.replace('test_', '').replace('f1_', 'F1-')
    summary_lines.append(f"{label:<20s} {m:>8.4f} {s:>8.4f}   [{per_fold}]")

summary_lines.append("")
summary_lines.append("Headline (paper-ready):")
if 'test_accuracy' in mean_std:
    m, s = mean_std['test_accuracy']
    summary_lines.append(f"  Accuracy   : {m*100:.2f}% ± {s*100:.2f}%")
if 'f1_macro' in mean_std:
    m, s = mean_std['f1_macro']
    summary_lines.append(f"  Macro-F1   : {m:.4f} ± {s:.4f}")
for c in class_names:
    key = f'f1_{c}'
    if key in mean_std:
        m, s = mean_std[key]
        summary_lines.append(f"  F1 {c:<8s}: {m:.4f} ± {s:.4f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# Write CSV (one row per fold + a mean and std row)
csv_path = OUT_DIR / "kfold_metrics_table.csv"
all_cols = ['fold'] + metric_keys + ['epochs_trained', 'n_test_pairs']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
    # mean / std rows
    mean_row = {'fold': 'MEAN'}
    std_row  = {'fold': 'STD'}
    for key in metric_keys:
        vals = col(key)
        if len(vals):
            mean_row[key] = f"{np.mean(vals):.4f}"
            std_row[key]  = f"{np.std(vals):.4f}"
    w.writerow(mean_row)
    w.writerow(std_row)
print(f"\n✓ {csv_path}")


# ── Pooled out-of-fold confusion matrix (every patient evaluated once) ─────────
if pooled_y_true:
    yt = np.array(pooled_y_true)
    yp = np.array(pooled_y_pred)
    n = len(class_names)
    cm = confusion_matrix(yt, yp, labels=list(range(n)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    pooled_acc = accuracy_score(yt, yp)
    pooled_macro_f1 = f1_score(yt, yp, average='macro', zero_division=0)

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(cm_norm, cmap=JOURNAL_CMAP, vmin=0, vmax=1.0,
                  aspect='equal', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            pct = cm_norm[i, j] * 100
            cnt = int(cm[i, j])
            tc = 'white' if cm_norm[i, j] > 0.5 else '#1A3D5C'
            ax.text(j, i, f"{pct:.1f}%\nn={cnt}", ha='center', va='center',
                    color=tc, fontsize=12, fontweight='bold', linespacing=1.3)
            if i == j:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor='#D4AF37', linewidth=2.5, zorder=10))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([_disp(c) for c in class_names], fontsize=12, fontweight='bold')
    ax.set_yticklabels([_disp(c) for c in class_names], fontsize=12,
                       fontweight='bold', rotation=0)
    ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Class', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(f"Pooled {args.n_folds}-Fold Confusion Matrix (All Patients)\n"
                 f"Accuracy = {pooled_acc:.4f}  •  Macro-F1 = {pooled_macro_f1:.4f}",
                 fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(axis='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#333'); sp.set_linewidth(1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label('Recall (Row-Normalized)', fontsize=11, fontweight='bold', labelpad=8)
    plt.tight_layout()
    _save(fig, "kfold_pooled_confusion")

    summary_lines.append("")
    summary_lines.append("-" * 68)
    summary_lines.append("POOLED (all out-of-fold predictions, every patient once):")
    summary_lines.append(f"  Pooled accuracy : {pooled_acc:.4f}")
    summary_lines.append(f"  Pooled macro-F1 : {pooled_macro_f1:.4f}")

    # ── Expected Calibration Error (ECE) — pooled, reported as a single scalar ──
    # We deliberately report ECE as ONE number over the pooled out-of-fold
    # predictions rather than per-fold reliability diagrams: at this dataset
    # size, 10-bin per-class reliability curves are dominated by sampling noise
    # (a few samples per bin) and are not statistically meaningful. A pooled
    # scalar over every patient is the honest calibration summary.
    if pooled_proba and len(pooled_proba) == len(yt):
        proba = np.asarray(pooled_proba)
        conf = proba.max(axis=1)                 # top-class confidence
        pred = proba.argmax(axis=1)
        correct = (pred == yt).astype(float)
        n_bins = 10
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        N = len(conf)
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
            if in_bin.sum() == 0:
                continue
            acc_bin  = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += (in_bin.sum() / N) * abs(acc_bin - conf_bin)
        summary_lines.append(f"  Pooled ECE      : {ece:.4f}  "
                             f"(N={N}, 10-bin; lower = better calibrated)")
        print(f"  Pooled ECE (calibration): {ece:.4f}")

        # ── NEW FIGURE: pooled reliability diagram ──────────────────────────
        # Visual companion to the ECE scalar: per-bin accuracy vs confidence
        # over ALL out-of-fold predictions (statistically meaningful because
        # the pool covers every patient once), with a sample-count histogram.
        bin_centers, bin_accs, bin_confs, bin_ns = [], [], [], []
        for b in range(n_bins):
            lo_b, hi_b = bins[b], bins[b + 1]
            in_b = ((conf > lo_b) & (conf <= hi_b)) if b > 0 else \
                   ((conf >= lo_b) & (conf <= hi_b))
            if in_b.sum() == 0:
                continue
            bin_centers.append((lo_b + hi_b) / 2)
            bin_accs.append(correct[in_b].mean())
            bin_confs.append(conf[in_b].mean())
            bin_ns.append(int(in_b.sum()))

        fig, (axr, axh) = plt.subplots(
            2, 1, figsize=(7, 8), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
        axr.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.5,
                 linestyle=(0, (1, 1)), label='Perfect calibration', zorder=1)
        # Gap bars (confidence − accuracy) drawn behind the curve
        for c0, a0 in zip(bin_confs, bin_accs):
            axr.plot([c0, c0], [a0, c0], color='#C62828', lw=3, alpha=0.45,
                     solid_capstyle='butt', zorder=2)
        axr.plot(bin_confs, bin_accs, color='#1A3D5C', lw=2.2, marker='o',
                 markersize=7, markerfacecolor='white', markeredgewidth=1.6,
                 label='Model (pooled OOF)', zorder=3)
        axr.set_ylabel('Empirical accuracy', fontsize=12, fontweight='bold')
        axr.set_xlim([0, 1.0]); axr.set_ylim([0, 1.02])
        axr.set_title(f'Reliability Diagram — Pooled {args.n_folds}-Fold '
                      f'Out-of-Fold Predictions\nECE = {ece:.4f}  (N = {N}, 10 bins)',
                      fontsize=13, fontweight='bold')
        axr.legend(loc='upper left', framealpha=0.95)
        axr.grid(True, alpha=0.3); axr.set_axisbelow(True)
        axh.bar(bin_centers, bin_ns, width=0.085, color='#7FA9C7',
                edgecolor='#1A3D5C', linewidth=0.8)
        axh.set_yscale('symlog')
        axh.set_xlabel('Predicted confidence (top class)',
                       fontsize=12, fontweight='bold')
        axh.set_ylabel('Samples', fontsize=10, fontweight='bold')
        axh.grid(True, alpha=0.3, axis='y'); axh.set_axisbelow(True)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.09)
        _save(fig, "kfold_reliability_diagram")
    else:
        summary_lines.append("  Pooled ECE      : n/a (no probabilities saved)")

    summary_lines.append("")
    summary_lines.append(classification_report(yt, yp, target_names=class_names,
                                                digits=4, zero_division=0))
    summary_text = "\n".join(summary_lines)


# ── Per-class metrics across folds (Precision/Recall/Specificity/F1) ──────────
# Grouped bars: x = class, 4 bars per class (one per metric), mean ± std across
# folds, with individual fold dots overlaid. B&W-safe via distinct hatches.
metrics_to_plot = ['precision', 'recall', 'specificity', 'f1']
metric_labels   = ['Precision', 'Recall', 'Specificity', 'F1-Score']
metric_colors   = ['#3D6A8A', '#7FA9C7', '#C77F3D', '#8A3D6A']
metric_hatches  = ['', '///', '...', 'xxx']

fig, ax = plt.subplots(figsize=(11, 6))
n_cls = len(class_names)
n_met = len(metrics_to_plot)
bar_w = 0.2
x = np.arange(n_cls)

for mi, (mkey, mlabel) in enumerate(zip(metrics_to_plot, metric_labels)):
    means = [mean_std.get(f'{mkey}_{c}', (np.nan, 0))[0] for c in class_names]
    stds  = [mean_std.get(f'{mkey}_{c}', (np.nan, 0))[1] for c in class_names]
    offset = (mi - (n_met - 1) / 2) * bar_w
    bars = ax.bar(x + offset, means, bar_w,
                  label=mlabel, color=metric_colors[mi],
                  hatch=metric_hatches[mi], yerr=stds, capsize=4,
                  edgecolor='#222', linewidth=0.8,
                  error_kw={'elinewidth': 1.2, 'ecolor': '#222'})
    # individual fold dots
    for ci, c in enumerate(class_names):
        vals = col(f'{mkey}_{c}')
        if len(vals):
            ax.scatter(np.full(len(vals), x[ci] + offset), vals,
                       color='#111', s=14, zorder=6, alpha=0.6,
                       edgecolors='white', linewidths=0.4)
    # value labels (placed above the error-bar cap to avoid collision)
    for b, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (s if not np.isnan(s) else 0) + 0.025,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#333')

ax.set_xticks(x)
ax.set_xticklabels([_disp(c) for c in class_names], fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.15])
ax.set_title(f'Per-Class Performance Across {args.n_folds} Folds '
             f'(mean ± std; dots = individual folds)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.95, ncol=2, fontsize=10)
ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_edgecolor('#333'); sp.set_linewidth(0.8)
plt.tight_layout()
_save(fig, "kfold_per_class_metrics")


# ── NEW FIGURE: fold × class F1 heatmap (stability at a glance) ───────────────
# Compact supplementary view: every (fold, class) F1 in one annotated grid,
# with per-class mean ± std in a summary row. Reads cleanly in greyscale
# because the value is printed inside each cell.
_f1_grid = np.full((len(rows), len(class_names)), np.nan)
for ri, r in enumerate(rows):
    for ci, c in enumerate(class_names):
        _f1_grid[ri, ci] = r.get(f'f1_{c}', np.nan)

if not np.all(np.isnan(_f1_grid)):
    fig, ax = plt.subplots(figsize=(6.6, 0.78 * (len(rows) + 1) + 1.8))
    im = ax.imshow(_f1_grid, cmap=JOURNAL_CMAP, vmin=0.5, vmax=1.0,
                   aspect='auto', interpolation='nearest')
    for ri in range(len(rows)):
        for ci in range(len(class_names)):
            v = _f1_grid[ri, ci]
            if np.isnan(v):
                continue
            tc = 'white' if (v - 0.5) / 0.5 > 0.55 else '#1A3D5C'
            ax.text(ci, ri, f"{v:.3f}", ha='center', va='center',
                    color=tc, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([_disp(c) for c in class_names],
                       fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"Fold {r['fold']}" for r in rows],
                       fontsize=11, fontweight='bold')
    # Per-class mean ± std beneath the grid
    for ci, c in enumerate(class_names):
        vals = _f1_grid[:, ci]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            ax.text(ci, len(rows) - 0.5 + 0.4,
                    f"{vals.mean():.3f} ± {vals.std():.3f}",
                    ha='center', va='top', fontsize=10, color='#333',
                    fontweight='bold', clip_on=False)
    ax.set_title(f'Per-Class F1 Across {args.n_folds} Folds',
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('#333'); sp.set_linewidth(1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label('F1-score', fontsize=10, fontweight='bold')
    plt.tight_layout()
    _save(fig, "kfold_f1_heatmap")


# ── NEW FIGURE: forest-style summary (mean ± 95% CI per metric) ───────────────
# Clinical-journal staple: each headline metric as a point estimate with a
# t-distribution 95% CI across folds, individual fold values shown as dots.
try:
    from scipy import stats as _st
    _tcrit = lambda n: _st.t.ppf(0.975, df=max(n - 1, 1))
except ImportError:
    _tcrit = lambda n: 2.776 if n == 5 else 1.96  # t(0.975, 4) fallback

forest_items = [('Accuracy', 'test_accuracy'),
                ('Macro-F1', 'f1_macro'),
                ('Weighted-F1', 'f1_weighted')] + \
               [(f'F1 — {_disp(c)}', f'f1_{c}') for c in class_names]

fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(forest_items) + 2.2))
ylocs = np.arange(len(forest_items))[::-1]
for (label, key), y0 in zip(forest_items, ylocs):
    vals = col(key)
    if len(vals) == 0:
        continue
    m = vals.mean()
    half = _tcrit(len(vals)) * vals.std(ddof=1) / np.sqrt(len(vals)) \
        if len(vals) > 1 else 0.0
    ax.plot([m - half, m + half], [y0, y0], color='#1A3D5C', lw=2.4,
            solid_capstyle='round', zorder=3)
    for cap in (m - half, m + half):
        ax.plot([cap, cap], [y0 - 0.13, y0 + 0.13], color='#1A3D5C',
                lw=1.6, zorder=3)
    ax.scatter(vals, np.full(len(vals), y0), s=26, color='#7FA9C7',
               edgecolors='#1A3D5C', linewidths=0.7, alpha=0.85, zorder=4)
    ax.scatter([m], [y0], s=130, marker='D', color='#B2182B',
               edgecolors='black', linewidths=0.8, zorder=5)
    ax.text(1.012, y0, f"{m:.3f}  [{m - half:.3f}, {m + half:.3f}]",
            va='center', ha='left', fontsize=10, color='#222',
            transform=ax.get_yaxis_transform())
ax.set_yticks(ylocs)
ax.set_yticklabels([lbl for lbl, _ in forest_items],
                   fontsize=11, fontweight='bold')
ax.set_xlabel('Score', fontsize=12, fontweight='bold')
ax.set_xlim([min(0.5, min(col('f1_macro'), default=0.5) - 0.05), 1.0])
ax.set_title(f'{args.n_folds}-Fold Performance Summary '
             f'(diamond = mean, bar = 95% CI, dots = folds)',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.3, axis='x'); ax.set_axisbelow(True)
plt.subplots_adjust(right=0.74)
_save(fig, "kfold_forest_summary")


# ── Per-fold consistency overview (accuracy + macro-F1 per fold) ──────────────
# Shows fold-to-fold stability — the core argument that the model isn't reliant
# on one lucky split. Two series distinguished by colour AND hatch (B&W-safe).
fold_ids = [r['fold'] for r in rows]
accs     = [r['test_accuracy'] for r in rows]
macrof1s = [r['f1_macro'] for r in rows]

fig, ax = plt.subplots(figsize=(9, 5.5))
xf = np.arange(len(fold_ids))
bw = 0.38
b1 = ax.bar(xf - bw/2, accs, bw, label='Accuracy',
            color='#3D6A8A', hatch='', edgecolor='#222', linewidth=0.8)
b2 = ax.bar(xf + bw/2, macrof1s, bw, label='Macro-F1',
            color='#C77F3D', hatch='///', edgecolor='#222', linewidth=0.8)
# mean ± std reference band for accuracy
if len(accs):
    a_m, a_s = np.mean(accs), np.std(accs)
    ax.axhline(a_m, color=CHANCE_COLOR, linestyle='--', linewidth=1.3,
               label=f'Mean acc = {a_m:.3f} ± {a_s:.3f}')
    ax.axhspan(a_m - a_s, a_m + a_s, color=CHANCE_COLOR, alpha=0.12, zorder=0)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f'{b.get_height():.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#333')
ax.set_xticks(xf)
ax.set_xticklabels([f'Fold {f}' for f in fold_ids], fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.12])
ax.set_title(f'Per-Fold Performance ({args.n_folds}-Fold Cross-Validation)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)
plt.tight_layout()
_save(fig, "kfold_per_fold_overview")


# ── Pooled ROC & PR curves (over ALL out-of-fold probabilities) ───────────────
# At this dataset size, per-fold ROC/PR are noisy; pooling every patient's
# out-of-fold probability gives one stable curve per class. B&W-safe via
# distinct linestyle + marker per class.
if pooled_proba and len(pooled_proba) == len(pooled_y_true):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
        average_precision_score
    from sklearn.preprocessing import label_binarize

    proba_all = np.asarray(pooled_proba)
    yt_all = np.asarray(pooled_y_true)
    n_cls = len(class_names)
    ytb = label_binarize(yt_all, classes=list(range(n_cls)))

    # ROC
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(ytb[:, i], proba_all[:, i])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=_cls_color(cls, i), lw=2,
                linestyle=CLASS_LINESTYLES[i % 3], marker=CLASS_MARKERS[i % 3],
                markevery=max(1, len(fpr)//10), markersize=6,
                label=f'{_disp(cls)}  (AUC = {a:.3f})')
    ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.5, linestyle=(0, (1, 1)),
            label='Chance (AUC = 0.500)')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title(f'Pooled ROC Curves — {args.n_folds}-Fold (One-vs-Rest)',
                 fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "kfold_pooled_roc")

    # PR
    fig, ax = plt.subplots(figsize=(7, 7))
    auprcs = []
    for i, cls in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(ytb[:, i], proba_all[:, i])
        ap = average_precision_score(ytb[:, i], proba_all[:, i])
        auprcs.append(ap)
        ax.plot(rec, prec, color=_cls_color(cls, i), lw=2,
                linestyle=CLASS_LINESTYLES[i % 3], marker=CLASS_MARKERS[i % 3],
                markevery=max(1, len(rec)//10), markersize=6,
                label=f'{_disp(cls)}  (AUPRC = {ap:.3f})')
    ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(f'Pooled Precision–Recall — {args.n_folds}-Fold (One-vs-Rest)\n'
                 f'Macro-AUPRC = {np.mean(auprcs):.3f}', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "kfold_pooled_pr")

    # =========================================================================
    # ADDITIONAL paper-candidate layouts (do not replace the two above).
    # Headline curve convention for all of these:
    #   • bold SOLID  = pooled OOF (concatenated probabilities, all patients once)
    #   • bold DASHED = mean across folds (vertical-averaged), with ±1 std band
    #   • thin lines  = the 5 individual folds
    # This lets a reviewer see pooled-vs-mean and fold dispersion at a glance.
    # =========================================================================
    folds_avail = sorted(per_fold_eval)
    have_perfold = len(folds_avail) >= 2

    def _binarize(yt):
        return label_binarize(yt, classes=list(range(n_cls)))

    # Common FPR / recall grid for vertical averaging across folds
    _grid = np.linspace(0, 1, 200)

    def _fold_roc(i):
        """Per-fold (fpr,tpr,auc) for class i; plus interpolated tpr on _grid."""
        out = []
        for k in folds_avail:
            yt = _binarize(per_fold_eval[k]['y_true'])[:, i]
            pr = per_fold_eval[k]['proba'][:, i]
            if yt.sum() == 0:        # class absent in this fold's test set
                continue
            fpr, tpr, _ = roc_curve(yt, pr)
            tpr_i = np.interp(_grid, fpr, tpr); tpr_i[0] = 0.0
            out.append((k, fpr, tpr, auc(fpr, tpr), tpr_i))
        return out

    def _fold_pr(i):
        out = []
        for k in folds_avail:
            yt = _binarize(per_fold_eval[k]['y_true'])[:, i]
            pr = per_fold_eval[k]['proba'][:, i]
            if yt.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(yt, pr)
            # precision_recall_curve returns decreasing recall; sort for interp
            order = np.argsort(rec)
            prec_i = np.interp(_grid, rec[order], prec[order])
            out.append((k, rec, prec, average_precision_score(yt, pr), prec_i))
        return out

    _fold_cmap = plt.get_cmap('viridis')
    _fold_colors = {k: _fold_cmap(t) for k, t in
                    zip(folds_avail, np.linspace(0.12, 0.85, len(folds_avail)))}

    # ── LAYOUT 1 — per class (3 subplots): 5 fold curves + bold pooled ───────
    if have_perfold:
        fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 5.2),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for i, cls in enumerate(class_names):
            ax = axes[i]
            for (k, fpr, tpr, a, _ti) in _fold_roc(i):
                ax.plot(fpr, tpr, color=_fold_colors[k], lw=1.1, alpha=0.75,
                        label=f'Fold {k} ({a:.3f})')
            fpr_p, tpr_p, _ = roc_curve(ytb[:, i], proba_all[:, i])
            ax.plot(fpr_p, tpr_p, color='#111', lw=2.6,
                    label=f'Pooled ({auc(fpr_p, tpr_p):.3f})')
            ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.2,
                    linestyle=(0, (1, 1)))
            ax.set_title(_disp(cls), fontweight='bold')
            ax.set_xlabel('FPR (1 − Specificity)', fontweight='bold')
            if i == 0:
                ax.set_ylabel('TPR (Sensitivity)', fontweight='bold')
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
                      title='AUC')
        fig.suptitle(f'ROC by Class — {args.n_folds}-Fold (per-fold curves + '
                     f'pooled OOF)', fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "kfold_roc_byclass_folds")

        # PR counterpart
        fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 5.2),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for i, cls in enumerate(class_names):
            ax = axes[i]
            for (k, rec, prec, ap, _pi) in _fold_pr(i):
                ax.plot(rec, prec, color=_fold_colors[k], lw=1.1, alpha=0.75,
                        label=f'Fold {k} ({ap:.3f})')
            prec_p, rec_p, _ = precision_recall_curve(ytb[:, i], proba_all[:, i])
            ax.plot(rec_p, prec_p, color='#111', lw=2.6,
                    label=f'Pooled ({average_precision_score(ytb[:, i], proba_all[:, i]):.3f})')
            ax.set_title(_disp(cls), fontweight='bold')
            ax.set_xlabel('Recall', fontweight='bold')
            if i == 0:
                ax.set_ylabel('Precision', fontweight='bold')
            ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9, title='AUPRC')
        fig.suptitle(f'Precision–Recall by Class — {args.n_folds}-Fold '
                     f'(per-fold curves + pooled OOF)', fontweight='bold',
                     fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "kfold_pr_byclass_folds")

    # ── LAYOUT 2 — pooled vs per-fold, side by side (2 subplots) ─────────────
    if have_perfold:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.2),
                                       sharex=True, sharey=True)
        # left: pooled, one curve per class
        for i, cls in enumerate(class_names):
            fpr_p, tpr_p, _ = roc_curve(ytb[:, i], proba_all[:, i])
            axL.plot(fpr_p, tpr_p, color=_cls_color(cls, i), lw=2.4,
                     linestyle=CLASS_LINESTYLES[i % 3],
                     label=f'{_disp(cls)} ({auc(fpr_p, tpr_p):.3f})')
        axL.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.2, linestyle=(0, (1, 1)))
        axL.set_title('Pooled OOF (all patients once)', fontweight='bold')
        axL.set_xlabel('FPR (1 − Specificity)', fontweight='bold')
        axL.set_ylabel('TPR (Sensitivity)', fontweight='bold')
        axL.legend(loc='lower right', fontsize=9, title='AUC', framealpha=0.95)
        axL.grid(True, alpha=0.3); axL.set_xlim([0, 1]); axL.set_ylim([0, 1.02])
        # right: all class×fold curves, colored by class, faint
        for i, cls in enumerate(class_names):
            for j, (k, fpr, tpr, a, _ti) in enumerate(_fold_roc(i)):
                axR.plot(fpr, tpr, color=_cls_color(cls, i), lw=1.0, alpha=0.55,
                         label=_disp(cls) if j == 0 else None)
        axR.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.2, linestyle=(0, (1, 1)))
        axR.set_title(f'All {len(folds_avail)} folds × {n_cls} classes',
                      fontweight='bold')
        axR.set_xlabel('FPR (1 − Specificity)', fontweight='bold')
        axR.legend(loc='lower right', fontsize=9, framealpha=0.95)
        axR.grid(True, alpha=0.3)
        fig.suptitle(f'ROC — Pooled vs Per-Fold ({args.n_folds}-Fold, One-vs-Rest)',
                     fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "kfold_roc_pooled_vs_folds")

        # PR counterpart
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.2),
                                       sharex=True, sharey=True)
        for i, cls in enumerate(class_names):
            prec_p, rec_p, _ = precision_recall_curve(ytb[:, i], proba_all[:, i])
            axL.plot(rec_p, prec_p, color=_cls_color(cls, i), lw=2.4,
                     linestyle=CLASS_LINESTYLES[i % 3],
                     label=f'{_disp(cls)} ({average_precision_score(ytb[:, i], proba_all[:, i]):.3f})')
        axL.set_title('Pooled OOF (all patients once)', fontweight='bold')
        axL.set_xlabel('Recall', fontweight='bold')
        axL.set_ylabel('Precision', fontweight='bold')
        axL.legend(loc='lower left', fontsize=9, title='AUPRC', framealpha=0.95)
        axL.grid(True, alpha=0.3); axL.set_xlim([0, 1.02]); axL.set_ylim([0, 1.02])
        for i, cls in enumerate(class_names):
            for j, (k, rec, prec, ap, _pi) in enumerate(_fold_pr(i)):
                axR.plot(rec, prec, color=_cls_color(cls, i), lw=1.0, alpha=0.55,
                         label=_disp(cls) if j == 0 else None)
        axR.set_title(f'All {len(folds_avail)} folds × {n_cls} classes',
                      fontweight='bold')
        axR.set_xlabel('Recall', fontweight='bold')
        axR.legend(loc='lower left', fontsize=9, framealpha=0.95)
        axR.grid(True, alpha=0.3)
        fig.suptitle(f'Precision–Recall — Pooled vs Per-Fold ({args.n_folds}-Fold)',
                     fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "kfold_pr_pooled_vs_folds")

    # ── LAYOUT 3 — per class: faint folds + bold MEAN + ±1 std band ──────────
    if have_perfold:
        fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 5.2),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for i, cls in enumerate(class_names):
            ax = axes[i]
            rocs = _fold_roc(i)
            for (k, fpr, tpr, a, _ti) in rocs:
                ax.plot(fpr, tpr, color='#9E9E9E', lw=0.9, alpha=0.6)
            if rocs:
                tpr_stack = np.vstack([r[4] for r in rocs])
                tpr_mean = tpr_stack.mean(0); tpr_mean[-1] = 1.0
                tpr_std = tpr_stack.std(0)
                mean_auc = np.mean([r[3] for r in rocs])
                std_auc = np.std([r[3] for r in rocs])
                ax.fill_between(_grid, np.clip(tpr_mean - tpr_std, 0, 1),
                                np.clip(tpr_mean + tpr_std, 0, 1),
                                color=_cls_color(cls, i), alpha=0.22,
                                label='±1 std')
                ax.plot(_grid, tpr_mean, color=_cls_color(cls, i), lw=2.6,
                        label=f'Mean ({mean_auc:.3f} ± {std_auc:.3f})')
            fpr_p, tpr_p, _ = roc_curve(ytb[:, i], proba_all[:, i])
            ax.plot(fpr_p, tpr_p, color='#111', lw=1.8, linestyle=(0, (4, 2)),
                    label=f'Pooled ({auc(fpr_p, tpr_p):.3f})')
            ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.2,
                    linestyle=(0, (1, 1)))
            ax.set_title(_disp(cls), fontweight='bold')
            ax.set_xlabel('FPR (1 − Specificity)', fontweight='bold')
            if i == 0:
                ax.set_ylabel('TPR (Sensitivity)', fontweight='bold')
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8, framealpha=0.92)
        fig.suptitle(f'ROC by Class — Mean ± std with Pooled Overlay '
                     f'({args.n_folds}-Fold)', fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "kfold_roc_mean_band")

        # PR counterpart
        fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 5.2),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for i, cls in enumerate(class_names):
            ax = axes[i]
            prs = _fold_pr(i)
            for (k, rec, prec, ap, _pi) in prs:
                ax.plot(rec, prec, color='#9E9E9E', lw=0.9, alpha=0.6)
            if prs:
                prec_stack = np.vstack([r[4] for r in prs])
                prec_mean = prec_stack.mean(0)
                prec_std = prec_stack.std(0)
                mean_ap = np.mean([r[3] for r in prs])
                std_ap = np.std([r[3] for r in prs])
                ax.fill_between(_grid, np.clip(prec_mean - prec_std, 0, 1),
                                np.clip(prec_mean + prec_std, 0, 1),
                                color=_cls_color(cls, i), alpha=0.22,
                                label='±1 std')
                ax.plot(_grid, prec_mean, color=_cls_color(cls, i), lw=2.6,
                        label=f'Mean ({mean_ap:.3f} ± {std_ap:.3f})')
            prec_p, rec_p, _ = precision_recall_curve(ytb[:, i], proba_all[:, i])
            ax.plot(rec_p, prec_p, color='#111', lw=1.8, linestyle=(0, (4, 2)),
                    label=f'Pooled ({average_precision_score(ytb[:, i], proba_all[:, i]):.3f})')
            ax.set_title(_disp(cls), fontweight='bold')
            ax.set_xlabel('Recall', fontweight='bold')
            if i == 0:
                ax.set_ylabel('Precision', fontweight='bold')
            ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=8, framealpha=0.92)
        fig.suptitle(f'Precision–Recall by Class — Mean ± std with Pooled Overlay '
                     f'({args.n_folds}-Fold)', fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "kfold_pr_mean_band")
else:
    print("  ⚠ No pooled probabilities — skipping pooled ROC/PR.")


# ── t-SNE handling (cannot be pooled — different model per fold) ───────────────
# Each fold's training run already produced tsne_features.{pdf,png,svg} under
#   output_figures/fold_{k}/evaluation/
# We do TWO things here:
#   (a) copy the REPRESENTATIVE fold's t-SNE (fold whose accuracy is closest to
#       the mean) into the summary folder as the paper's main t-SNE figure;
#   (b) build a 5-panel grid of all folds' t-SNE for the supplementary material.
# Neither pools embeddings (which would be meaningless across distinct models).

FIGURES_BASE = OUT_DIR.parent          # e.g. output_figures/
def _fold_tsne_path(fold_k, ext='png'):
    return FIGURES_BASE / f"fold_{fold_k}" / "evaluation" / f"tsne_features.{ext}"

if rows:
    # (a) Representative fold = accuracy closest to the mean accuracy
    valid = [(r['fold'], r['test_accuracy']) for r in rows
             if not np.isnan(r.get('test_accuracy', np.nan))]
    if valid:
        mean_acc = np.mean([a for _, a in valid])
        rep_fold = min(valid, key=lambda fa: abs(fa[1] - mean_acc))[0]
        copied = False
        for ext in ['pdf', 'png', 'svg']:
            src = _fold_tsne_path(rep_fold, ext)
            if src.exists():
                shutil.copy(src, OUT_DIR / f"kfold_tsne_representative.{ext}")
                copied = True
        if copied:
            print(f"  ✓ kfold_tsne_representative.* "
                  f"(copied from Fold {rep_fold}, acc nearest mean)")
            summary_lines.append("")
            summary_lines.append(
                f"t-SNE: representative figure is Fold {rep_fold} "
                f"(test accuracy {dict(valid)[rep_fold]:.4f}, closest to the "
                f"{mean_acc:.4f} mean). t-SNE is NOT pooled across folds because "
                f"each fold is a distinct model with its own feature space.")
            summary_text = "\n".join(summary_lines)
        else:
            print(f"  ⚠ Fold {rep_fold} t-SNE image not found "
                  f"({_fold_tsne_path(rep_fold)}); skipping representative copy.")

        # (b) 5-panel grid of all folds' t-SNE (supplementary)
        avail = [(r['fold'], r['test_accuracy']) for r in rows
                 if _fold_tsne_path(r['fold']).exists()]
        if avail:
            n = len(avail)
            ncol = min(3, n)
            nrow = int(np.ceil(n / ncol))
            # Taller cells + headroom so each embedded PNG (which carries its
            # OWN title/axes) doesn't collide with the panel label above the
            # row below it.
            fig, axes = plt.subplots(nrow, ncol,
                                     figsize=(5.2 * ncol, 5.4 * nrow))
            axes = np.atleast_1d(axes).ravel()
            for ax in axes:
                ax.axis('off')
            for idx, (fk, fa) in enumerate(avail):
                img = plt.imread(str(_fold_tsne_path(fk)))
                axes[idx].imshow(img)
                axes[idx].axis('off')
                is_rep = (fk == rep_fold)
                # Subtle, non-alarming cue: a small star prefix and a muted
                # accent colour — not alarm-red, which reads like an error.
                tag = "  \u2605" if is_rep else ""
                title = f"Fold {fk}  (acc = {fa:.3f}){tag}"
                axes[idx].set_title(
                    title, fontsize=12, fontweight='bold', pad=10,
                    color=('#1A5276' if is_rep else '#222'))
            fig.suptitle(
                f"t-SNE of Penultimate Features — All {args.n_folds} Folds\n"
                f"(\u2605 = representative fold, acc nearest mean; distinct model "
                f"per fold, so embeddings are not comparable across panels)",
                fontsize=13, fontweight='bold')
            # Reserve generous vertical gaps between rows; the embedded PNGs
            # already include their own captions/axes.
            fig.subplots_adjust(top=0.90, bottom=0.02, hspace=0.28, wspace=0.06)
            _save(fig, "kfold_tsne_grid")
        else:
            print("  ⚠ No per-fold t-SNE images found; skipping t-SNE grid. "
                  "(Run training so each fold writes "
                  "output_figures/fold_k/evaluation/tsne_features.png.)")


# ── Aggregated PATIENT-LEVEL diagnosis across all folds ───────────────────────
# Reads each fold's patient_level_analysis.csv (written by 03_post_analysis.py),
# pools every patient (each patient is tested in exactly ONE fold, so the union
# is leakage-free and each patient appears once), and produces a two-panel
# figure mirroring the per-fold patient_level_rescue chart:
#   (top)    pooled error-patient bar chart — only patients with >=1 wrong pair,
#            one bar each, sorted by wrong-pair count (same logic as per-fold);
#   (bottom) compact per-fold summary — error-free / rescued / still-misdiagnosed
#            stacked counts per fold.
import csv as _csv

def _find_patient_csvs():
    """Locate each fold's patient_level_analysis.csv under output_figures/.
    Path pattern: <FIGURES_BASE>/post_analysis/fold_{k}/run_*/patient_level/."""
    found = {}
    pa_root = FIGURES_BASE / "post_analysis"
    if not pa_root.exists():
        return found
    for k in range(1, args.n_folds + 1):
        cands = sorted(pa_root.glob(
            f"fold_{k}/run_*/patient_level/patient_level_analysis.csv"))
        if cands:
            found[k] = cands[-1]      # newest run for this fold
    return found

_pt_csvs = _find_patient_csvs()
if not _pt_csvs:
    print("  ℹ No per-fold patient_level_analysis.csv found under "
          f"{FIGURES_BASE}/post_analysis/ — run 03_post_analysis.py --patient "
          "(or --all) per fold first; skipping aggregated patient-level figure.")
else:
    def _to_int(x, d=0):
        try:    return int(float(x))
        except (TypeError, ValueError): return d
    def _to_bool(x):
        return str(x).strip().lower() in ('true', '1', 'yes')

    pooled = []          # one dict per patient (errors-only kept for top panel)
    per_fold = {}        # k -> {'error_free':, 'rescued':, 'still_wrong':, 'total':}
    for k in sorted(_pt_csvs):
        ef = res = sw = tot = 0
        with open(_pt_csvs[k], newline='') as f:
            for r in _csv.DictReader(f):
                tot += 1
                n_wrong   = _to_int(r.get('n_wrong_images'))
                n_correct = _to_int(r.get('n_correct_images'))
                vote_ok   = _to_bool(r.get('vote_correct'))
                if n_wrong == 0:
                    ef += 1
                elif vote_ok:
                    res += 1
                else:
                    sw += 1
                if n_wrong > 0:        # errors-only, like the per-fold chart
                    pooled.append({
                        'fold': k,
                        'patient': r.get('patient', '?'),
                        'majority_vote': r.get('majority_vote', '?'),
                        'vote_correct': vote_ok,
                        'n_cr': _to_int(r.get('n_cr_images')),
                        'n_sg': _to_int(r.get('n_sg_images')),
                        'n_correct': n_correct,
                        'n_wrong': n_wrong,
                    })
        per_fold[k] = {'error_free': ef, 'rescued': res,
                       'still_wrong': sw, 'total': tot}

    tot_pat   = sum(v['total']       for v in per_fold.values())
    tot_ef    = sum(v['error_free']  for v in per_fold.values())
    tot_res   = sum(v['rescued']     for v in per_fold.values())
    tot_sw    = sum(v['still_wrong'] for v in per_fold.values())
    pl_acc    = (tot_pat - tot_sw) / tot_pat if tot_pat else float('nan')

    summary_lines.append("")
    summary_lines.append(f"PATIENT-LEVEL (pooled across {len(per_fold)} folds): "
                         f"{tot_pat} patients — {tot_ef} error-free, "
                         f"{tot_res} rescued by aggregation, {tot_sw} still "
                         f"misdiagnosed (patient-level accuracy {pl_acc:.1%}).")
    summary_text = "\n".join(summary_lines)

    C_OK, C_BAD = '#2E7D32', '#B2182B'
    n_err = len(pooled)
    # Group by patient-level outcome so the clinically important failures stand
    # out: still-misdiagnosed (✗) first, then rescued (✓); within each group,
    # most misclassified pairs first. n_still tells us where to draw the divider.
    pooled.sort(key=lambda d: (d['vote_correct'], -d['n_wrong'],
                               d['fold'], d['patient']))
    n_still = sum(1 for d in pooled if not d['vote_correct'])

    fig = plt.figure(figsize=(11.5, max(0.42 * n_err, 2.0) + 4.2))
    gs  = fig.add_gridspec(2, 1, height_ratios=[max(n_err, 4), 5],
                           hspace=0.18)

    # ── Top panel: pooled error patients ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    if n_err:
        ypos = np.arange(n_err)
        cor  = [d['n_correct'] for d in pooled]
        wrg  = [d['n_wrong']   for d in pooled]
        labels = [f"f{d['fold']}·{d['patient']} → {d['majority_vote']} "
                  f"{'✓' if d['vote_correct'] else '✗'}"
                  f"  [{d['n_cr']} CR, {d['n_sg']} SG]" for d in pooled]
        ax0.barh(ypos, cor, color=C_OK, label='Correct pairs',
                 edgecolor='black', linewidth=0.7)
        ax0.barh(ypos, wrg, left=cor, color=C_BAD, label='Misclassified pairs',
                 edgecolor='black', linewidth=0.7, hatch='xxx')
        ax0.set_yticks(ypos)
        ax0.set_yticklabels(labels, fontsize=8)
        # Colour each y-label by outcome: red = still misdiagnosed, green = rescued
        for tick, d in zip(ax0.get_yticklabels(), pooled):
            tick.set_color(C_BAD if not d['vote_correct'] else '#1B5E20')
        ax0.invert_yaxis()
        # Divider + group annotations between the still-misdiagnosed and rescued
        if 0 < n_still < n_err:
            ax0.axhline(n_still - 0.5, color='#444', linewidth=1.2,
                        linestyle='--')
            ax0.annotate(f"Still misdiagnosed ({n_still})",
                         xy=(0.99, 0.5 / n_err), xycoords='axes fraction',
                         ha='right', va='bottom', fontsize=9, fontweight='bold',
                         color=C_BAD)
            ax0.annotate(f"Rescued by aggregation ({n_err - n_still})",
                         xy=(0.99, (n_still + 0.5) / n_err),
                         xycoords='axes fraction', ha='right', va='bottom',
                         fontsize=9, fontweight='bold', color='#1B5E20')
        ax0.legend(loc='lower right', fontsize=9)
    else:
        ax0.text(0.5, 0.5, "No patient had a misclassified pair in any fold.",
                 ha='center', va='center', fontsize=12, transform=ax0.transAxes)
        ax0.set_yticks([])
    ax0.set_xlabel("Number of coronal×sagittal image pairs", fontweight='bold')
    ax0.set_title(
        f"Patient-level diagnosis, pooled across {len(per_fold)} folds  —  "
        f"{tot_pat} patients total\n{tot_ef} error-free  |  {tot_res} rescued "
        f"by aggregation  |  {tot_sw} still misdiagnosed   "
        f"(patient-level accuracy {pl_acc:.1%}; bars = the {n_err} patients with "
        f"\u22651 misclassified pair, i.e. rescued + still-misdiagnosed)",
        fontweight='bold', fontsize=12)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    # ── Bottom panel: per-fold stacked summary ──────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ks   = sorted(per_fold)
    xpos = np.arange(len(ks))
    ef_v = [per_fold[k]['error_free']  for k in ks]
    rs_v = [per_fold[k]['rescued']     for k in ks]
    sw_v = [per_fold[k]['still_wrong'] for k in ks]
    b1 = ax1.bar(xpos, ef_v, color=C_OK,      edgecolor='black', linewidth=0.7,
                 label='Error-free')
    b2 = ax1.bar(xpos, rs_v, bottom=ef_v, color='#7FB069', edgecolor='black',
                 linewidth=0.7, label='Rescued by aggregation')
    b3 = ax1.bar(xpos, sw_v, bottom=[a + b for a, b in zip(ef_v, rs_v)],
                 color=C_BAD, edgecolor='black', linewidth=0.7, hatch='xxx',
                 label='Still misdiagnosed')
    for k_i, k in enumerate(ks):
        for seg, base in ((ef_v, 0), (rs_v, ef_v[k_i]),
                          (sw_v, ef_v[k_i] + rs_v[k_i])):
            v = seg[k_i]
            if v:
                ax1.text(k_i, base + v / 2, str(v), ha='center', va='center',
                         fontsize=10, fontweight='bold',
                         color='white' if seg is not rs_v else '#1A3D1A')
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"Fold {k}\n(n={per_fold[k]['total']})" for k in ks],
                        fontsize=11, fontweight='bold')
    ax1.set_ylabel("Patients", fontweight='bold')
    ax1.set_title("Per-fold patient-level outcome", fontsize=12,
                  fontweight='bold', pad=28)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=9,
               ncol=3, framealpha=0.95, borderaxespad=0)
    # Headroom so the tallest stack doesn't crowd the legend/title
    _ymax = max((per_fold[k]['total'] for k in ks), default=1)
    ax1.set_ylim(0, _ymax * 1.18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(True, axis='y', alpha=0.3)

    _save(fig, "kfold_patient_level_pooled")


# ── Aggregated DATASET COMPOSITION across all folds ───────────────────────────
# Reads each fold's dataset_composition.csv (written by 03_post_analysis.py).
# IMPORTANT: train/val sets OVERLAP across folds (a patient trains in 4 folds),
# so summing train/val across folds would multiply-count patients and is wrong.
# Only the TEST split pools cleanly (each patient is tested in exactly one fold),
# and the pooled test composition equals the full study cohort. We therefore:
#   (top-left)  pooled TEST composition = full cohort (patients + images/class);
#   (top-right) per-fold TEST patients per class (the leakage-free per-fold view);
#   (bottom)    per-fold train/val/test patient totals SIDE BY SIDE (not summed),
#               with an explicit note that train/val intentionally repeat patients.
def _find_dataset_csvs():
    found = {}
    pa_root = FIGURES_BASE / "post_analysis"
    if not pa_root.exists():
        return found
    for k in range(1, args.n_folds + 1):
        cands = sorted(pa_root.glob(
            f"fold_{k}/run_*/dataset/dataset_composition.csv"))
        if cands:
            found[k] = cands[-1]
    return found

_ds_csvs = _find_dataset_csvs()
if not _ds_csvs:
    print("  ℹ No per-fold dataset_composition.csv found under "
          f"{FIGURES_BASE}/post_analysis/ — run 03_post_analysis.py --dataset "
          "(or --all) per fold first; skipping aggregated dataset figure.")
else:
    def _to_int(x, d=0):
        try:    return int(float(x))
        except (TypeError, ValueError): return d

    # ds[k][split][class] = {'patients','coronal','sagittal','images'}
    ds = {}
    classes_ds = ['healthy', 'lesion', 'mrc']
    splits_ds  = ['train', 'val', 'test']
    for k in sorted(_ds_csvs):
        ds[k] = {sp: {c: {'patients': 0, 'coronal': 0, 'sagittal': 0,
                          'images': 0} for c in classes_ds} for sp in splits_ds}
        with open(_ds_csvs[k], newline='') as f:
            for r in _csv.DictReader(f):
                sp, c = r.get('split', ''), r.get('class', '')
                if sp in ds[k] and c in ds[k][sp]:
                    ds[k][sp][c] = {
                        'patients':  _to_int(r.get('patients')),
                        'coronal':   _to_int(r.get('coronal')),
                        'sagittal':  _to_int(r.get('sagittal')),
                        'images':    _to_int(r.get('coronal')) + _to_int(r.get('sagittal')),
                    }
    ks_ds = sorted(ds)

    # Pooled TEST composition = full cohort (each patient tested once)
    pooled_test = {c: {'patients': 0, 'coronal': 0, 'sagittal': 0, 'images': 0}
                   for c in classes_ds}
    for k in ks_ds:
        for c in classes_ds:
            for key in ('patients', 'coronal', 'sagittal', 'images'):
                pooled_test[c][key] += ds[k]['test'][c][key]

    tot_patients = sum(pooled_test[c]['patients'] for c in classes_ds)
    tot_images   = sum(pooled_test[c]['images']   for c in classes_ds)
    summary_lines.append("")
    summary_lines.append(
        f"DATASET (pooled TEST = full cohort, each patient once): "
        f"{tot_patients} patients, {tot_images} images "
        + ", ".join(f"{_disp(c)} {pooled_test[c]['patients']}p/"
                    f"{pooled_test[c]['images']}img" for c in classes_ds) + ".")
    summary_text = "\n".join(summary_lines)

    fig = plt.figure(figsize=(13.5, 9.5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[5, 4.5], hspace=0.48, wspace=0.42)
    cls_col   = [_cls_color(c, i) for i, c in enumerate(classes_ds)]
    cls_hatch = ['', '///', '...']

    # ── (top-left) pooled TEST composition: patients + images per class ──────
    axA = fig.add_subplot(gs[0, 0])
    xc  = np.arange(len(classes_ds)); w = 0.38
    pat_v = [pooled_test[c]['patients'] for c in classes_ds]
    img_v = [pooled_test[c]['images']   for c in classes_ds]
    bA1 = axA.bar(xc - w/2, pat_v, w, label='Patients', color='#3D6A8A',
                  edgecolor='black', linewidth=0.7)
    axA.bar_label(bA1, fontsize=9, fontweight='bold', padding=2)
    axA.set_ylabel('Patients', fontweight='bold', color='#3D6A8A')
    axA.tick_params(axis='y', labelcolor='#3D6A8A')
    axA.set_xticks(xc); axA.set_xticklabels([_disp(c) for c in classes_ds],
                                            fontweight='bold')
    axA_t = axA.twinx()
    bA2 = axA_t.bar(xc + w/2, img_v, w, label='Images (CR+SG)', color='#C77F3D',
                    edgecolor='black', linewidth=0.7)
    axA_t.bar_label(bA2, fontsize=9, fontweight='bold', padding=2)
    axA_t.set_ylabel('Images (CR+SG)', fontweight='bold', color='#C77F3D',
                     labelpad=2)
    axA_t.tick_params(axis='y', labelcolor='#C77F3D')
    axA.set_title(f"Pooled test = full cohort  ({tot_patients} patients, "
                  f"{tot_images} images)", fontweight='bold', fontsize=11)
    axA.spines['top'].set_visible(False); axA_t.spines['top'].set_visible(False)

    # ── (top-right) per-fold TEST patients per class (grouped) ───────────────
    axB = fig.add_subplot(gs[0, 1])
    xf = np.arange(len(ks_ds)); bw = 0.26
    for i, c in enumerate(classes_ds):
        vals = [ds[k]['test'][c]['patients'] for k in ks_ds]
        bars = axB.bar(xf + (i - 1) * bw, vals, bw, label=_disp(c),
                       color=cls_col[i], hatch=cls_hatch[i],
                       edgecolor='black', linewidth=0.7)
        axB.bar_label(bars, fontsize=8, padding=1)
    axB.set_xticks(xf); axB.set_xticklabels([f"Fold {k}" for k in ks_ds],
                                            fontweight='bold')
    axB.set_ylabel('Test patients', fontweight='bold')
    axB.set_title("Per-fold test patients per class\n"
                  "(each patient tested in exactly one fold)",
                  fontweight='bold', fontsize=11)
    axB.legend(fontsize=9, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, -0.12), framealpha=0.95)
    axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

    # ── (bottom) per-fold train/val/test patient totals, side by side ────────
    axC = fig.add_subplot(gs[1, :])
    split_col = {'train': '#6B9080', 'val': '#C49157', 'test': '#8B6B92'}
    sw = 0.26
    for j, sp in enumerate(splits_ds):
        vals = [sum(ds[k][sp][c]['patients'] for c in classes_ds) for k in ks_ds]
        bars = axC.bar(xf + (j - 1) * sw, vals, sw, label=sp.capitalize(),
                       color=split_col[sp], edgecolor='black', linewidth=0.7)
        axC.bar_label(bars, fontsize=8, fontweight='bold', padding=1)
    axC.set_xticks(xf); axC.set_xticklabels([f"Fold {k}" for k in ks_ds],
                                            fontweight='bold')
    axC.set_ylabel('Patients in split', fontweight='bold')
    axC.set_title("Per-fold split sizes (patients), shown side by side — NOT summed\n"
                  "(train/val intentionally reuse patients across folds; only the "
                  "test split partitions the cohort)",
                  fontweight='bold', fontsize=11, pad=24)
    axC.legend(fontsize=9, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, 1.005), framealpha=0.95, borderaxespad=0)
    _ymaxC = max(sum(ds[k][sp][c]['patients'] for c in classes_ds)
                 for k in ks_ds for sp in splits_ds)
    axC.set_ylim(0, _ymaxC * 1.15)
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)
    axC.set_axisbelow(True); axC.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f"Dataset Composition Across {len(ks_ds)} Folds",
                 fontweight='bold', fontsize=14, y=0.98)
    _save(fig, "kfold_dataset_composition")


# ── Write text summary ─────────────────────────────────────────────────────────
with open(OUT_DIR / "kfold_summary_report.txt", 'w') as f:
    f.write(summary_text + "\n")
print(f"✓ {OUT_DIR}/kfold_summary_report.txt")
print("\nDone.")
