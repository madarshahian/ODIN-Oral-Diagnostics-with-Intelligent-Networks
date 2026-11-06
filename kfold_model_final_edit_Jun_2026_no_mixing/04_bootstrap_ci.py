#!/usr/bin/env python3
"""
04_bootstrap_ci.py — Patient-level (cluster) bootstrap confidence intervals
===========================================================================
Adds honest 95% confidence intervals to the POOLED out-of-fold metrics
produced by the 5-fold pipeline, for the paper's headline numbers.

WHY A *PATIENT-LEVEL* BOOTSTRAP (and not the naive row bootstrap):
    The pooled predictions are coronal×sagittal *pairs*, and one patient
    contributes many pairs (lesion patients ~120 pairs, healthy ~8). Those
    pairs are NOT independent — they share a patient. Resampling individual
    pairs would treat correlated rows as independent and produce an
    artificially NARROW CI. The statistically correct unit of resampling is
    the PATIENT: we resample patients with replacement, then take all of each
    selected patient's pairs. This yields an honest (slightly wider) CI and is
    the standard "cluster bootstrap".

WHAT IT REPORTS (each with a 95% percentile CI):
    - pooled accuracy
    - pooled macro-F1
    - per-class F1 (healthy / lesion / mrc)   ← lesion CI will be wide; that is
                                                the small-class uncertainty being
                                                reported honestly, not a bug.

IMPORTANT SCOPE CAVEAT (state this in the paper):
    A bootstrap CI quantifies sampling variability WITHIN this one dataset.
    It does NOT capture between-scanner / between-site variation and is
    therefore NOT a substitute for external validation.

USAGE:
    python 04_bootstrap_ci.py --n-folds 5 --logs-dir logs [--n-boot 5000] [--seed 42]

Reads each fold's logs/fold_k/run_*/artifacts/test_predictions.pkl (the same
artifacts the aggregator uses) and appends results to
output_figures/kfold_summary/kfold_bootstrap_ci.txt
"""
import argparse
import os
import re
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


# ----------------------------------------------------------------------------- helpers
_PNUM_RE = re.compile(r'(P\d+)', re.IGNORECASE)


def _patient_number(nm):
    """Extract the patient token (e.g. 'P006') from an image filename.

    Robust to position: searches for a P-number token anywhere in the
    basename rather than assuming it is the first underscore-delimited field.
    Falls back to the first underscore field, then the whole stem, if no
    P-token is present.
    """
    if not nm or nm == 'BLANK':
        return None
    base = os.path.basename(nm)
    m = _PNUM_RE.search(base)
    if m:
        return m.group(1).upper()
    stem = os.path.splitext(base)[0]
    return stem.split('_')[0] if '_' in stem else stem


def _patient_key(fold, class_name, cr_name, sg_name):
    """Stable per-patient key: fold + class + patient-number.

    The same P-number can recur across classes/folds, so we namespace by
    fold+class to keep distinct patients distinct. We prefer the coronal name;
    if the coronal view is BLANK we fall back to the sagittal name.
    """
    nm = cr_name if (cr_name and cr_name != 'BLANK') else sg_name
    pnum = _patient_number(nm) or 'UNK'
    return f"f{fold}|{class_name}|{pnum}"


def _load_pooled(n_folds, logs_dir):
    """Concatenate all folds' OOF predictions, with a patient key per row."""
    y_true, y_pred, pid = [], [], []
    class_names = None
    folds_found = 0
    for k in range(1, n_folds + 1):
        cands = sorted(Path(logs_dir).glob(f"fold_{k}/run_*/artifacts/test_predictions.pkl"))
        if not cands:
            print(f"  ⚠ fold {k}: no test_predictions.pkl found — skipping")
            continue
        with open(cands[-1], 'rb') as f:
            d = pickle.load(f)
        yt = list(d['y_true'])
        yp = list(d['y_pred'])
        cls = list(d.get('class_names', []))
        if class_names is None:
            class_names = cls
        patients = d.get('patients', None)
        if patients is None:
            # No per-row patient info: fall back to a per-row unique key. This
            # degrades the cluster bootstrap to a row bootstrap for this fold;
            # warn loudly because it understates uncertainty.
            print(f"  ⚠ fold {k}: predictions have no 'patients' field — "
                  f"cluster bootstrap unavailable for this fold (rows treated "
                  f"as independent). Re-run training to record patient IDs.")
            keys = [f"f{k}|row{i}" for i in range(len(yt))]
        else:
            keys = []
            for i, (yi, pr) in enumerate(zip(yt, patients)):
                cr_name, sg_name = (pr[0], pr[1]) if isinstance(pr, (list, tuple)) else ('BLANK', 'BLANK')
                cname = cls[yi] if (cls and 0 <= yi < len(cls)) else str(yi)
                keys.append(_patient_key(k, cname, cr_name, sg_name))
        y_true.extend(yt)
        y_pred.extend(yp)
        pid.extend(keys)
        folds_found += 1
    if folds_found == 0:
        raise SystemExit("No fold predictions found; nothing to bootstrap.")
    return (np.asarray(y_true), np.asarray(y_pred),
            np.asarray(pid), class_names, folds_found)


def _metrics(y_true, y_pred, n_cls):
    """Return (accuracy, macro_f1, [per-class f1...]) for one (resampled) set."""
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per = f1_score(y_true, y_pred, labels=list(range(n_cls)),
                   average=None, zero_division=0)
    return acc, mf1, per


def _ci(samples, lo=2.5, hi=97.5):
    return np.percentile(samples, lo), np.percentile(samples, hi)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--logs-dir', default='logs')
    ap.add_argument('--n-boot', type=int, default=5000,
                    help='number of bootstrap resamples (default 5000)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='output_figures/kfold_summary')
    ap.add_argument('--plot', action='store_true',
                    help='also save histograms of the bootstrap distributions '
                         '(kfold_bootstrap_hist.{png,pdf,svg})')
    args = ap.parse_args()

    print(f"Loading pooled OOF predictions from {args.logs_dir}/fold_1..{args.n_folds} ...")
    y_true, y_pred, pid, class_names, folds = _load_pooled(args.n_folds, args.logs_dir)
    n = len(y_true)
    n_cls = len(class_names) if class_names else int(max(y_true.max(), y_pred.max()) + 1)
    uniq_pids = np.unique(pid)
    print(f"  pooled rows (pairs): {n}")
    print(f"  distinct patients  : {len(uniq_pids)}  (resampling unit)")
    print(f"  classes            : {class_names}")
    print(f"  folds used         : {folds}")

    # Pre-group row indices by patient for fast resampling
    groups = {p: np.where(pid == p)[0] for p in uniq_pids}

    # Point estimates on the real pool
    pt_acc, pt_mf1, pt_per = _metrics(y_true, y_pred, n_cls)

    # Bootstrap: resample PATIENTS with replacement, gather their rows
    rng = np.random.default_rng(args.seed)
    boot_acc = np.empty(args.n_boot)
    boot_mf1 = np.empty(args.n_boot)
    boot_per = np.empty((args.n_boot, n_cls))
    pid_array = uniq_pids
    n_pat = len(pid_array)
    for b in range(args.n_boot):
        sample_pids = rng.choice(pid_array, size=n_pat, replace=True)
        rows = np.concatenate([groups[p] for p in sample_pids])
        a, m, per = _metrics(y_true[rows], y_pred[rows], n_cls)
        boot_acc[b] = a
        boot_mf1[b] = m
        boot_per[b] = per
        if (b + 1) % max(1, args.n_boot // 10) == 0:
            print(f"    bootstrap {b + 1}/{args.n_boot}", end='\r')
    print()

    # Assemble report
    acc_lo, acc_hi = _ci(boot_acc)
    mf1_lo, mf1_hi = _ci(boot_mf1)
    lines = []
    lines.append("=" * 68)
    lines.append("PATIENT-LEVEL CLUSTER BOOTSTRAP — 95% CONFIDENCE INTERVALS")
    lines.append("=" * 68)
    lines.append(f"Resampling unit : patient (cluster bootstrap)")
    lines.append(f"Patients        : {n_pat}")
    lines.append(f"Pooled pairs    : {n}")
    lines.append(f"Resamples       : {args.n_boot}   seed={args.seed}")
    lines.append("")
    lines.append(f"Pooled accuracy : {pt_acc:.4f}   95% CI [{acc_lo:.4f}, {acc_hi:.4f}]")
    lines.append(f"Pooled macro-F1 : {pt_mf1:.4f}   95% CI [{mf1_lo:.4f}, {mf1_hi:.4f}]")
    lines.append("")
    lines.append("Per-class F1 (note: small classes have wider CIs — this is")
    lines.append("honest small-sample uncertainty, not a defect):")
    for i in range(n_cls):
        name = class_names[i] if class_names else f"class{i}"
        lo, hi = _ci(boot_per[:, i])
        lines.append(f"  {name:<10s} F1 {pt_per[i]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    lines.append("")
    lines.append("Paper-ready sentence:")
    lines.append(f"  Pooled out-of-fold accuracy was {pt_acc:.3f} "
                 f"(95% CI {acc_lo:.3f}-{acc_hi:.3f}; patient-level cluster")
    lines.append(f"  bootstrap, {args.n_boot} resamples) and macro-F1 was "
                 f"{pt_mf1:.3f} (95% CI {mf1_lo:.3f}-{mf1_hi:.3f}).")
    lines.append("")
    lines.append("CAVEAT: this CI reflects sampling variability within this")
    lines.append("dataset only; it does NOT capture between-scanner / between-")
    lines.append("site variation and is not a substitute for external validation.")
    lines.append("=" * 68)
    report = "\n".join(lines)
    print("\n" + report)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kfold_bootstrap_ci.txt").write_text(report + "\n")
    print(f"\n✓ {out}/kfold_bootstrap_ci.txt")

    # ---------------------------------------------------------------- histograms
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Five panels: accuracy, macro-F1, and the three per-class F1s.
        panels = [("Pooled accuracy", boot_acc, pt_acc, '#3D6A8A')]
        panels.append(("Pooled macro-F1", boot_mf1, pt_mf1, '#6B9080'))
        cls_cols = ['#4A6FA5', '#B2182B', '#2E7D32']
        for i in range(n_cls):
            name = class_names[i] if class_names else f"class{i}"
            panels.append((f"F1 — {name}", boot_per[:, i], pt_per[i],
                           cls_cols[i % len(cls_cols)]))

        ncol = 3
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(panels):]:
            ax.axis('off')

        for ax, (title, samp, point, col) in zip(axes, panels):
            lo, hi = _ci(samp)
            ax.hist(samp, bins=40, color=col, alpha=0.75, edgecolor='white',
                    linewidth=0.4)
            # point estimate (solid) and 95% CI bounds (dashed)
            ax.axvline(point, color='black', lw=2.0,
                       label=f'point = {point:.3f}')
            ax.axvline(lo, color='#555', lw=1.4, linestyle='--')
            ax.axvline(hi, color='#555', lw=1.4, linestyle='--',
                       label=f'95% CI [{lo:.3f}, {hi:.3f}]')
            # shade the central 95%
            ax.axvspan(lo, hi, color=col, alpha=0.10)
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_xlabel('Bootstrap value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8, framealpha=0.9, loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle(
            f"Patient-level cluster bootstrap distributions "
            f"({args.n_boot} resamples, {n_pat} patients)\n"
            f"Solid = point estimate · dashed = 2.5/97.5 percentiles · "
            f"shaded = 95% CI",
            fontweight='bold', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / f"kfold_bootstrap_hist.{ext}", dpi=200,
                        bbox_inches='tight')
        plt.close(fig)
        print(f"✓ {out}/kfold_bootstrap_hist.png / .pdf / .svg")


if __name__ == "__main__":
    main()
