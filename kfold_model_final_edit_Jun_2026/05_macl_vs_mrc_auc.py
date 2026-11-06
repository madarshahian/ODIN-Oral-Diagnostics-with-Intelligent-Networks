#!/usr/bin/env python3
"""
05_macl_vs_mrc_auc.py
============================================================================
Pairwise MACL-vs-MRC discrimination on the pooled out-of-fold predictions.

The three one-vs-rest AUCs in the paper mix two questions (e.g. "MACL vs
healthy + MRC"). The clinically important question is narrower:
    "Given that a sinus is pathological, can the model tell MACL from MRC?"

Method
  * Keep only pairs whose TRUE class is MACL ('lesion') or MRC.
    Healthy pairs are excluded.
  * Score each pair with the conditional probability
        s = P(MACL) / (P(MACL) + P(MRC))
    i.e. the model's MACL-vs-MRC preference, ignoring P(healthy).
    (This is the standard pairwise AUC of Hand & Till, 2001.)
    The AUC of the raw P(MACL) score is also printed for comparison.
  * 95% CI by patient-level cluster bootstrap (resample patients, take all
    their pairs), 5,000 resamples, seed 42 -- same scheme as the paper.
  * Also reports per-fold AUC (mean +/- SD, ddof=0 as in the paper) and the
    restricted two-class accuracy (argmax over MACL/MRC only).

Reads: logs/fold_k/run_*/artifacts/test_predictions.pkl  (same files as
03_aggregate_kfold_results.py and 04_bootstrap_ci.py). No GPU, no training.

Usage:
  python 05_macl_vs_mrc_auc.py --logs-dir logs
============================================================================
"""
import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

_PNUM_RE = re.compile(r'(P\d+)', re.IGNORECASE)


def _pnum(nm):
    if not nm or nm == 'BLANK':
        return None
    base = os.path.basename(str(nm))
    m = _PNUM_RE.search(base)
    if m:
        return m.group(1).upper()
    stem = os.path.splitext(base)[0]
    return stem.split('_')[0] if '_' in stem else stem


def load(n_folds, logs_dir):
    rows = []   # (fold, y_true, proba[3], patient_key)
    class_names = None
    for k in range(1, n_folds + 1):
        cands = sorted(Path(logs_dir).glob(f"fold_{k}/run_*/artifacts/test_predictions.pkl"))
        if not cands:
            raise SystemExit(f"fold {k}: no test_predictions.pkl found")
        with open(cands[-1], 'rb') as f:
            d = pickle.load(f)
        cls = [str(c) for c in d['class_names']]
        if class_names is None:
            class_names = cls
        assert cls == class_names, "class order differs between folds"
        yt = np.asarray(d['y_true'])
        pr = np.asarray(d['y_pred_proba'])
        pats = d.get('patients')
        if pats is None:
            raise SystemExit(f"fold {k}: no 'patients' field -> cannot cluster-bootstrap")
        for i in range(len(yt)):
            cr, sg = pats[i][0], pats[i][1]
            nm = cr if (cr and cr != 'BLANK') else sg
            key = f"f{k}|{cls[yt[i]]}|{_pnum(nm) or 'UNK'}"
            rows.append((k, int(yt[i]), pr[i], key))
    return rows, class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--logs-dir', default='logs')
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows, cn = load(args.n_folds, args.logs_dir)
    i_mac, i_mrc = cn.index('lesion'), cn.index('mrc')
    print(f"Pooled pairs (all classes): {len(rows)}   (paper: 6024)")

    sub = [r for r in rows if r[1] in (i_mac, i_mrc)]
    fold = np.array([r[0] for r in sub])
    y = np.array([1 if r[1] == i_mac else 0 for r in sub])        # 1 = MACL
    P = np.vstack([r[2] for r in sub])
    s_cond = P[:, i_mac] / np.clip(P[:, i_mac] + P[:, i_mrc], 1e-12, None)
    s_raw = P[:, i_mac]
    pid = np.array([r[3] for r in sub])
    print(f"MACL pairs: {int(y.sum())}   MRC pairs: {int((1 - y).sum())}   "
          f"(paper: 2608 / 2226)")

    auc_c = roc_auc_score(y, s_cond)
    auc_r = roc_auc_score(y, s_raw)

    # restricted two-class decision (MACL vs MRC only)
    pred2 = (s_cond >= 0.5).astype(int)
    acc2 = (pred2 == y).mean()
    # what the 3-class model actually predicted for these pairs
    pred3 = np.array([int(np.argmax(r[2])) for r in sub])
    cm3 = confusion_matrix([r[1] for r in sub], pred3, labels=[0, 1, 2])

    # per-fold AUC
    fa = []
    for k in sorted(set(fold)):
        m = fold == k
        if len(set(y[m])) == 2:
            fa.append(roc_auc_score(y[m], s_cond[m]))
    fa = np.array(fa)

    # patient-level cluster bootstrap
    rng = np.random.default_rng(args.seed)
    upid = np.unique(pid)
    idx_by_pid = {p: np.where(pid == p)[0] for p in upid}
    boots, boots_acc = [], []
    for _ in range(args.n_boot):
        pick = rng.choice(upid, size=len(upid), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in pick])
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s_cond[idx]))
        boots_acc.append((pred2[idx] == y[idx]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    alo, ahi = np.percentile(boots_acc, [2.5, 97.5])

    print("\n================ MACL vs MRC (pathological pairs only) ================")
    print(f"Patients (resampling unit): {len(upid)}   (paper: 38 MACL + 51 MRC = 89)")
    print(f"Pooled AUC, conditional score P(MACL)/(P(MACL)+P(MRC)): {auc_c:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f}, {len(boots)} cluster-bootstrap resamples)")
    print(f"Pooled AUC, raw P(MACL) score (for comparison):        {auc_r:.3f}")
    print(f"Per-fold AUC: {np.round(fa, 3)}  mean {fa.mean():.3f} +/- {fa.std(ddof=0):.3f}")
    print(f"Two-class accuracy (argmax over MACL/MRC only): {acc2:.3f} "
          f"(95% CI {alo:.3f}-{ahi:.3f})")
    print("\n3-class predictions for these pairs (rows: true, cols: pred; "
          f"order {cn}):")
    print(cm3)
    print("\nSuggested sentence:")
    print(f"  Restricted to the two pathological classes, the pooled MACL-versus-MRC "
          f"AUC was {auc_c:.3f} (95% CI {lo:.3f}-{hi:.3f}; patient-level cluster "
          f"bootstrap).")


if __name__ == '__main__':
    main()
