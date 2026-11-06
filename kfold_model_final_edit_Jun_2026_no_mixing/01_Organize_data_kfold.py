#!/usr/bin/env python3
"""
============================================================================
DATA PREPARATION — 5-FOLD PATIENT-LEVEL CROSS-VALIDATION + square padding
============================================================================

Design (per your specification):
  • Patient-level, class-stratified K-fold (k=5):
      - Patients of each class are deterministically shuffled and assigned
        round-robin to folds (equivalent in effect to a stratified group
        K-fold: no patient appears in more than one fold — zero leakage —
        and class proportions are balanced across folds).
  • For each fold k (k = 1..5):
        TEST       = patients in fold k
        VALIDATION = a small slice carved out of the remaining 4 folds
        TRAIN      = the rest of the remaining 4 folds
      Every patient is in TEST exactly once across the 5 folds.
  • Pairing convention (enforced later in the TRAINING script, not here):
        TRAIN       → mix_across_patients=True   (max combinatorial pairs)
        VAL / TEST  → mix_across_patients=False  (within-patient, no leakage)

Disk strategy:
  • Each original image is padded to a square ONCE into a canonical pool:
        padded_pool/{class}/{view}/{filename}.jpg
  • Each fold directory is built from SYMLINKS into that pool:
        kfold_data/fold_1/{train,validation,test}/{class}/{view}/...
    → 5 folds cost ~no extra image storage (just symlinks).

Outputs (per fold) match the folder layout the training script already expects:
        kfold_data/fold_{k}/train/{class}/{view}/
        kfold_data/fold_{k}/validation/{class}/{view}/
        kfold_data/fold_{k}/test/{class}/{view}/

Logs / CSVs:
  fold_assignment.csv          — patient → (fold, role) for every fold
  image_manifest.csv           — every padded image in the pool
  image_dimension_stats.csv    — per-image dims (manuscript appendix)
  kfold_summary.csv            — image counts per fold × split × class × view
  leakage_report.txt           — explicit verification per fold

Patient ID convention (MUST match training):
    patient = filename.split('_')[0].strip()
  Patient IDs are class-scoped (same number in different classes = different
  individuals); cross-class checks are intentionally NOT performed.
"""

import os
import sys
import csv
import zlib
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
INPUT_BASE_DIR  = Path('Data_Oct_2025_Data_Reorganized')
POOL_DIR        = Path('padded_pool')          # canonical padded images (real files)
KFOLD_BASE_DIR  = Path('kfold_data')           # per-fold symlink trees

CLASSES = ['healthy', 'lesion', 'mrc']
VIEWS   = ['cr', 'sg']
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ── K-fold configuration ─────────────────────────────────────────────────────
N_FOLDS       = 5
# Fraction of the (train+val) patients to reserve as validation for the
# composite-checkpoint logic in the training script.  ~0.15 of the 4 training
# folds ≈ a half-fold, matching the "3.5 train + 0.5 val" idea.
VAL_FRACTION  = 0.15

# Reproducibility — must match training script
RANDOM_SEED = 9
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Logging
LOG_ROOT = Path('logs')
LOG_DIR  = LOG_ROOT / f"data_prep_kfold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Manuscript figure style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'grid.alpha': 0.35,
})
ROLE_COLORS = {'train': '#2166AC', 'validation': '#F4A582', 'test': '#B2182B'}


# =============================================================================
# Logging helpers
# =============================================================================
class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, 'w')
    def __call__(self, msg: str = ""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

log = Logger(LOG_DIR / 'data_prep_kfold.log')

log("=" * 74)
log("DATA PREPARATION — 5-FOLD PATIENT-LEVEL CROSS-VALIDATION + square padding")
log("=" * 74)
log(f"Timestamp           : {datetime.now().isoformat(timespec='seconds')}")
log(f"Random seed         : {RANDOM_SEED}")
log(f"Input directory     : {INPUT_BASE_DIR.resolve()}")
log(f"Padded pool         : {POOL_DIR.resolve()}")
log(f"K-fold base         : {KFOLD_BASE_DIR.resolve()}")
log(f"Number of folds     : {N_FOLDS}")
log(f"Validation fraction : {VAL_FRACTION}  (carved from the 4 training folds)")
log(f"Classes             : {CLASSES}")
log(f"Views               : {VIEWS}")


# =============================================================================
# Helpers
# =============================================================================
def find_dir_case_insensitive(parent: Path, name: str) -> Path | None:
    if not parent.exists():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def get_image_files(directory: Path) -> list[str]:
    if directory is None or not directory.exists():
        return []
    return sorted([f.name for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS])


def get_person_name(filename: str) -> str:
    """Extract patient ID. MUST match training: filename.split('_')[0]."""
    stem = Path(filename).name
    return stem.split('_', 1)[0].strip() if '_' in stem else stem


def read_image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        log(f"  ! Could not read {path}: {e}")
        return None


def pad_image_to_square(src_path: Path, target_size: int) -> np.ndarray | None:
    """Symmetric zero-pad an image to target_size × target_size."""
    try:
        with Image.open(src_path) as img:
            arr = np.array(img.convert('RGB'))
        if arr.max() == 0:
            log(f"    ! blank image: {src_path}")
            return None
        h, w = arr.shape[:2]
        if w > target_size or h > target_size:
            log(f"    ! image larger than PAD_SIZE: {src_path} ({w}×{h})")
            return None
        pad_w = target_size - w
        pad_h = target_size - h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        return np.pad(arr, ((top, bottom), (left, right), (0, 0)),
                      mode='constant', constant_values=0)
    except Exception as e:
        log(f"    ! error processing {src_path}: {e}")
        return None


# =============================================================================
# Pre-flight: resolve inputs, gather dimensions
# =============================================================================
log("")
log("─" * 74)
log("PRE-FLIGHT — verifying input directories and gathering dimensions")
log("─" * 74)

source_dirs: dict[tuple[str, str], Path | None] = {}
log("\n  Resolving input directories (case-insensitive):")
for cls in CLASSES:
    class_dir = find_dir_case_insensitive(INPUT_BASE_DIR, cls)
    if class_dir is None:
        log(f"    ✗ no class directory matching '{cls}' under {INPUT_BASE_DIR}")
        for view in VIEWS:
            source_dirs[(cls, view)] = None
        continue
    for view in VIEWS:
        view_dir = find_dir_case_insensitive(class_dir, view)
        source_dirs[(cls, view)] = view_dir
        if view_dir is None:
            log(f"    ✗ no view directory matching '{view}' under {class_dir}")
        else:
            log(f"    ✓ {cls}/{view}  →  {view_dir}")

dimension_records: list[dict] = []
all_dims: list[tuple[int, int]] = []

log("")
for cls in CLASSES:
    for view in VIEWS:
        source_dir = source_dirs[(cls, view)]
        if source_dir is None:
            continue
        files = get_image_files(source_dir)
        log(f"  {cls:8s}/{view}: {len(files):4d} images at {source_dir}")
        for fname in files:
            dims = read_image_dimensions(source_dir / fname)
            if dims is not None:
                all_dims.append(dims)
                dimension_records.append({
                    'class': cls, 'view': view, 'filename': fname,
                    'width': dims[0], 'height': dims[1],
                })

if not all_dims:
    log("✗ No readable images found — aborting.")
    log.close()
    sys.exit(1)

widths  = np.array([w for w, _ in all_dims])
heights = np.array([h for _, h in all_dims])
PAD_SIZE = int(max(widths.max(), heights.max()))

log("")
log(f"Image dimension statistics (n={len(all_dims)}):")
log(f"  width  : min={widths.min()}  max={widths.max()}  "
    f"mean={widths.mean():.1f}  median={np.median(widths):.0f}  std={widths.std():.1f}")
log(f"  height : min={heights.min()}  max={heights.max()}  "
    f"mean={heights.mean():.1f}  median={np.median(heights):.0f}  std={heights.std():.1f}")
log(f"  PAD_SIZE chosen   : {PAD_SIZE} × {PAD_SIZE}  (max of all w/h)")

with open(LOG_DIR / 'image_dimension_stats.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['class', 'view', 'filename', 'width', 'height'])
    w.writeheader()
    w.writerows(dimension_records)
log(f"  ✓ image_dimension_stats.csv  ({len(dimension_records)} rows)")


# =============================================================================
# Build patient → image map
# =============================================================================
log("")
log("─" * 74)
log("PATIENT INDEXING")
log("─" * 74)

# {class: {patient: {view: [filenames]}}}
person_images: dict[str, dict[str, dict[str, list[str]]]] = {
    cls: defaultdict(lambda: {v: [] for v in VIEWS}) for cls in CLASSES
}
for cls in CLASSES:
    for view in VIEWS:
        source_dir = source_dirs[(cls, view)]
        if source_dir is None:
            continue
        for fname in get_image_files(source_dir):
            patient = get_person_name(fname)
            person_images[cls][patient][view].append(fname)

for cls in CLASSES:
    n_patients = len(person_images[cls])
    n_cr = sum(len(v['cr']) for v in person_images[cls].values())
    n_sg = sum(len(v['sg']) for v in person_images[cls].values())
    log(f"  {cls:8s}: {n_patients} unique patients  ({n_cr} cr, {n_sg} sg)")
    if n_patients < N_FOLDS:
        log(f"    ⚠ fewer than {N_FOLDS} patients in '{cls}' — k-fold will be degenerate.")


# =============================================================================
# Pad and save ALL images ONCE into the canonical pool
# =============================================================================
log("")
log("─" * 74)
log("PADDING → CANONICAL POOL (real files, written once)")
log("─" * 74)

for cls in CLASSES:
    for view in VIEWS:
        (POOL_DIR / cls / view).mkdir(parents=True, exist_ok=True)

manifest_rows: list[dict] = []
total_processed, total_skipped = 0, 0
# Map (class, patient, view) -> list of absolute padded file paths
pool_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)

for cls in CLASSES:
    for view in VIEWS:
        src_dir = source_dirs[(cls, view)]
        if src_dir is None:
            continue
        out_dir = POOL_DIR / cls / view
        saved = 0
        for fname in get_image_files(src_dir):
            src_path = src_dir / fname
            with Image.open(src_path) as img:
                orig_w, orig_h = img.size
            padded = pad_image_to_square(src_path, PAD_SIZE)
            if padded is None:
                total_skipped += 1
                continue
            out_path = (out_dir / fname).resolve()
            Image.fromarray(padded).save(out_path)
            saved += 1
            total_processed += 1
            patient = get_person_name(fname)
            pool_index[(cls, patient, view)].append(str(out_path))
            manifest_rows.append({
                'class': cls, 'view': view, 'patient_id': patient,
                'filename': fname, 'original_path': str(src_path),
                'padded_path': str(out_path),
                'original_width': orig_w, 'original_height': orig_h,
                'padded_size': PAD_SIZE,
            })
        log(f"  {cls:8s} | {view} : {saved:4d} padded images → {out_dir}")

log("")
log(f"  ✓ pooled: {total_processed}   skipped: {total_skipped}")

with open(LOG_DIR / 'image_manifest.csv', 'w', newline='') as f:
    fields = ['class', 'view', 'patient_id', 'filename', 'original_path',
              'padded_path', 'original_width', 'original_height', 'padded_size']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(manifest_rows)
log(f"  ✓ image_manifest.csv  ({len(manifest_rows)} rows)")


# =============================================================================
# Build the K folds at PATIENT level, per class (no leakage, balanced)
# =============================================================================
log("")
log("─" * 74)
log(f"BUILDING {N_FOLDS} PATIENT-LEVEL FOLDS (per class)")
log("─" * 74)

# For each class, split its patients into N_FOLDS groups deterministically.
# {class: [ [patients_fold0], [patients_fold1], ... ]}
class_folds: dict[str, list[list[str]]] = {}

def _stable_hash(s: str) -> int:
    """Deterministic across Python processes (built-in hash() is salted by
    PYTHONHASHSEED, which silently breaks run-to-run reproducibility)."""
    return zlib.crc32(s.encode('utf-8'))


for cls in CLASSES:
    patients = sorted(person_images[cls].keys())   # deterministic order
    rng = random.Random(RANDOM_SEED + _stable_hash(cls) % 1000)
    rng.shuffle(patients)
    # Round-robin assignment → near-equal fold sizes regardless of count
    folds = [[] for _ in range(N_FOLDS)]
    for i, p in enumerate(patients):
        folds[i % N_FOLDS].append(p)
    class_folds[cls] = folds
    sizes = " / ".join(str(len(f)) for f in folds)
    log(f"  {cls:8s}: {len(patients)} patients → fold sizes [{sizes}]")


# =============================================================================
# Materialize each fold as symlink trees + assign train/val/test roles
# =============================================================================
log("")
log("─" * 74)
log("MATERIALIZING FOLD DIRECTORIES (symlinks) + role assignment")
log("─" * 74)

def link_patient_images(role_dir: Path, cls: str, patient: str):
    """Symlink all pooled images of one patient/class into role_dir/{cls}/{view}."""
    n_linked = 0
    for view in VIEWS:
        dst_dir = role_dir / cls / view
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in pool_index.get((cls, patient, view), []):
            src_path = Path(src)
            dst_path = dst_dir / src_path.name
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(src_path, dst_path)
            n_linked += 1
    return n_linked

fold_assignment_rows: list[dict] = []   # patient → (fold, role)
kfold_counts = {}   # (fold, role, cls, view) -> count
leakage_report = []

for k in range(N_FOLDS):
    fold_name = f"fold_{k+1}"
    log(f"\n  ── {fold_name} ──")
    fold_dir = KFOLD_BASE_DIR / fold_name

    # Track patients per role for leakage verification
    role_patients = {'train': defaultdict(set),
                     'validation': defaultdict(set),
                     'test': defaultdict(set)}

    for cls in CLASSES:
        folds = class_folds[cls]
        test_patients = list(folds[k])

        # Remaining patients = all other folds, pooled then val-sliced
        remaining = [p for j in range(N_FOLDS) if j != k for p in folds[j]]
        rng = random.Random(RANDOM_SEED + 100 * (k + 1) + _stable_hash(cls) % 1000)
        rng.shuffle(remaining)

        n_val = max(1, int(round(len(remaining) * VAL_FRACTION))) if len(remaining) > 1 else 0
        val_patients   = remaining[:n_val]
        train_patients = remaining[n_val:]

        for role, plist in [('train', train_patients),
                            ('validation', val_patients),
                            ('test', test_patients)]:
            for patient in plist:
                link_patient_images(fold_dir / role, cls, patient)
                role_patients[role][cls].add(patient)
                cr = len(pool_index.get((cls, patient, 'cr'), []))
                sg = len(pool_index.get((cls, patient, 'sg'), []))
                kfold_counts[(fold_name, role, cls, 'cr')] = \
                    kfold_counts.get((fold_name, role, cls, 'cr'), 0) + cr
                kfold_counts[(fold_name, role, cls, 'sg')] = \
                    kfold_counts.get((fold_name, role, cls, 'sg'), 0) + sg
                fold_assignment_rows.append({
                    'fold': fold_name, 'role': role, 'class': cls,
                    'patient_id': patient, 'n_coronal': cr, 'n_sagittal': sg,
                    'n_total': cr + sg,
                })

        log(f"    {cls:8s}: train={len(train_patients):3d}  "
            f"val={len(val_patients):3d}  test={len(test_patients):3d} patients")

    # ── Leakage verification within this fold ──
    fold_ok = True
    for cls in CLASSES:
        tr = role_patients['train'][cls]
        va = role_patients['validation'][cls]
        te = role_patients['test'][cls]
        if (tr & va) or (tr & te) or (va & te):
            fold_ok = False
            overlap = (tr & va) | (tr & te) | (va & te)
            leakage_report.append(f"{fold_name} / {cls}: LEAKAGE {sorted(overlap)}")
    status = "✓ no leakage" if fold_ok else "✗ LEAKAGE DETECTED"
    log(f"    {status} across train/val/test in {fold_name}")
    leakage_report.append(f"{fold_name}: {'OK' if fold_ok else 'LEAKAGE'}")

# ── Cross-fold test coverage check: each patient is in TEST exactly once ──
log("")
log("  Verifying each patient appears in TEST exactly once across folds...")
coverage_ok = True
for cls in CLASSES:
    test_counts = defaultdict(int)
    for k in range(N_FOLDS):
        for p in class_folds[cls][k]:
            test_counts[p] += 1
    bad = {p: c for p, c in test_counts.items() if c != 1}
    if bad:
        coverage_ok = False
        log(f"    ✗ {cls}: patients not tested exactly once: {bad}")
    else:
        log(f"    ✓ {cls}: all {len(test_counts)} patients tested exactly once")
leakage_report.append(f"Test coverage exactly-once: {'OK' if coverage_ok else 'FAIL'}")

with open(LOG_DIR / 'leakage_report.txt', 'w') as f:
    f.write("\n".join(leakage_report) + "\n")
log(f"  ✓ leakage_report.txt")

with open(LOG_DIR / 'fold_assignment.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['fold', 'role', 'class', 'patient_id',
                                      'n_coronal', 'n_sagittal', 'n_total'])
    w.writeheader()
    w.writerows(fold_assignment_rows)
log(f"  ✓ fold_assignment.csv  ({len(fold_assignment_rows)} rows)")


# =============================================================================
# Summary CSV + figure
# =============================================================================
log("")
log("─" * 74)
log("SUMMARY (image counts per fold × role × class)")
log("─" * 74)

summary_rows = []
for k in range(N_FOLDS):
    fold_name = f"fold_{k+1}"
    log(f"\n  {fold_name}")
    log("  Role        | Class    |  #CR |  #SG | #Total")
    log("  ------------+----------+------+------+-------")
    for role in ['train', 'validation', 'test']:
        for cls in CLASSES:
            cr = kfold_counts.get((fold_name, role, cls, 'cr'), 0)
            sg = kfold_counts.get((fold_name, role, cls, 'sg'), 0)
            summary_rows.append({
                'fold': fold_name, 'role': role, 'class': cls,
                'n_cr': cr, 'n_sg': sg, 'n_total': cr + sg,
            })
            log(f"  {role:11s} | {cls:8s} | {cr:4d} | {sg:4d} | {cr+sg:5d}")

with open(LOG_DIR / 'kfold_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['fold', 'role', 'class', 'n_cr', 'n_sg', 'n_total'])
    w.writeheader()
    w.writerows(summary_rows)
log(f"\n  ✓ kfold_summary.csv  ({len(summary_rows)} rows)")

# ── Figure: per-fold test patient counts per class ──
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(N_FOLDS)
bar_w = 0.25
class_colors = {'healthy': '#2166AC', 'lesion': '#B2182B', 'mrc': '#1B7837'}
for i, cls in enumerate(CLASSES):
    counts = [len(class_folds[cls][k]) for k in range(N_FOLDS)]
    bars = ax.bar(x + (i - 1) * bar_w, counts, bar_w,
                  label=cls.capitalize(), color=class_colors[cls],
                  edgecolor='#333', linewidth=0.7)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, str(c),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {k+1}" for k in range(N_FOLDS)], fontweight='bold')
ax.set_ylabel('Test patients per class', fontweight='bold')
ax.set_title(f'{N_FOLDS}-Fold Patient-Level Cross-Validation — Test Patients per Fold',
             fontweight='bold')
ax.legend(framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)
plt.tight_layout()
fig_dir = Path('output_figures/data'); fig_dir.mkdir(parents=True, exist_ok=True)
for fmt in ['png', 'pdf', 'svg']:
    fig.savefig(fig_dir / f'kfold_composition.{fmt}')
plt.close(fig)
log(f"  ✓ kfold_composition.[png|pdf|svg] → {fig_dir}/")

log("")
log("=" * 74)
log(f"DONE. {N_FOLDS}-fold structure ready under: {KFOLD_BASE_DIR.resolve()}/")
log(f"      Canonical padded pool: {POOL_DIR.resolve()}/")
log(f"      Logs: {LOG_DIR.resolve()}/")
log("=" * 74)
log("")
log("Next: run training once per fold, e.g.")
log(f"  for k in $(seq 1 {N_FOLDS}); do \\")
log(f"      python 02_model_training_kfold.py --fold $k; \\")
log(f"  done")
log.close()
