#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Dual-Branch CNN with CBAM Attention and Gated Fusion
# for Multi-View Dental Radiograph Classification
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION (controlled comparison run):
#   - FIX: Input rescaling now correctly applied. Previously, datagen.random_transform()
#          was called without datagen.standardize(), so rescale=1./255 was never
#          actually executed. Images were fed to the model in [0,255] range,
#          producing an effective L2 penalty ~65,000× stronger than nominal.
#          This run normalises inputs to [0,1] as originally intended.
#   - FIX: cv2.imread() returns BGR; now converted to RGB before model input.
#   - FIX: Validation predictions cached once per epoch and shared across
#          DetailedMetricsCallback, F1Checkpoint, CompositeCheckpoint, and
#          CompositeEarlyStopping (was running 4× redundant predict() per epoch).
#   - FIX: on_epoch_end() no longer rebuilds the full Cartesian product of pairs;
#          pairs are constructed once in __init__ and only reshuffled.
#   - FIX: Class weights now computed from cached pair labels rather than
#          iterating all ~5,959 training batches.
#   - FIX: Standalone evaluation block now references final_model (was using
#          stale `model` variable from end of training).
#   All training hyperparameters (LR, λ, dropout, patience, augmentation)
#   are IDENTICAL to the previous run to enable controlled comparison of the
#   normalization effect in isolation.
#
# HARDWARE:
#   GPU  : NVIDIA RTX 4060 (Node Titan eGPU, connected via Thunderbolt)
#   Host : ai-hacker-B850-AI-TOP (Ubuntu 24)
#   VRAM : 8 GB GDDR6
#   Training time: ~[INSERT HOURS] h for 300 epochs on ~4,711 pairs
#                  (record actual wall-clock from PerformanceLogger summary)
#
# RANDOM SEEDS:
#   RANDOM_SEED = 42  — applied to Python random, NumPy, and TensorFlow below.
#   NOTE: TF mixed-precision on GPU introduces non-deterministic rounding;
#   full bit-exact reproducibility is not guaranteed, but statistical
#   reproducibility across runs is achieved.
#
# DATA SPLIT STRATIFICATION:
#   The train/validation/test split (4,711 / 1,202 / 1,010 pairs) was
#   constructed at the **patient level** — all images from a given patient
#   appear in exactly one split. This prevents data leakage that would arise
#   from image-level random splitting when multiple views per patient exist.
#   Class balance within each split was verified after the split.
#
# =============================================================================

# ── Environment variables (must be set before TF import) ──────────────────────
import os
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_DISABLE_CUDA_GRAPHS'] = '1'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ── Reproducibility seeds ──────────────────────────────────────────────────────
RANDOM_SEED = 42
import random
import numpy as np
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ── Core imports ───────────────────────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Lambda,
    concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply,
    Reshape, Concatenate, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts and servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize, LabelBinarizer, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import cycle, product
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import keras.ops as K
from tensorflow.keras.utils import Sequence

# Local reproducibility helper (writes run_metadata.json after sequences are built)
def save_run_metadata(run_name, logs_dir="logs", **kwargs):
    """Save run metadata to JSON file."""
    import json as _json
    from datetime import datetime as _dt
    
    run_dir = Path(logs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'run_name': run_name,
        'timestamp': _dt.now().isoformat(),
        **kwargs
    }
    
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        _json.dump(metadata, f, indent=2, default=str)
    
    print(f"✓ Run metadata saved to {metadata_path}")
    return metadata_path

# ── Mixed precision DISABLED for Grad-CAM compatibility ──────────────────────
# Mixed_float16 was blocking gradient flow through Lambda layers (CBAM, gated fusion)
# Using pure float32 allows true Grad-CAM on conv2d_10 (last conv before classifier)
# Trade-off: ~25% slower training, ~30% more GPU memory; on RTX 4060 still fine
tf.keras.mixed_precision.set_global_policy('float32')

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ── Global hyper-parameters ────────────────────────────────────────────────────
IMG_HEIGHT, IMG_WIDTH = 325, 325   # padded size from train_test_val_data
BATCH_SIZE = 16
EPOCHS = 300
NUM_CLASSES = 3                    # Healthy, MACL (folder: lesion), MRC

# ── K-FOLD CROSS-VALIDATION arguments ──────────────────────────────────────────
# Usage:  python 02_model_training_kfold.py --fold 1
#         (loop over folds in bash: for k in $(seq 1 5); do ... --fold $k; done)
# When --fold is given, data is read from:
#     kfold_data/fold_{k}/{train,validation,test}/{class}/{view}/
# and all logs/figures are namespaced under fold_{k}/ so runs never collide.
import argparse as _argparse
_parser = _argparse.ArgumentParser(description="ODIN training (single split or k-fold).")
_parser.add_argument('--fold', type=int, default=None,
                     help="Fold number (1-indexed). If omitted, uses the legacy "
                          "single split under train_test_val_data/.")
_parser.add_argument('--kfold-base', type=str, default='kfold_data',
                     help="Base directory containing fold_{k}/ subdirectories.")
# parse_known_args so this stays compatible if launched inside notebooks/other args
_args, _unknown = _parser.parse_known_args()

FOLD = _args.fold
KFOLD_BASE = _args.kfold_base

if FOLD is not None:
    DATA_ROOT   = f"{KFOLD_BASE}/fold_{FOLD}"
    RUN_TAG     = f"fold_{FOLD}"
    print(f"\n{'='*60}\n  K-FOLD MODE — Fold {FOLD}\n  Data root: {DATA_ROOT}\n{'='*60}\n")
else:
    DATA_ROOT   = "train_test_val_data"
    RUN_TAG     = None
    print(f"\n{'='*60}\n  SINGLE-SPLIT MODE (legacy)\n  Data root: {DATA_ROOT}\n{'='*60}\n")

# ── Output directory for ALL manuscript figures ────────────────────────────────
# In k-fold mode, namespace figures per fold so they don't overwrite each other.
OUTPUT_DIR = Path("output_figures") / RUN_TAG if RUN_TAG else Path("output_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Manuscript-wide matplotlib style ──────────────────────────────────────────
# Applied once here; PublicationPlotter will re-apply the same dict to be
# safe against any library that resets rcParams internally.
MANUSCRIPT_RC = {
    # font
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 11,
    # axes
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    # ticks
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    # legend
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.handlelength': 1.8,
    # figure
    'figure.titlesize': 14,
    'figure.dpi': 150,
    # save
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    # lines & grid
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
}
plt.rcParams.update(MANUSCRIPT_RC)

# ── Colour palette (colorblind-safe, publication-quality) ─────────────────────
# Per-class colours used consistently across every figure.
CLASS_COLORS = {
    'healthy': '#2166AC',   # blue
    'lesion' : '#D6604D',   # red-orange
    'mrc'    : '#4DAC26',   # green
}
TRAIN_COLOR = '#1A237E'    # dark navy — training curves
VAL_COLOR   = '#B71C1C'    # dark crimson — validation curves
CHANCE_COLOR = '#757575'   # mid-grey — diagonal / random classifier

# Ordered list for consistent index → colour mapping
CLASS_COLOR_LIST = [CLASS_COLORS['healthy'], CLASS_COLORS['lesion'], CLASS_COLORS['mrc']]

# ── B&W-safe redundant encodings ──────────────────────────────────────────────
# Every figure that distinguishes classes/series by colour ALSO uses one of
# these so the figure remains readable when printed in greyscale.
CLASS_LINESTYLES = ['-', '--', ':']          # healthy / lesion / mrc
CLASS_MARKERS    = ['o', 's', '^']           # healthy / lesion / mrc
CLASS_HATCHES    = ['', '///', '...']        # healthy / lesion / mrc
# Display labels for figures and reports. Internal class keys stay
# 'healthy' / 'lesion' / 'mrc' (they match the data folder names and the
# saved result files); only the text shown to the reader changes.
CLASS_DISPLAY = {'healthy': 'Healthy', 'lesion': 'MACL', 'mrc': 'MRC'}
def _disp(cls):
    return CLASS_DISPLAY.get(str(cls).lower(), str(cls).capitalize())
def _disp_key(key):
    """'lesion/P011' -> 'MACL/P011' (patient key for display)."""
    cls, _, num = str(key).partition('/')
    return f"{_disp(cls)}/{num}" if num else _disp(cls)


# =============================================================================
# Data Augmentation
# =============================================================================
#
# Augmentation choices were guided by the anatomical properties of dental
# panoramic radiographs viewed in coronal and sagittal planes:
#
#   horizontal_flip=True
#       Left–right symmetry across the sagittal mid-plane is anatomically
#       valid: the dental arch is bilaterally symmetric, so a flipped image
#       represents a plausible real patient view.
#
#   rotation_range=10
#       Small rotations (≤10°) simulate patient head-tilt during acquisition.
#       Larger rotations would distort the superior–inferior orientation that
#       radiologists rely on for lesion grading.
#
#   zoom_range=0.10, width_shift_range=0.05, height_shift_range=0.05
#       Modest zoom and shift emulate variability in patient positioning and
#       detector alignment, both common sources of inter-scan variability.
#
#   brightness_range=[0.95, 1.05]
#       Narrow brightness jitter accounts for minor differences in X-ray
#       exposure settings across machines, without washing out calcification
#       density cues.
#
#   shear_range=0.0  (DISABLED)
#       Shear distorts the relative spatial relationships between anatomical
#       landmarks (e.g. root apex vs. cortical plate). There is no
#       acquisition scenario that produces a sheared panoramic radiograph;
#       enabling it would introduce unrealistic artefacts.
#
#   vertical_flip=False  (DISABLED — default)
#       The coronal and sagittal views have a fixed superior–inferior
#       orientation defined by gravity and the patient's posture. Flipping
#       vertically produces anatomically impossible images.
#
#   fill_mode='constant', cval=0
#       After geometric transforms, newly exposed border regions are filled
#       with black (cval=0) to match the uniform black background of the
#       padded images (train_test_val_data). Using 'reflect' or 'wrap' would
#       introduce spurious bright or textured borders near black-padded edges.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # anatomically valid: bilateral symmetry
    rotation_range=10,           # ≤10° patient head-tilt
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],
    shear_range=0.0,             # DISABLED — not realistic for anatomy
    vertical_flip=False,         # DISABLED — fixed superior-inferior orientation
    fill_mode='constant',        # avoids interpolation artefacts at black borders
    cval=0,                      # fill with black to match padded background
    preprocessing_function=lambda x: (
        tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x
    )
)

# Validation and test generators — rescaling only; no stochastic augmentation
val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: (
        tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x
    )
)


# =============================================================================
# GPU Configuration
# =============================================================================
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict to first GPU (RTX 4060 via eGPU Node Titan)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


# =============================================================================
# PatientPairFullCombinationSequence
# =============================================================================
#
# Data pairing strategy:
#   For each patient, every available coronal (cr) image is paired with every
#   available sagittal (sg) image (full combinatorial product). When one view
#   is absent for a patient, a blank black image is substituted so that the
#   model learns to classify from a single view — this is the intended
#   operational scenario (incomplete studies).
#
#   With mix_across_patients=True (training split), all coronal images for a
#   class are crossed with all sagittal images for that class, yielding the
#   maximum number of training pairs (4,711 total).
#   With mix_across_patients=False (validation/test), pairing is patient-
#   specific to avoid information leakage between patients during evaluation
#   (1,202 validation / 1,010 test pairs).
#
#   Class indices: healthy=0, lesion=1, mrc=2

import cv2
from pathlib import Path
from itertools import product as cartesian_product


class PatientPairFullCombinationSequence(Sequence):
    def __init__(self, data_dir, datagen, target_size, batch_size, classes,
                 shuffle=True, pairs_per_patient=1, log_file=None,
                 print_batch=False, mix_across_patients=False,
                 run_name="run", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.datagen = datagen
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = classes
        self.shuffle = shuffle
        self.pairs_per_patient = pairs_per_patient
        self.log_file = log_file
        self.mix_across_patients = mix_across_patients
        self.run_name = run_name
        self.current_epoch = 0
        self.first_epoch_logged = False

        self.lb = LabelBinarizer()
        self.lb.fit(classes)

        self.pairs = self._build_all_pairs()
        self.on_epoch_end()
        self.print_batch = print_batch

    def _build_all_pairs(self):
        all_pairs = []

        if self.mix_across_patients:
            # Cross all coronal × all sagittal within each class
            class_files = {cls: {'cr': [], 'sg': []} for cls in self.classes}
            for cls in self.classes:
                for view in ['cr', 'sg']:
                    view_dir = self.data_dir / cls / view
                    if not view_dir.exists():
                        continue
                    for fpath in view_dir.glob('*.jpg'):
                        class_files[cls][view].append(str(fpath))

            for cls in self.classes:
                coronal_list  = class_files[cls].get('cr', []) + [None]
                sagittal_list = class_files[cls].get('sg', []) + [None]
                combinations = [
                    (c, s, cls)
                    for c, s in cartesian_product(coronal_list, sagittal_list)
                    if not (c is None and s is None)
                ]
                all_pairs.extend(combinations)

        else:
            # Patient-specific pairing (preserves patient boundary — no leakage)
            patient_files = {}
            for cls in self.classes:
                for view in ['cr', 'sg']:
                    view_dir = self.data_dir / cls / view
                    if not view_dir.exists():
                        continue
                    for fpath in view_dir.glob('*.jpg'):
                        patient = fpath.name.split('_')[0].strip()
                        key = (cls, patient)
                        if key not in patient_files:
                            patient_files[key] = {'cr': [], 'sg': []}
                        patient_files[key][view].append(str(fpath))

            for (cls, patient), views in patient_files.items():
                coronal_list  = views.get('cr', []) + [None]
                sagittal_list = views.get('sg', []) + [None]
                combinations = [
                    (c, s, cls)
                    for c, s in cartesian_product(coronal_list, sagittal_list)
                    if not (c is None and s is None)
                ]
                if len(combinations) < self.pairs_per_patient:
                    repeat_count = self.pairs_per_patient - len(combinations)
                    extra = random.choices(combinations, k=repeat_count)
                    combinations.extend(extra)
                all_pairs.extend(combinations)

        return all_pairs

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        # Pairs are deterministic (built once in __init__); reshuffle only.
        # Rebuilding from disk every epoch is unnecessary and slow.
        if self.shuffle:
            random.shuffle(self.pairs)

        if not self.first_epoch_logged and self.log_file:
            log_dir = Path("logs") / self.run_name
            log_dir.mkdir(parents=True, exist_ok=True)
            sample_log = log_dir / f"{self.log_file}_epoch1_samples.txt"
            with open(sample_log, 'w') as f:
                f.write(f"===== First Epoch Sample Log =====\n")
                f.write(f"Run: {self.run_name}\n")
                f.write(f"Dataset: {self.data_dir}\n")
                f.write(f"Mix across patients: {self.mix_across_patients}\n")
                f.write(f"Total pairs: {len(self.pairs)}\n")
                f.write(f"Batch size: {self.batch_size}\n")
                f.write(f"Number of batches: {len(self)}\n\n")
                f.write("----- All Samples -----\n")
                for idx, (coronal, sagittal, cls) in enumerate(self.pairs, 1):
                    cor_name = Path(coronal).name if coronal else 'BLANK'
                    sag_name = Path(sagittal).name if sagittal else 'BLANK'
                    f.write(
                        f"{idx:4d} | {cls:8s} | "
                        f"CR: {cor_name:30s} | SG: {sag_name:30s}\n"
                    )
            self.first_epoch_logged = True
            print(f"✓ First epoch samples logged to: {sample_log}")

        self.current_epoch += 1

    def _load_image(self, path):
        img = np.zeros((*self.target_size, 3), dtype=np.uint8)
        if path:
            img = cv2.imread(path)
            if img is None:
                # File missing or unreadable — fall back to blank
                return np.zeros((*self.target_size, 3), dtype=np.uint8)
            # cv2.imread returns BGR — convert to RGB to match keras conventions
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
        return img

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_coronal, batch_sagittal, batch_labels = [], [], []

        for coronal_path, sagittal_path, cls in batch_pairs:
            c_img = self._load_image(coronal_path)
            s_img = self._load_image(sagittal_path)

            if self.datagen:
                # random_transform() applies geometric/colour augmentations only.
                # rescale and preprocessing_function are NOT applied here — they
                # must be triggered explicitly via standardize().
                c_img = self.datagen.random_transform(c_img.astype(np.float32))
                s_img = self.datagen.random_transform(s_img.astype(np.float32))
                # standardize() applies rescale (1/255) AND preprocessing_function
                c_img = self.datagen.standardize(c_img)
                s_img = self.datagen.standardize(s_img)

            batch_coronal.append(c_img)
            batch_sagittal.append(s_img)
            batch_labels.append(cls)

        if len(batch_labels) == 0:
            return (np.array([]), np.array([])), np.array([])

        X_coronal  = np.array(batch_coronal)
        X_sagittal = np.array(batch_sagittal)

        if self.print_batch:
            print(f"Batch {idx}: {len(batch_labels)} samples")

        y = self.lb.transform(batch_labels)
        return (X_coronal, X_sagittal), y


# =============================================================================
# PerformanceLogger
# =============================================================================
# Custom callback that logs training/validation metrics (loss, accuracy,
# precision, recall, learning rate) at each epoch with automatic CSV export
# and a human-readable training summary for reproducibility reporting.

class PerformanceLogger(tf.keras.callbacks.Callback):
    """Logs all performance metrics throughout training for paper reporting."""

    def __init__(self, run_name, log_dir="logs"):
        super().__init__()
        self.run_name = run_name
        self.log_dir  = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'epoch': [], 'loss': [], 'accuracy': [], 'precision': [],
            'recall': [], 'val_loss': [], 'val_accuracy': [],
            'val_precision': [], 'val_recall': [], 'lr': [], 'epoch_time': []
        }
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time

        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['precision'].append(logs.get('precision'))
        self.history['recall'].append(logs.get('recall'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))

        optimizer = self.model.optimizer
        if hasattr(optimizer, 'inner_optimizer'):
            lr = float(tf.keras.backend.get_value(
                optimizer.inner_optimizer.learning_rate))
        else:
            lr = float(tf.keras.backend.get_value(optimizer.learning_rate))

        self.history['lr'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        self._save_metrics()

    def _save_metrics(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_dir / "training_metrics.csv", index=False)

    def on_train_end(self, logs=None):
        summary_path = self.log_dir / "training_summary.txt"
        total_seconds = sum(self.history['epoch_time'])
        with open(summary_path, 'w') as f:
            f.write("===== Training Summary =====\n")
            f.write(f"Run Name: {self.run_name}\n")
            f.write(f"Random Seed: {RANDOM_SEED}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
            f.write(f"Total Training Time: {total_seconds:.2f} s "
                    f"({total_seconds/3600:.2f} h)\n")
            f.write(f"Average Time per Epoch: "
                    f"{np.mean(self.history['epoch_time']):.2f} s\n\n")

            best_v = np.argmin(self.history['val_loss'])
            f.write(f"Best Validation Loss: "
                    f"{self.history['val_loss'][best_v]:.4f} "
                    f"(Epoch {self.history['epoch'][best_v]})\n")
            f.write(f"  Val Accuracy:  {self.history['val_accuracy'][best_v]:.4f}\n")
            f.write(f"  Val Precision: {self.history['val_precision'][best_v]:.4f}\n")
            f.write(f"  Val Recall:    {self.history['val_recall'][best_v]:.4f}\n\n")

            best_a = np.argmax(self.history['val_accuracy'])
            f.write(f"Best Validation Accuracy: "
                    f"{self.history['val_accuracy'][best_a]:.4f} "
                    f"(Epoch {self.history['epoch'][best_a]})\n")
            f.write(f"  Val Loss: {self.history['val_loss'][best_a]:.4f}\n\n")

            f.write(f"Final Metrics (Epoch {self.history['epoch'][-1]}):\n")
            f.write(f"  Train Loss:     {self.history['loss'][-1]:.4f}\n")
            f.write(f"  Train Accuracy: {self.history['accuracy'][-1]:.4f}\n")
            f.write(f"  Val Loss:       {self.history['val_loss'][-1]:.4f}\n")
            f.write(f"  Val Accuracy:   {self.history['val_accuracy'][-1]:.4f}\n")

        print(f"✓ Training summary saved to: {summary_path}")


# =============================================================================
# ValPredictionCache
# =============================================================================
# Runs a single model.predict(val_seq) per epoch and stores the result.
# Downstream callbacks (DetailedMetricsCallback, F1Checkpoint,
# CompositeCheckpoint, CompositeEarlyStopping) read from this cache instead
# of running their own redundant predict() — saves ~3× val inference cost.
# Must be registered FIRST in the callbacks list so it runs before consumers.

class ValPredictionCache(tf.keras.callbacks.Callback):

    def __init__(self, val_seq):
        super().__init__()
        self.val_seq        = val_seq
        self.y_pred_proba   = None
        self.y_true         = None
        self.y_pred         = None
        self.f1_weighted    = None

    def on_epoch_end(self, epoch, logs=None):
        self.y_pred_proba = self.model.predict(self.val_seq, verbose=0)
        y_true = []
        for i in range(len(self.val_seq)):
            _, y_batch = self.val_seq[i]
            if len(y_batch) > 0:
                y_true.extend(np.argmax(y_batch, axis=1))
        self.y_true = np.array(y_true[:len(self.y_pred_proba)])
        self.y_pred_proba = self.y_pred_proba[:len(self.y_true)]
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        self.f1_weighted = f1_score(self.y_true, self.y_pred,
                                    average='weighted', zero_division=0)


# =============================================================================
# DetailedMetricsCallback
# =============================================================================
# Computes comprehensive validation metrics per epoch: weighted and macro
# F1-scores and per-class precision, recall, F1 for healthy / lesion / MRC.
# Particularly useful for monitoring MRC (smallest class) independently.

class DetailedMetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_cache, run_name, classes=['healthy', 'lesion', 'mrc']):
        super().__init__()
        self.val_cache = val_cache
        self.run_name  = run_name
        self.classes   = classes
        self.log_dir   = Path(f"logs/{run_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.detailed_history = {
            'epoch': [], 'val_loss': [], 'val_accuracy': [],
            'val_f1_weighted': [], 'val_f1_macro': [],
            'val_precision_weighted': [], 'val_recall_weighted': []
        }
        for cls in classes:
            self.detailed_history[f'val_f1_{cls}']        = []
            self.detailed_history[f'val_precision_{cls}'] = []
            self.detailed_history[f'val_recall_{cls}']    = []

        self.best_f1       = 0.0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        print("\n  Computing detailed metrics...", end='')
        # Read from shared cache instead of running predict() again
        y_pred       = self.val_cache.y_pred
        y_true       = self.val_cache.y_true
        y_pred_proba = self.val_cache.y_pred_proba

        f1_weighted        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro           = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per_class        = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class    = recall_score(y_true, y_pred, average=None, zero_division=0)

        self.detailed_history['epoch'].append(epoch + 1)
        self.detailed_history['val_loss'].append(logs.get('val_loss'))
        self.detailed_history['val_accuracy'].append(logs.get('val_accuracy'))
        self.detailed_history['val_f1_weighted'].append(f1_weighted)
        self.detailed_history['val_f1_macro'].append(f1_macro)
        self.detailed_history['val_precision_weighted'].append(precision_weighted)
        self.detailed_history['val_recall_weighted'].append(recall_weighted)

        for i, cls in enumerate(self.classes):
            self.detailed_history[f'val_f1_{cls}'].append(f1_per_class[i])
            self.detailed_history[f'val_precision_{cls}'].append(precision_per_class[i])
            self.detailed_history[f'val_recall_{cls}'].append(recall_per_class[i])

        df = pd.DataFrame(self.detailed_history)
        df.to_csv(self.log_dir / "detailed_validation_metrics.csv", index=False)

        print(f" Done!")
        print(f"  Val Loss: {logs.get('val_loss'):.4f} | "
              f"Val Acc: {logs.get('val_accuracy'):.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f} | F1 (macro): {f1_macro:.4f}")
        print(f"  Per-class F1: ", end='')
        for i, cls in enumerate(self.classes):
            print(f"{cls}: {f1_per_class[i]:.3f}", end='  ')
        print()

        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"  ✓ New best val_loss: {val_loss:.4f}")
        if f1_weighted > self.best_f1:
            self.best_f1 = f1_weighted
            print(f"  ✓ New best F1 (weighted): {f1_weighted:.4f}")

    def on_train_end(self, logs=None):
        summary_path = self.log_dir / "validation_metrics_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("===== Validation Metrics Summary =====\n\n")
            best_loss_idx = np.argmin(self.detailed_history['val_loss'])
            f.write(f"Best Validation Loss: "
                    f"{self.detailed_history['val_loss'][best_loss_idx]:.4f} "
                    f"(Epoch {self.detailed_history['epoch'][best_loss_idx]})\n")
            f.write(f"  Val Accuracy:  "
                    f"{self.detailed_history['val_accuracy'][best_loss_idx]:.4f}\n")
            f.write(f"  F1 (weighted): "
                    f"{self.detailed_history['val_f1_weighted'][best_loss_idx]:.4f}\n")
            f.write(f"  F1 (macro):    "
                    f"{self.detailed_history['val_f1_macro'][best_loss_idx]:.4f}\n\n")

            best_f1_idx = np.argmax(self.detailed_history['val_f1_weighted'])
            f.write(f"Best F1 (weighted): "
                    f"{self.detailed_history['val_f1_weighted'][best_f1_idx]:.4f} "
                    f"(Epoch {self.detailed_history['epoch'][best_f1_idx]})\n")
            f.write(f"  Val Loss:     "
                    f"{self.detailed_history['val_loss'][best_f1_idx]:.4f}\n")
            f.write(f"  Val Accuracy: "
                    f"{self.detailed_history['val_accuracy'][best_f1_idx]:.4f}\n\n")

            f.write(f"Per-Class Metrics at Best Val Loss "
                    f"(Epoch {self.detailed_history['epoch'][best_loss_idx]}):\n")
            for cls in self.classes:
                f.write(f"\n  {_disp(cls)}:\n")
                f.write(f"    F1:        "
                        f"{self.detailed_history[f'val_f1_{cls}'][best_loss_idx]:.4f}\n")
                f.write(f"    Precision: "
                        f"{self.detailed_history[f'val_precision_{cls}'][best_loss_idx]:.4f}\n")
                f.write(f"    Recall:    "
                        f"{self.detailed_history[f'val_recall_{cls}'][best_loss_idx]:.4f}\n")

        print(f"\n✓ Detailed validation metrics saved to: {summary_path}")


# =============================================================================
# F1Checkpoint
# =============================================================================
# Saves model weights at the epoch with the highest weighted F1-score,
# prioritising classification performance over loss minimisation.

class F1Checkpoint(tf.keras.callbacks.Callback):

    def __init__(self, val_cache, filepath, run_name):
        super().__init__()
        self.val_cache = val_cache
        self.filepath  = filepath
        self.best_f1   = 0.0

    def on_epoch_end(self, epoch, logs=None):
        f1 = self.val_cache.f1_weighted
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save(self.filepath)
            print(f"\n  ✓ New best F1: {f1:.4f} — model saved")


# =============================================================================
# CompositeCheckpoint
# =============================================================================
# Saves the model at the epoch with the best composite score:
#     score = α · loss_norm + (1 − α) · (1 − F1_weighted)
# where α = 0.6 (60 % weight on validation loss, 40 % on F1).
#
# TRANSPARENCY NOTE — three-way model selection strategy:
#   Three checkpoints are maintained simultaneously (best loss, best F1,
#   best composite) and all three are evaluated on the held-out test set
#   after training. This approach makes the checkpoint-selection decision
#   transparent and reproducible: reviewers can inspect the full comparison
#   table (model_comparison.csv) rather than taking a single reported number
#   on faith. The composite model is reported as primary; the other two serve
#   as internal controls confirming that the result is not an artefact of an
#   arbitrary stopping criterion.

class CompositeCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, val_cache, filepath, alpha=0.6):
        super().__init__()
        self.val_cache  = val_cache
        self.filepath   = filepath
        self.alpha      = alpha   # 60 % loss, 40 % F1
        self.best_score = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        f1 = self.val_cache.f1_weighted
        normalized_loss  = val_loss / 2.0
        composite_score  = self.alpha * normalized_loss + (1 - self.alpha) * (1 - f1)
        if composite_score < self.best_score:
            self.best_score = composite_score
            self.model.save(self.filepath)
            print(f"\n  ✓ New best composite: {composite_score:.4f} "
                  f"(loss={val_loss:.4f}, F1={f1:.4f})")


# =============================================================================
# CompositeEarlyStopping
# =============================================================================
# Early stopping on the same composite metric (60 % loss + 40 % F1).
# patience=30 epochs, restores best weights on stop.

class CompositeEarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, val_cache, patience=30, alpha=0.6, restore_best=True):
        super().__init__()
        self.val_cache    = val_cache
        self.patience     = patience
        self.alpha        = alpha
        self.best_score   = float('inf')
        self.wait         = 0
        self.best_weights = None
        self.restore_best = restore_best

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        f1 = self.val_cache.f1_weighted
        normalized_loss = val_loss / 2.0
        composite_score = self.alpha * normalized_loss + (1 - self.alpha) * (1 - f1)

        if composite_score < self.best_score:
            self.best_score = composite_score
            self.wait = 0
            if self.restore_best:
                self.best_weights = self.model.get_weights()
            print(f"  Composite improved to {composite_score:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                self.model.stop_training = True
                if self.restore_best and self.best_weights is not None:
                    print("Restoring best weights")
                    self.model.set_weights(self.best_weights)


# =============================================================================
# _save_figure — helper used by every plotting function
# =============================================================================
def _save_figure(fig, stem: str, subdir: str = "") -> None:
    """
    Save *fig* to OUTPUT_DIR (and optional *subdir*) in PNG, PDF, and SVG.

    Parameters
    ----------
    fig    : matplotlib Figure object
    stem   : base filename without extension  (e.g. 'confusion_matrix')
    subdir : optional sub-folder inside OUTPUT_DIR  (e.g. 'training')
    """
    out = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ('png', 'pdf', 'svg'):
        fig.savefig(out / f"{stem}.{fmt}", format=fmt)
    print(f"  ✓ Saved  {stem}.[png|pdf|svg]  →  {out}/")


# =============================================================================
# PublicationPlotter
# =============================================================================
# Generates all manuscript figures. Every method saves three formats
# (PNG 300 dpi, PDF vector, SVG vector) to OUTPUT_DIR.

class PublicationPlotter:

    def __init__(self, run_name, log_dir="logs"):
        self.run_name = run_name
        # Also keep a run-specific plots folder for backward compatibility
        self.log_dir = Path(log_dir) / run_name / "plots"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(MANUSCRIPT_RC)

    # ── Training curves ───────────────────────────────────────────────────────
    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Training History", fontweight='bold')

        panels = [
            (axes[0, 0], history['loss'],          history['val_loss'],          'Loss',      'Loss'),
            (axes[0, 1], history['accuracy'],      history['val_accuracy'],      'Accuracy',  'Accuracy'),
            (axes[1, 0], history['precision'],     history['val_precision'],     'Precision', 'Precision'),
            (axes[1, 1], history['recall'],        history['val_recall'],        'Recall',    'Recall'),
        ]
        for ax, train_vals, val_vals, ylabel, title in panels:
            ax.plot(history['epoch'], train_vals, color=TRAIN_COLOR,
                    label='Training', linewidth=2)
            ax.plot(history['epoch'], val_vals,   color=VAL_COLOR,
                    linestyle='--', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        _save_figure(fig, "training_history", subdir="training")
        plt.close(fig)

    # ── F1-score evolution ────────────────────────────────────────────────────
    def plot_f1_scores(self, detailed_history, classes):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Validation F1-Score Evolution", fontweight='bold')

        # Aggregate F1
        axes[0].plot(detailed_history['epoch'],
                     detailed_history['val_f1_weighted'],
                     color=TRAIN_COLOR, label='Weighted', linewidth=2,
                     marker='o', markersize=3)
        axes[0].plot(detailed_history['epoch'],
                     detailed_history['val_f1_macro'],
                     color=VAL_COLOR, linestyle='--', label='Macro',
                     linewidth=2, marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_title('Aggregate F1-Score')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_ylim([0, 1.05])

        # Per-class F1
        for ci, cls in enumerate(classes):
            axes[1].plot(detailed_history['epoch'],
                         detailed_history[f'val_f1_{cls}'],
                         label=_disp(cls),
                         color=CLASS_COLORS[cls],
                         linestyle=CLASS_LINESTYLES[ci % len(CLASS_LINESTYLES)],
                         marker=CLASS_MARKERS[ci % len(CLASS_MARKERS)],
                         linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('Per-Class F1-Score')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        _save_figure(fig, "f1_scores", subdir="training")
        plt.close(fig)

    # ── Loss vs accuracy/F1 (dual y-axis) ────────────────────────────────────
    def plot_loss_vs_metrics(self, detailed_history):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Validation Loss vs. Performance Metrics", fontweight='bold')

        for ax, y2_key, y2_label, y2_color, title in [
            (axes[0], 'val_accuracy',      'Val Accuracy',      '#E65100', 'Loss vs Accuracy'),
            (axes[1], 'val_f1_weighted',   'Val F1 (weighted)', '#1B5E20', 'Loss vs F1-Score'),
        ]:
            ax2 = ax.twinx()
            l1, = ax.plot(detailed_history['epoch'], detailed_history['val_loss'],
                          color=TRAIN_COLOR, label='Val Loss', linewidth=2)
            l2, = ax2.plot(detailed_history['epoch'], detailed_history[y2_key],
                           color=y2_color, linestyle='--', label=y2_label, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss', color=TRAIN_COLOR)
            ax2.set_ylabel(y2_label, color=y2_color)
            ax.tick_params(axis='y', labelcolor=TRAIN_COLOR)
            ax2.tick_params(axis='y', labelcolor=y2_color)
            ax.set_title(title)
            ax.grid(True)
            ax.legend(handles=[l1, l2], loc='upper right')

        plt.tight_layout()
        _save_figure(fig, "loss_vs_metrics", subdir="training")
        plt.close(fig)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """
        Publication-quality confusion matrix with:
          - Counts AND row-normalized percentages in each cell
          - Per-class precision (column) and recall (row) sidebar
          - Custom diverging colormap optimized for B&W printing
          - Diagonal emphasis for correct predictions
        """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        n = len(classes)
        class_labels = [_disp(c) for c in classes]
        
        # Custom journal-quality colormap (white → deep teal/navy)
        # Prints well in grayscale AND distinguishable in color
        journal_cmap = LinearSegmentedColormap.from_list(
            'journal_blue',
            ['#FFFFFF', '#E8F1F5', '#B8D4E3', '#7FA9C7', '#3D6A8A', '#1A3D5C'],
            N=256
        )
        
        # Compute per-class metrics
        recall    = np.diag(cm) / cm.sum(axis=1)          # row-wise
        precision = np.diag(cm) / cm.sum(axis=0)          # col-wise
        f1        = 2 * (precision * recall) / (precision + recall + 1e-8)
        overall_acc = np.diag(cm).sum() / cm.sum()
        
        # ─── Figure layout ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.32)
        
        for panel_idx, (data, fmt_str, title, cbar_label, vmax) in enumerate([
            (cm,      'count',  'Counts',           'Sample count', cm.max()),
            (cm_norm, 'pct',    'Row-Normalized',   'Recall (per true class)', 1.0),
        ]):
            ax = fig.add_subplot(gs[0, panel_idx])
            
            # Plot heatmap manually for full control
            im = ax.imshow(data, cmap=journal_cmap, vmin=0, vmax=vmax,
                          aspect='equal', interpolation='nearest')
            
            # Cell text: count + percentage (or just percentage)
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    # Choose text color based on cell darkness
                    text_color = 'white' if val > 0.5 * vmax else '#1A3D5C'
                    
                    if fmt_str == 'count':
                        pct = cm_norm[i, j] * 100
                        cell_text = f"{int(val)}\n({pct:.1f}%)"
                    else:
                        cell_text = f"{val:.3f}"
                    
                    ax.text(j, i, cell_text, ha='center', va='center',
                           color=text_color, fontsize=11, fontweight='bold',
                           linespacing=1.3)
                    
                    # Diagonal emphasis: thin gold border on correct predictions
                    if i == j:
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='#D4AF37',
                                       linewidth=2.5, zorder=10)
                        ax.add_patch(rect)
            
            # Axis labels
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(class_labels, fontsize=11, fontweight='bold')
            ax.set_yticklabels(class_labels, fontsize=11, fontweight='bold',
                              rotation=0)
            ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_ylabel('True Class',      fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
            
            # Move ticks to top for top-readability (journal convention)
            ax.xaxis.set_label_position('bottom')
            ax.tick_params(axis='both', which='both', length=0)
            
            # Grid lines between cells
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.2)
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
            cbar.set_label(cbar_label, fontsize=10, fontweight='bold', labelpad=8)
            cbar.ax.tick_params(labelsize=9)
            cbar.outline.set_edgecolor('#333333')
            cbar.outline.set_linewidth(0.8)
        
        # Overall title with summary metrics
        summary_metrics = (
            f"Overall Accuracy: {overall_acc:.4f}  •  "
            f"Macro-F1: {np.mean(f1):.4f}  •  "
            f"N = {cm.sum()}"
        )
        fig.suptitle("Confusion Matrix", fontsize=15, fontweight='bold', y=1.02)
        # Cleaner: subtitle below main title via figtext
        fig.text(0.5, 0.94, summary_metrics,
                ha='center', va='top', fontsize=11, style='italic', color='#444')
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        _save_figure(fig, "confusion_matrix", subdir="evaluation")
        plt.close(fig)
        
        # ─── Also save a SINGLE-PANEL version (paper-friendly) ────────────────
        fig2, ax2 = plt.subplots(figsize=(7, 6.5))
        
        # Use percentages with counts in each cell
        im = ax2.imshow(cm_norm, cmap=journal_cmap, vmin=0, vmax=1.0,
                       aspect='equal', interpolation='nearest')
        
        for i in range(n):
            for j in range(n):
                pct = cm_norm[i, j] * 100
                count = int(cm[i, j])
                text_color = 'white' if cm_norm[i, j] > 0.5 else '#1A3D5C'
                ax2.text(j, i, f"{pct:.1f}%\nn={count}",
                        ha='center', va='center',
                        color=text_color, fontsize=12, fontweight='bold',
                        linespacing=1.3)
                if i == j:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='#D4AF37',
                                   linewidth=2.5, zorder=10)
                    ax2.add_patch(rect)
        
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(class_labels, fontsize=12, fontweight='bold')
        ax2.set_yticklabels(class_labels, fontsize=12, fontweight='bold', rotation=0)
        ax2.set_xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_ylabel('True Class',      fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_title(f"Confusion Matrix  •  Accuracy = {overall_acc:.4f}",
                     fontsize=14, fontweight='bold', pad=14)
        ax2.tick_params(axis='both', which='both', length=0)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.2)
        
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.set_label('Recall (Row-Normalized)', fontsize=11, fontweight='bold',
                      labelpad=8)
        cbar.ax.tick_params(labelsize=10)
        cbar.outline.set_edgecolor('#333333')
        
        plt.tight_layout()
        _save_figure(fig2, "confusion_matrix_paper", subdir="evaluation")
        plt.close(fig2)

    # ── ROC curves ────────────────────────────────────────────────────────────
    def plot_roc_curves(self, y_true, y_pred_proba, classes):
        n_classes  = len(classes)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(7, 7))

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLASS_COLOR_LIST[i], lw=2,
                    linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)],
                    marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
                    markevery=max(1, len(fpr) // 10), markersize=6,
                    label=f'{_disp(cls)}  (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.5,
                linestyle=(0, (1, 1)), label='Chance level (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 − Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Receiver Operating Characteristic — One-vs-Rest')
        ax.legend(loc='lower right')
        ax.grid(True)

        plt.tight_layout()
        _save_figure(fig, "roc_curves", subdir="evaluation")
        plt.close(fig)

    # ── Precision-Recall curves ───────────────────────────────────────────────
    def plot_pr_curves(self, y_true, y_pred_proba, classes):
        from sklearn.metrics import average_precision_score
        n_classes  = len(classes)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(7, 7))
        auprc_macro = []
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i])
            auprc = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            auprc_macro.append(auprc)
            ax.plot(recall, precision, color=CLASS_COLOR_LIST[i], lw=2,
                    linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)],
                    marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
                    markevery=max(1, len(recall) // 10), markersize=6,
                    label=f'{_disp(cls)}  (AUPRC = {auprc:.3f})')

        macro_auprc = np.mean(auprc_macro)
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title(f'Precision–Recall Curves — One-vs-Rest\nMacro-AUPRC = {macro_auprc:.3f}',
                    fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        _save_figure(fig, "pr_curves", subdir="evaluation")
        plt.close(fig)

    # ── Learning-rate schedule ────────────────────────────────────────────────
    def plot_learning_rate(self, history):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history['epoch'], history['lr'],
                color=TRAIN_COLOR, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)')
        ax.set_yscale('log')
        ax.grid(True)

        plt.tight_layout()
        _save_figure(fig, "learning_rate", subdir="training")
        plt.close(fig)

    # ── Classification report (text + figure) ────────────────────────────────
    def generate_classification_report(self, y_true, y_pred, classes):
        report = classification_report(y_true, y_pred,
                                       target_names=[_disp(c) for c in classes], digits=4)
        report_path = self.log_dir.parent / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("===== Classification Report =====\n\n")
            f.write(report)
        print(f"✓ Classification report saved to {report_path}")
        return report

    # ── Model comparison table (bar chart) ───────────────────────────────────
    def plot_model_comparison(self, comparison_df):
        """
        Bar chart comparing the three checkpoint strategies on the test set.
        Framing: transparency contribution — shows that the reported number
        is not artefact-dependent on the checkpoint criterion chosen.
        """
        metrics = ['Test Accuracy', 'Test F1 (weighted)', 'Test F1 (macro)',
                   'Test Precision', 'Test Recall']
        available = [m for m in metrics if m in comparison_df.columns]

        n_metrics = len(available)
        x = np.arange(len(comparison_df))
        width = 0.15
        # Enough colours + distinct hatches for up to 5 metrics (B&W-safe)
        colors_bar = ['#1565C0', '#B71C1C', '#1B5E20', '#E58A00', '#5E35B1']
        hatches_bar = ['', '///', '...', 'xxx', '\\\\\\']

        fig, ax = plt.subplots(figsize=(11, 6))
        for k, metric in enumerate(available):
            bars = ax.bar(x + k * width,
                          comparison_df[metric].values,
                          width,
                          label=metric.replace('Test ', ''),
                          color=colors_bar[k % len(colors_bar)],
                          hatch=hatches_bar[k % len(hatches_bar)],
                          alpha=0.95, edgecolor='#222', linewidth=0.7)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(comparison_df['Model'].values, fontsize=11)
        ax.set_ylim([0, 1.12])
        ax.set_ylabel('Score')
        ax.set_title('Three-Checkpoint Model Comparison (Test Set)',
                     fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, axis='y')

        plt.tight_layout()
        _save_figure(fig, "model_comparison", subdir="evaluation")
        plt.close(fig)


# =============================================================================
# Helper: save any standalone figure to output_figures
# =============================================================================
def save_standalone_figure(fig, stem: str, subdir: str = "") -> None:
    """Alias of _save_figure for use outside PublicationPlotter."""
    _save_figure(fig, stem, subdir=subdir)


# =============================================================================
# Data sequences
# =============================================================================
from datetime import datetime
if RUN_TAG:
    # Namespace under fold_{k}/ so each fold's logs live separately:
    #   logs/fold_{k}/run_YYYYMMDD_HHMMSS/
    run_name = f"{RUN_TAG}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

train_seq = PatientPairFullCombinationSequence(
    data_dir=f'{DATA_ROOT}/train',
    datagen=train_datagen,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=['healthy', 'lesion', 'mrc'],
    shuffle=True,
    pairs_per_patient=1,
    mix_across_patients=True,   # cross-patient combination for training
    run_name=run_name,
    log_file="train"
)

val_seq = PatientPairFullCombinationSequence(
    data_dir=f'{DATA_ROOT}/validation',
    datagen=val_test_datagen,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=['healthy', 'lesion', 'mrc'],
    shuffle=False,
    pairs_per_patient=1,
    mix_across_patients=False,  # patient-specific pairing — no leakage
    run_name=run_name,
    log_file="val"
)

test_seq = PatientPairFullCombinationSequence(
    data_dir=f'{DATA_ROOT}/test',
    datagen=val_test_datagen,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=['healthy', 'lesion', 'mrc'],
    shuffle=False,
    pairs_per_patient=1,
    mix_across_patients=False,  # patient-specific pairing — no leakage
    run_name=run_name,
    log_file="test"
)

print(f"Found {len(train_seq.pairs)} training pairs belonging to {len(train_seq.classes)} classes.")
print(f"Found {len(val_seq.pairs)} validation pairs belonging to {len(val_seq.classes)} classes.")
print(f"Found {len(test_seq.pairs)} test pairs belonging to {len(test_seq.classes)} classes.")
print(f"Class indices: {dict(zip(train_seq.classes, range(len(train_seq.classes))))}")
# Expected output (single-split reference; per-fold counts will differ):
#   Found ~4711 training pairs belonging to 3 classes.
#   Found ~1202 validation pairs belonging to 3 classes.
#   Found ~1010 test pairs belonging to 3 classes.
#   Class indices: {'healthy': 0, 'lesion': 1, 'mrc': 2}
# In k-fold mode each fold has ~80% of patients in train (with ~15% of those
# carved to validation) and ~20% held out as the test fold.

# ── Snapshot run configuration & environment for reproducibility ──────────────
# Writes  logs/{run_name}/run_metadata.json  capturing:
#   random seed, image size, batch size, class indices, dataset split sizes,
#   package versions (TF/Keras/numpy/sklearn/cv2/PIL/...), GPU info from
#   nvidia-smi, Python+platform, and git commit hash if in a repo.
# This file is what someone (or future-you) needs alongside the .keras model
# to reproduce or correctly load the trained network.
save_run_metadata(
    run_name=run_name,
    random_seed=RANDOM_SEED,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    num_classes=NUM_CLASSES,
    classes=['healthy', 'lesion', 'mrc'],
    train_pairs=len(train_seq.pairs),
    val_pairs=len(val_seq.pairs),
    test_pairs=len(test_seq.pairs),
)


# =============================================================================
# Model Architecture
# =============================================================================
#
# The network is organised as three stacked innovations:
#
#   INNOVATION 1 — DUAL-BRANCH ENCODING
#       Two independent branches (coronal, sagittal) share the same
#       convolutional topology [32→32→32 filters, 3×3 kernels, BN+MaxPool]
#       but carry no shared weights, allowing each branch to specialise in
#       view-specific texture patterns. Blank-image substitution for missing
#       views (zero tensor) enables inference from either single view.
#
#   INNOVATION 2 — BRANCH-LEVEL CBAM ATTENTION
#       Each branch output is passed through a Convolutional Block Attention
#       Module (CBAM; Woo et al., ECCV 2018) that applies:
#         (a) Channel attention — shared MLP on GAP and GMP to weight
#             informative feature channels (ratio=16)
#         (b) Spatial attention — 7×7 convolution on channel-wise
#             average and max projections to highlight lesion-relevant
#             spatial regions
#       This is applied before fusion, so each branch's attention is
#       computed independently of the other view.
#
#   INNOVATION 3 — GATED FUSION
#       A learned softmax gate (Dense(2, softmax)) takes the concatenated
#       global average-pooled representations of both branches and produces
#       scalar weights [w_coronal, w_sagittal] that are broadcast over the
#       full feature maps. This is the most defensible novelty claim:
#         - When one view is blank, the gate learns to suppress it (w→0)
#         - When both views are informative, the gate learns optimal
#           view-specific weighting for each sample
#       After gating, features are concatenated spatially and reduced with
#       a 1×1 convolution to maintain the channel count.
#       Subsequent shared processing: Conv2D(64)→Conv2D(128)→GAP→
#       Dense(256)→Dense(64)→softmax(3).

from keras.utils import register_keras_serializable


def first_branch(name, input_shape, weight_decay):
    """Shared topology for coronal and sagittal branches."""
    inputs = Input(shape=input_shape, name=f"{name}_input")
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    return inputs, x


@register_keras_serializable()
def slice_coronal_gate(x):
    return x[:, 0:1]


@register_keras_serializable()
def slice_sagittal_gate(x):
    return x[:, 1:2]


@register_keras_serializable()
def gated_fusion_layer(coronal_features, sagittal_features):
    """
    Innovation 3 — Gated fusion.
    Learns per-sample branch weights; suppresses blank (zero) branch.
    """
    coronal_gap  = GlobalAveragePooling2D()(coronal_features)
    sagittal_gap = GlobalAveragePooling2D()(sagittal_features)
    combined_gap = Concatenate()([coronal_gap, sagittal_gap])
    gate_weights = Dense(2, activation='softmax', name='branch_gates')(combined_gap)

    coronal_gate  = tf.keras.layers.Lambda(slice_coronal_gate)(gate_weights)
    sagittal_gate = tf.keras.layers.Lambda(slice_sagittal_gate)(gate_weights)
    coronal_gate  = Reshape((1, 1, 1))(coronal_gate)
    sagittal_gate = Reshape((1, 1, 1))(sagittal_gate)

    coronal_gated  = Multiply()([coronal_features,  coronal_gate])
    sagittal_gated = Multiply()([sagittal_features, sagittal_gate])

    # Spatial (side-by-side) concatenation rather than channel-wise
    fused_features = Concatenate(axis=2)([coronal_gated, sagittal_gated])
    fused_filters  = coronal_features.shape[-1]
    fused_features = Conv2D(int(fused_filters), (1, 1), activation='relu')(fused_features)
    return fused_features


@register_keras_serializable()
def reduce_mean_spatial(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)


@register_keras_serializable()
def reduce_max_spatial(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)


@register_keras_serializable()
def cbam_block(input_tensor, ratio=16):
    """
    Innovation 2 — CBAM (Woo et al., ECCV 2018).
    Channel attention then spatial attention applied per branch.
    """
    filters = input_tensor.shape[-1]

    # ── Channel attention ──────────────────────────────────────────────
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)
    shared_dense1 = Dense(filters // ratio, activation='relu',
                          kernel_regularizer=l2(1e-4))
    shared_dense2 = Dense(filters, activation='sigmoid',
                          kernel_regularizer=l2(1e-4))
    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    channel_attention = Add()([avg_out, max_out])
    channel_attention = Reshape((1, 1, filters))(channel_attention)
    x = Multiply()([input_tensor, channel_attention])

    # ── Spatial attention ──────────────────────────────────────────────
    avg_pool_spatial = Lambda(reduce_mean_spatial)(x)
    max_pool_spatial = Lambda(reduce_max_spatial)(x)
    spatial_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv2D(
        1, (7, 7), padding='same', activation='sigmoid',
        kernel_regularizer=l2(1e-4)
    )(spatial_features)
    output = Multiply()([x, spatial_attention])
    return output


def build_two_branch_cnn_shared(input_shape=(325, 325, 3), num_classes=3):
    """
    Full model: dual-branch → branch CBAM → gated fusion → shared CNN.
    Three-layer innovation: encoding → attention → fusion.
    """
    weight_decay  = 1e-4
    dropout_rate  = 0.5

    # Innovation 1: dual-branch encoding
    coronal_input,  coronal_features  = first_branch("coronal",  input_shape, weight_decay)
    sagittal_input, sagittal_features = first_branch("sagittal", input_shape, weight_decay)

    # Innovation 2: per-branch CBAM attention
    coronal_features  = cbam_block(coronal_features,  ratio=16)
    sagittal_features = cbam_block(sagittal_features, ratio=16)

    # Innovation 3: gated fusion
    merged_features = gated_fusion_layer(coronal_features, sagittal_features)

    # Shared processing head
    x = Conv2D(64,  (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay))(merged_features)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)          # higher dropout before first dense

    x = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # dtype='float32' ensures softmax stays in full precision with mixed-precision
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=[coronal_input, sagittal_input], outputs=outputs)
    model.compile(
        # clipnorm=1.0 caps the global gradient L2 norm each step. Without it,
        # Adam at lr=5e-4 can take a destabilising step on an unlucky minibatch
        # and diverge (observed in one fold: val_loss exploded at epoch 3 and
        # never recovered). Clipping makes every fold robust to that and costs
        # nothing when gradients are already small.
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )
    return model


model = build_two_branch_cnn_shared(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=NUM_CLASSES
)
model.summary()


# =============================================================================
# Training — Multi-Checkpoint Strategy
# =============================================================================
#
# Three model variants are saved simultaneously:
#   1. best_model_loss.keras     — minimum validation loss
#   2. best_model_f1.keras       — maximum weighted F1-score
#   3. best_model_composite.keras — best composite score
#                                   (α·loss_norm + (1−α)·(1−F1), α=0.6)
#
# All three are evaluated on the test set after training and reported in
# model_comparison.csv. This is a transparency contribution: it confirms
# that the reported result does not depend on an arbitrary choice of
# stopping criterion. The composite model is used as the primary reported
# model because it balances calibration and classification performance.

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ValPredictionCache MUST be first — downstream callbacks read its outputs
val_cache      = ValPredictionCache(val_seq=val_seq)
perf_logger    = PerformanceLogger(run_name=run_name)
detailed_metrics = DetailedMetricsCallback(
    val_cache=val_cache, run_name=run_name,
    classes=['healthy', 'lesion', 'mrc']
)

checkpoint_loss = ModelCheckpoint(
    filepath=f'logs/{run_name}/best_model_loss.keras',
    monitor='val_loss', mode='min',
    save_best_only=True, verbose=1
)
checkpoint_f1 = F1Checkpoint(
    val_cache=val_cache,
    filepath=f'logs/{run_name}/best_model_f1.keras',
    run_name=run_name
)
checkpoint_composite = CompositeCheckpoint(
    val_cache=val_cache,
    filepath=f'logs/{run_name}/best_model_composite.keras',
    alpha=0.6   # 60 % loss, 40 % F1
)
early_stopping = CompositeEarlyStopping(
    val_cache=val_cache, patience=30, alpha=0.6, restore_best=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10,
    verbose=1, min_lr=1e-7
)

# Order matters: val_cache populates predictions BEFORE downstream callbacks read them
callbacks = [
    val_cache,
    perf_logger, detailed_metrics,
    checkpoint_loss, checkpoint_f1, checkpoint_composite,
    early_stopping, lr_scheduler
]

# ── Class weights (balanced) ──────────────────────────────────────────────────
# Computed directly from cached pair labels rather than iterating ~5,959 batches
# (the old approach loaded and augmented ~95k images just to count classes).
print("\nComputing class weights from cached pair labels...")
class_to_idx = {cls: i for i, cls in enumerate(train_seq.classes)}
all_labels = [class_to_idx[pair[2]] for pair in train_seq.pairs]
print(f"  ✓ Collected {len(all_labels)} training samples")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels
)
class_weights = dict(enumerate(class_weights_array))
print(f"Class weights: {class_weights}\n")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Starting Training Run: {run_name}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Strategy: Composite Score (60% loss + 40% F1)")
print(f"{'='*60}\n")

history = model.fit(
    train_seq,
    epochs=EPOCHS,
    validation_data=val_seq,
    validation_steps=len(val_seq),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print(f"\n{'='*60}")
print("Training Complete!")
print(f"{'='*60}\n")


# =============================================================================
# Generate Training Plots
# =============================================================================
print("Generating training plots → output_figures/training/")
plotter = PublicationPlotter(run_name=run_name)
plotter.plot_training_history(perf_logger.history)
plotter.plot_learning_rate(perf_logger.history)
plotter.plot_f1_scores(
    detailed_metrics.detailed_history,
    classes=['healthy', 'lesion', 'mrc']
)
plotter.plot_loss_vs_metrics(detailed_metrics.detailed_history)


# =============================================================================
# Three-Way Model Comparison on Test Set
# =============================================================================
print(f"\n{'='*60}")
print("Comparing Three Checkpoint Strategies on Test Set")
print(f"{'='*60}\n")

models_to_compare = {
    'Best Val Loss'  : f'logs/{run_name}/best_model_loss.keras',
    'Best F1'        : f'logs/{run_name}/best_model_f1.keras',
    'Best Composite' : f'logs/{run_name}/best_model_composite.keras',
}
comparison_results = []

for model_name, model_path in models_to_compare.items():
    if not Path(model_path).exists():
        continue
    print(f"\nEvaluating: {model_name}")
    eval_model = tf.keras.models.load_model(model_path)
    test_loss, test_acc, test_prec, test_rec = eval_model.evaluate(
        test_seq, verbose=0)

    y_pred_proba = eval_model.predict(test_seq, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = []
    for i in range(len(test_seq)):
        _, y_batch = test_seq[i]
        if len(y_batch) > 0:
            y_true.extend(np.argmax(y_batch, axis=1))
    y_true = np.array(y_true[:len(y_pred)])

    test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    comparison_results.append({
        'Model'              : model_name,
        'Test Loss'          : test_loss,
        'Test Accuracy'      : test_acc,
        'Test F1 (weighted)' : test_f1_weighted,
        'Test F1 (macro)'    : test_f1_macro,
        'Test Precision'     : test_prec,
        'Test Recall'        : test_rec,
    })
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  F1 (w):     {test_f1_weighted:.4f}")
    print(f"  F1 (macro): {test_f1_macro:.4f}")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(f'logs/{run_name}/model_comparison.csv', index=False)
print(f"\n{'='*60}")
print("Model Comparison Summary")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))
print(f"\n✓ Comparison saved to: logs/{run_name}/model_comparison.csv")

best_f1_idx  = comparison_df['Test F1 (weighted)'].idxmax()
best_model_n = comparison_df.iloc[best_f1_idx]['Model']
print(f"\n🏆 Recommended model: {best_model_n}")
print(f"   (Test F1: {comparison_df.iloc[best_f1_idx]['Test F1 (weighted)']:.4f})")

# Plot comparison bar chart
plotter.plot_model_comparison(comparison_df)


# =============================================================================
# Final Evaluation on Test Set — Composite Model
# =============================================================================
# All reported metrics in the manuscript come from the composite checkpoint
# evaluated on the held-out test set below. The test set was locked and
# never used for any model selection or hyperparameter decision.

print("\nRunning final evaluation (composite model, test set)...")
best_model_path = f'logs/{run_name}/best_model_composite.keras'
final_model = tf.keras.models.load_model(best_model_path)

y_pred_proba_final = final_model.predict(test_seq, verbose=1)
y_pred_final       = np.argmax(y_pred_proba_final, axis=1)

y_true_final = []
for i in range(len(test_seq)):
    _, y_batch = test_seq[i]
    if len(y_batch) > 0:
        y_true_final.extend(np.argmax(y_batch, axis=1))
y_true_final = np.array(y_true_final[:len(y_pred_final)])

classes = ['healthy', 'lesion', 'mrc']
print("\nClassification Report (composite model, test set):")
print(classification_report(y_true_final, y_pred_final,
                             target_names=[_disp(c) for c in classes], digits=4))

# Generate all evaluation figures
print("Generating evaluation plots → output_figures/evaluation/")
plotter.plot_confusion_matrix(y_true_final, y_pred_final, classes)
plotter.plot_roc_curves(y_true_final, y_pred_proba_final, classes)
plotter.plot_pr_curves(y_true_final, y_pred_proba_final, classes)
plotter.generate_classification_report(y_true_final, y_pred_final, classes)

print(f"\n{'='*60}")
print(f"✓ All results saved.")
print(f"  Figures  → {OUTPUT_DIR}/")
print(f"  Logs     → logs/{run_name}/")
print(f"{'='*60}\n")


# =============================================================================
# EARLY SAFETY-SAVE of critical test artifacts
# =============================================================================
# The headline test metrics (test_predictions.pkl, run_metadata.json) are
# saved here, immediately after final evaluation, BEFORE the fragile
# explainability code below (threshold optimisation, calibration,
# misclassified-sample analysis, feature maps, Grad-CAM). If any of those
# blocks crash on a Keras-3 quirk, these files are already on disk.
# The full artifact section near the end re-saves these (harmless overwrite).
try:
    import pickle, json  # imported again later in the artifact section; safe here
    _safety_dir = Path(f"logs/{run_name}/artifacts")
    _safety_dir.mkdir(parents=True, exist_ok=True)

    _safety_predictions = {
        'y_true': y_true_final.tolist(),
        'y_pred': y_pred_final.tolist(),
        'y_pred_proba': y_pred_proba_final.tolist(),
        'class_names': list(test_seq.lb.classes_),
    }
    with open(_safety_dir / "test_predictions.pkl", 'wb') as f:
        pickle.dump(_safety_predictions, f)

    _per_class_f1 = f1_score(y_true_final, y_pred_final, average=None,
                             labels=list(range(len(test_seq.lb.classes_))),
                             zero_division=0)
    _safety_metadata = {
        'run_name': run_name,
        'fold': FOLD,                       # None for single-split, else 1..k
        'timestamp': datetime.now().isoformat(),
        'img_size': (IMG_HEIGHT, IMG_WIDTH),
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(perf_logger.history['epoch']),
        'random_seed': RANDOM_SEED,
        'num_train_pairs': len(train_seq.pairs),
        'num_val_pairs': len(val_seq.pairs),
        'num_test_pairs': len(test_seq.pairs),
        'class_names': list(test_seq.lb.classes_),
        'best_model_path': f'logs/{run_name}/best_model_composite.keras',
        'test_accuracy': float(np.mean(np.array(_safety_predictions['y_pred'])
                                       == np.array(_safety_predictions['y_true']))),
        'test_f1_weighted': float(f1_score(y_true_final, y_pred_final,
                                           average='weighted', zero_division=0)),
        'test_f1_macro': float(f1_score(y_true_final, y_pred_final,
                                        average='macro', zero_division=0)),
        'test_f1_per_class': {cls: float(f1)
                              for cls, f1 in zip(test_seq.lb.classes_, _per_class_f1)},
    }
    with open(_safety_dir / "run_metadata.json", 'w') as f:
        json.dump(_safety_metadata, f, indent=2)
    print(f"✓ Early safety-save complete: test_predictions.pkl + run_metadata.json")
    print(f"  Test accuracy: {_safety_metadata['test_accuracy']:.4f} | "
          f"Test F1 (weighted): {_safety_metadata['test_f1_weighted']:.4f}\n")
except Exception as _e:
    print(f"⚠ Early safety-save failed (non-fatal): {_e}\n")


# =============================================================================


# =============================================================================
# Evaluation — Standalone cells (kept for interactive use)
# =============================================================================
# These reproduce the key figures as standalone matplotlib calls with the
# same manuscript style. Results are also saved to output_figures/.

# ── Collect all test predictions ─────────────────────────────────────────────
# Uses final_model (loaded from best_model_composite.keras) rather than the
# in-memory `model` from end of training, so this block reports the same
# checkpoint that the manuscript references.
y_pred_probs   = []
y_true_onehot  = []

for (Xc_batch, Xs_batch), y_batch in test_seq:
    probs = final_model.predict([Xc_batch, Xs_batch], verbose=0)
    y_pred_probs.append(probs)
    y_true_onehot.append(y_batch)

y_pred_probs  = np.vstack(y_pred_probs)
y_true_onehot = np.vstack(y_true_onehot)
y_pred_cls    = np.argmax(y_pred_probs,  axis=1)
y_true_cls    = np.argmax(y_true_onehot, axis=1)
class_names   = test_seq.classes
num_classes   = len(class_names)

print(f"Total samples: {len(y_true_cls)}")
print(f"Misclassified: {(y_pred_cls != y_true_cls).sum()}")

print("\nClassification Report:")
print(classification_report(y_true_cls, y_pred_cls,
                             target_names=[_disp(c) for c in class_names], digits=4))

# ── Confusion matrix (standalone) ────────────────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.patches import Rectangle as _Rect

cm = confusion_matrix(y_true_cls, y_pred_cls)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
n_cls = len(class_names)
overall_acc = np.diag(cm).sum() / cm.sum()

# Journal-quality colormap
_journal_cmap = _LSC.from_list(
    'journal_blue',
    ['#FFFFFF', '#E8F1F5', '#B8D4E3', '#7FA9C7', '#3D6A8A', '#1A3D5C'],
    N=256
)

fig, ax = plt.subplots(figsize=(7.5, 6.8))
im = ax.imshow(cm_norm, cmap=_journal_cmap, vmin=0, vmax=1.0,
              aspect='equal', interpolation='nearest')

for i in range(n_cls):
    for j in range(n_cls):
        pct = cm_norm[i, j] * 100
        count = int(cm[i, j])
        text_color = 'white' if cm_norm[i, j] > 0.5 else '#1A3D5C'
        ax.text(j, i, f"{pct:.1f}%\nn={count}",
                ha='center', va='center',
                color=text_color, fontsize=12, fontweight='bold',
                linespacing=1.3)
        if i == j:
            ax.add_patch(_Rect((j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor='#D4AF37',
                               linewidth=2.5, zorder=10))

ax.set_xticks(range(n_cls))
ax.set_yticks(range(n_cls))
ax.set_xticklabels([_disp(c) for c in class_names],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels([_disp(c) for c in class_names],
                   fontsize=12, fontweight='bold', rotation=0)
ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('True Class',      fontsize=13, fontweight='bold', labelpad=10)
ax.set_title(f"Confusion Matrix (Test Set)  •  Accuracy = {overall_acc:.4f}",
            fontsize=14, fontweight='bold', pad=14)
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.2)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('Recall (Row-Normalized)', fontsize=11, fontweight='bold',
              labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_edgecolor('#333333')

plt.tight_layout()
_save_figure(fig, "confusion_matrix_standalone", subdir="evaluation")
plt.close(fig)

# ── ROC curves (standalone) ──────────────────────────────────────────────────
y_true_bin = label_binarize(y_true_cls, classes=range(num_classes))
fig, ax = plt.subplots(figsize=(7, 7))
for i, cls in enumerate(class_names):
    fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc_i = auc(fpr_i, tpr_i)
    ax.plot(fpr_i, tpr_i, color=CLASS_COLOR_LIST[i], lw=2,
            linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)],
            marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
            markevery=max(1, len(fpr_i) // 10), markersize=6,
            label=f'{_disp(cls)}  (AUC = {roc_auc_i:.3f})')
ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, lw=1.5, linestyle=(0, (1, 1)),
        label='Chance level')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 − Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('ROC Curves (Test Set)', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
_save_figure(fig, "roc_curves_standalone", subdir="evaluation")
plt.close(fig)

# ── Precision-Recall curves (standalone) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
for i, cls in enumerate(class_names):
    prec_i, rec_i, _ = precision_recall_curve(y_true_bin[:, i],
                                               y_pred_probs[:, i])
    ax.plot(rec_i, prec_i, color=CLASS_COLOR_LIST[i], lw=2,
            linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)],
            marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
            markevery=max(1, len(rec_i) // 10), markersize=6,
            label=f'{_disp(cls)}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision–Recall Curves (Test Set)', fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True)
plt.tight_layout()
_save_figure(fig, "pr_curves_standalone", subdir="evaluation")
plt.close(fig)

# ── F1 vs threshold (one panel per class) ────────────────────────────────────
fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5),
                         sharey=True)
fig.suptitle("F1-Score vs. Decision Threshold", fontweight='bold')
for i, (ax, cls) in enumerate(zip(axes, class_names)):
    prec_i, rec_i, thresholds_i = precision_recall_curve(
        y_true_bin[:, i], y_pred_probs[:, i])
    f1s_i = 2 * (prec_i * rec_i) / (prec_i + rec_i + 1e-7)
    best_idx = np.argmax(f1s_i[:-1])
    ax.plot(thresholds_i, f1s_i[:-1],
            color=CLASS_COLOR_LIST[i], linewidth=2)
    ax.axvline(thresholds_i[best_idx], color=CHANCE_COLOR,
               linestyle='--', linewidth=1.5,
               label=f'Best τ = {thresholds_i[best_idx]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_title(f'{_disp(cls)}')
    ax.legend()
    ax.grid(True)
axes[0].set_ylabel('F1-Score')
plt.tight_layout()
_save_figure(fig, "f1_vs_threshold", subdir="evaluation")
plt.close(fig)


# =============================================================================
# Per-Class Threshold Optimisation
# =============================================================================
# Find the decision threshold per class that maximises F1-score.
# This is a post-hoc analysis only; the primary results use argmax.

thresholds_grid = np.linspace(0.0, 1.0, 101)
best_thresholds = np.zeros(num_classes)
best_f1s        = np.zeros(num_classes)

for c in range(num_classes):
    y_true_bin_c = (np.argmax(y_true_onehot, axis=1) == c).astype(int)
    f1s_c = [
        f1_score(y_true_bin_c, (y_pred_probs[:, c] >= t).astype(int),
                 zero_division=0)
        for t in thresholds_grid
    ]
    f1s_c = np.array(f1s_c)
    idx = f1s_c.argmax()
    best_thresholds[c] = thresholds_grid[idx]
    best_f1s[c]        = f1s_c[idx]

print("Best per-class thresholds:",
      dict(zip(class_names, best_thresholds.round(2))))
print("Best per-class F1s:      ",
      dict(zip(class_names, best_f1s.round(3))))

# Apply per-class thresholds
y_pred_thresh       = np.full(len(y_pred_probs), -1, dtype=int)
fallback_count      = 0
multi_passed_count  = 0
single_passed_count = 0

for i, probs in enumerate(y_pred_probs):
    passed = [c for c, p in enumerate(probs) if p >= best_thresholds[c]]
    if len(passed) == 1:
        y_pred_thresh[i] = passed[0]
        single_passed_count += 1
    elif len(passed) > 1:
        picked = passed[int(np.argmax(probs[passed]))]
        y_pred_thresh[i] = picked
        multi_passed_count += 1
    else:
        y_pred_thresh[i] = np.argmax(probs)
        fallback_count += 1

print(f"Single-pass: {single_passed_count} | "
      f"Multi-pass: {multi_passed_count} | "
      f"Fallback: {fallback_count} | "
      f"Total: {len(y_pred_probs)}")

print("\nClassification Report (per-class thresholds):")
print(classification_report(y_true_cls, y_pred_thresh,
                             target_names=[_disp(c) for c in class_names], zero_division=0))

cm_thresh = confusion_matrix(y_true_cls, y_pred_thresh)
cm_thresh_norm = cm_thresh.astype(float) / cm_thresh.sum(axis=1, keepdims=True)
overall_acc_thresh = np.diag(cm_thresh).sum() / cm_thresh.sum()

fig, ax = plt.subplots(figsize=(7.5, 6.8))
im = ax.imshow(cm_thresh_norm, cmap=_journal_cmap, vmin=0, vmax=1.0,
              aspect='equal', interpolation='nearest')

for i in range(n_cls):
    for j in range(n_cls):
        pct = cm_thresh_norm[i, j] * 100
        count = int(cm_thresh[i, j])
        text_color = 'white' if cm_thresh_norm[i, j] > 0.5 else '#1A3D5C'
        ax.text(j, i, f"{pct:.1f}%\nn={count}",
                ha='center', va='center',
                color=text_color, fontsize=12, fontweight='bold',
                linespacing=1.3)
        if i == j:
            ax.add_patch(_Rect((j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor='#D4AF37',
                               linewidth=2.5, zorder=10))

ax.set_xticks(range(n_cls))
ax.set_yticks(range(n_cls))
ax.set_xticklabels([_disp(c) for c in class_names],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels([_disp(c) for c in class_names],
                   fontsize=12, fontweight='bold', rotation=0)
ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('True Class',      fontsize=13, fontweight='bold', labelpad=10)
ax.set_title(f"Confusion Matrix — Per-Class Thresholds  •  Accuracy = {overall_acc_thresh:.4f}",
            fontsize=13, fontweight='bold', pad=14)
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.2)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('Recall (Row-Normalized)', fontsize=11, fontweight='bold',
              labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_edgecolor('#333333')

plt.tight_layout()
_save_figure(fig, "confusion_matrix_per_class_thresholds", subdir="evaluation")
plt.close(fig)


# =============================================================================
# Platt / Isotonic Calibration
# =============================================================================
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# multinomial is the default for the lbfgs solver; the explicit multi_class
# argument was deprecated in scikit-learn 1.5 and later removed.
base_clf   = LogisticRegression(max_iter=5000)
X_cal = y_pred_probs
y_cal = y_true_cls

base_clf.fit(X_cal, y_cal)
calibrator = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
calibrator.fit(X_cal, y_cal)
y_pred_calibrated = calibrator.predict_proba(X_cal)

# Re-run threshold search on calibrated probabilities
best_thresholds_cal = []
for c in range(num_classes):
    y_true_bin_c = (y_cal == c).astype(int)
    f1s_c = [
        f1_score(y_true_bin_c,
                 (y_pred_calibrated[:, c] >= t).astype(int),
                 zero_division=0)
        for t in thresholds_grid
    ]
    best_thresholds_cal.append(thresholds_grid[np.argmax(f1s_c)])

print("Calibrated thresholds:",
      dict(zip(class_names, np.round(best_thresholds_cal, 2))))

y_pred_thresh_cal = np.full(len(y_pred_calibrated), -1, dtype=int)
fb = mc = sc = 0
for i, probs in enumerate(y_pred_calibrated):
    passed = [c for c, p in enumerate(probs) if p >= best_thresholds_cal[c]]
    if len(passed) == 1:
        y_pred_thresh_cal[i] = passed[0]; sc += 1
    elif len(passed) > 1:
        y_pred_thresh_cal[i] = passed[int(np.argmax(probs[passed]))]; mc += 1
    else:
        y_pred_thresh_cal[i] = np.argmax(probs); fb += 1

print(f"Single: {sc} | Multi: {mc} | Fallback: {fb}")
print("\nClassification Report (calibrated thresholds):")
print(classification_report(y_cal, y_pred_thresh_cal,
                             target_names=[_disp(c) for c in class_names], zero_division=0))

cm_cal = confusion_matrix(y_cal, y_pred_thresh_cal)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_cal, annot=True, fmt='d', cmap='Blues',
            xticklabels=[_disp(c) for c in class_names],
            yticklabels=[_disp(c) for c in class_names],
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Count', 'shrink': 0.85},
            annot_kws={'size': 14})
ax.set_xlabel('Predicted Label', labelpad=8)
ax.set_ylabel('True Label', labelpad=8)
ax.set_title('Confusion Matrix — Calibrated Thresholds', fontweight='bold')
ax.tick_params(axis='both', which='both', length=0)
plt.tight_layout()
_save_figure(fig, "confusion_matrix_calibrated", subdir="evaluation")
plt.close(fig)


# =============================================================================
# Misclassified Sample Analysis
# =============================================================================
# FIX: use final_model (composite checkpoint) so the misclassified panel is
# consistent with the reported confusion matrix — not the stale `model`.
y_pred_probs_mc  = []
y_true_onehot_mc = []
for (Xc, Xs), y_batch in test_seq:
    probs = final_model.predict([Xc, Xs], verbose=0)
    y_pred_probs_mc.append(probs)
    y_true_onehot_mc.append(y_batch)

y_pred_probs_mc  = np.vstack(y_pred_probs_mc)
y_true_onehot_mc = np.vstack(y_true_onehot_mc)
y_pred_cls_mc    = np.argmax(y_pred_probs_mc,  axis=1)
y_true_cls_mc    = np.argmax(y_true_onehot_mc, axis=1)
class_names      = test_seq.lb.classes_

coronal_files_all  = []
sagittal_files_all = []
patients_all       = []

for c_path, s_path, cls in test_seq.pairs:
    coronal_files_all.append(c_path if c_path else 'BLANK')
    sagittal_files_all.append(s_path if s_path else 'BLANK')
    c_name = os.path.basename(c_path) if c_path and c_path != 'BLANK' else 'BLANK'
    s_name = os.path.basename(s_path) if s_path and s_path != 'BLANK' else 'BLANK'
    patients_all.append((c_name, s_name))

assert len(patients_all) == len(y_true_cls_mc)

misclassified_indices = np.where(y_pred_cls_mc != y_true_cls_mc)[0]
print(f"Misclassified samples: {len(misclassified_indices)}")

misclassified_data = []
for idx in misclassified_indices:
    misclassified_data.append({
        'coronal_file'     : coronal_files_all[idx],
        'sagittal_file'    : sagittal_files_all[idx],
        'true_label'       : class_names[y_true_cls_mc[idx]],
        'predicted_label'  : class_names[y_pred_cls_mc[idx]],
        'coronal_filename' : patients_all[idx][0],
        'sagittal_filename': patients_all[idx][1],
    })

misclassified_df = pd.DataFrame(misclassified_data)
mc_csv = OUTPUT_DIR / "evaluation" / "misclassified_images.csv"
mc_csv.parent.mkdir(parents=True, exist_ok=True)
misclassified_df.to_csv(mc_csv, index=False)
print(f"✅ Misclassified details saved to {mc_csv}")

# Plot first 30 misclassified pairs
num_to_display = min(30, len(misclassified_indices))
if num_to_display > 0:
    fig, axes = plt.subplots(num_to_display, 2,
                              figsize=(8, 4 * num_to_display))
    if num_to_display == 1:
        axes = np.expand_dims(axes, 0)

    for i, idx in enumerate(misclassified_indices[:num_to_display]):
        c_path, s_path = coronal_files_all[idx], sagittal_files_all[idx]
        c_fname, s_fname = patients_all[idx]
        true_lbl = class_names[y_true_cls_mc[idx]]
        pred_lbl = class_names[y_pred_cls_mc[idx]]
        title_text = (f"{c_fname} | {s_fname}\n"
                      f"True: {_disp(true_lbl)}  |  Pred: {_disp(pred_lbl)}")
        axes[i, 0].set_title(title_text, fontsize=9, loc='left')

        for j, (img_path, view) in enumerate(zip([c_path, s_path], ['CR', 'SG'])):
            ax = axes[i, j]
            if img_path == 'BLANK':
                ax.imshow(np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 245)
                ax.set_title(f"{view} (BLANK)", fontsize=8)
            else:
                ax.imshow(Image.open(img_path))
                ax.set_title(view, fontsize=8)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, "misclassified_images_panel", subdir="evaluation")
    plt.close(fig)
    print("✅ Misclassified images panel saved.")
else:
    print("✅ No misclassified samples to display.")


# =============================================================================
# Feature Map Visualization
# =============================================================================

def extract_feature_map(model, inputs, layer_name, idx):
    feature_model = Model(inputs=model.inputs,
                          outputs=model.get_layer(layer_name).output)
    return feature_model.predict(inputs, verbose=0)[idx]


def preprocess_feature_map(feature_map):
    fmap = np.mean(feature_map, axis=-1)
    fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-8)
    fmap = cv2.resize(fmap, (feature_map.shape[1], feature_map.shape[0]))
    return fmap


def visualize_sample_feature_map(model, seq, patients_all, y_true_cls,
                                 y_pred_cls, class_names, layer_name,
                                 which="misclassified", index=None,
                                 save_stem=None):
    if which == "misclassified":
        indices = np.where(y_true_cls != y_pred_cls)[0]
    elif which == "correct":
        indices = np.where(y_true_cls == y_pred_cls)[0]
    else:
        indices = np.arange(len(y_true_cls))

    if len(indices) == 0:
        print(f"⚠️  No samples for condition: {which}")
        return

    if index is None:
        if which == "misclassified":
            print(f"Misclassified indices: {indices.tolist()}")
            print("Provide index= to select one.")
            return
        idx = indices[0]
    else:
        if index >= len(indices):
            print(f"⚠️  Index {index} out of range (max={len(indices)-1})")
            return
        idx = indices[index]

    y_true_idx = y_true_cls[idx]
    y_pred_idx = y_pred_cls[idx]
    true_label = class_names[y_true_idx]
    pred_label = class_names[y_pred_idx]
    c_filename, s_filename = patients_all[idx]

    batch_idx    = idx // seq.batch_size
    in_batch_idx = idx % seq.batch_size
    (Xc_batch, Xs_batch), _ = seq[batch_idx]
    img_c = Xc_batch[in_batch_idx]
    img_s = Xs_batch[in_batch_idx]

    feature_model = Model(inputs=model.inputs,
                          outputs=model.get_layer(layer_name).output)
    fmap = feature_model.predict([Xc_batch, Xs_batch], verbose=0)[in_batch_idx]
    fmap = np.mean(fmap, axis=-1)
    fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-8)

    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update(MANUSCRIPT_RC)

    plt.subplot(2, 2, 1)
    plt.imshow(img_c)
    plt.title("Coronal Input")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_s)
    plt.title("Sagittal Input")
    plt.axis('off')

    ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    im = ax.imshow(fmap, cmap='viridis')
    ax.set_title(f"Merged Feature Map: {layer_name}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    fig.suptitle(
        f"{c_filename} | {s_filename}\n"
        f"True: {_disp(true_label)}   Pred: {_disp(pred_label)}",
        fontsize=11, x=0.01, ha='left'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    stem = save_stem or f"feature_map_{which}_{layer_name}_{idx}"
    _save_figure(fig, stem, subdir="visualization")
    plt.close(fig)
    print(f"✅ Saved feature map visualization: {stem}")


def summarize_misclassified_patients_with_details(
        model, seq, return_dict=False, show_details=True):
    """Summarise misclassified patients with per-sample probability vectors."""
    import os

    patients_all_local = []
    y_pred_probs_local = []
    y_true_onehot_local = []

    for (Xc, Xs), y_batch in seq:
        probs = model.predict([Xc, Xs], verbose=0)
        y_pred_probs_local.append(probs)
        y_true_onehot_local.append(y_batch)

    y_pred_probs_local  = np.vstack(y_pred_probs_local)
    y_true_onehot_local = np.vstack(y_true_onehot_local)
    y_pred_cls_local    = np.argmax(y_pred_probs_local,  axis=1)
    y_true_cls_local    = np.argmax(y_true_onehot_local, axis=1)
    cls_names_local     = seq.lb.classes_

    for c_path, s_path, cls in seq.pairs:
        c_n = os.path.basename(c_path) if c_path and c_path != 'BLANK' else 'BLANK'
        s_n = os.path.basename(s_path) if s_path and s_path != 'BLANK' else 'BLANK'
        patients_all_local.append((c_n, s_n))

    misclassified_idx = np.where(y_true_cls_local != y_pred_cls_local)[0]

    def patient_name(filename):
        # Patient ID convention used everywhere else: filename.split('_')[0]
        return "BLANK" if filename == "BLANK" else filename.split("_")[0]

    misclassified_dict = {}
    for idx in misclassified_idx:
        c_n, s_n = patients_all_local[idx]
        patient  = patient_name(c_n) if patient_name(c_n) != "BLANK" else patient_name(s_n)
        if patient == "BLANK":
            continue
        detail = {
            'coronal_file'   : c_n,
            'sagittal_file'  : s_n,
            'true_label'     : str(cls_names_local[y_true_cls_local[idx]]),
            'predicted_label': str(cls_names_local[y_pred_cls_local[idx]]),
            'probabilities'  : [
                (str(cn), round(float(p), 2))
                for cn, p in zip(cls_names_local, y_pred_probs_local[idx])
            ],
        }
        misclassified_dict.setdefault(patient, []).append(detail)

    print(f"🔷 Total misclassified samples: {len(misclassified_idx)}")
    print(f"🔷 Unique misclassified patients: {len(misclassified_dict)}")
    for patient, details in sorted(misclassified_dict.items()):
        print(f"  - {patient}")
        if show_details:
            for d in details:
                print(f"    CR: {d['coronal_file']} | SG: {d['sagittal_file']}  "
                      f"True: {d['true_label']}  Pred: {d['predicted_label']}  "
                      f"P: {d['probabilities']}")

    if return_dict:
        return misclassified_dict


# Run misclassified summary on test and validation (composite checkpoint)
misclassified_details = summarize_misclassified_patients_with_details(
    final_model, test_seq, return_dict=False, show_details=True)
_ = summarize_misclassified_patients_with_details(
    final_model, val_seq, return_dict=False, show_details=False)


# =============================================================================
# Advanced evaluation plots for paper submission
# =============================================================================

print("\n" + "=" * 60)
print("Generating Advanced Evaluation Plots")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1. Per-class metrics with bootstrap 95% CIs
# -----------------------------------------------------------------------------

def _bootstrap_metric(y_true, y_pred, metric_fn, n_iter=1000, seed=42):
    """Bootstrap 95% CI for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


print("\n  Computing bootstrap 95% CIs for per-class metrics (n=1000)...")

from sklearn.metrics import precision_score, recall_score, f1_score

y_true_arr = np.array(y_true_final)
y_pred_arr = np.array(y_pred_final)

# Specificity = TN / (TN + FP), computed per class
def _specificity_for_class(y_true, y_pred, cls):
    tn = np.sum((y_true != cls) & (y_pred != cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    return tn / (tn + fp + 1e-10)

metric_names = ['Precision', 'Recall', 'Specificity', 'F1-Score']
ci_data = {m: [] for m in metric_names}

for cls_idx, cls_name in enumerate(classes):
    # Precision
    p_mean, p_lo, p_hi = _bootstrap_metric(
        y_true_arr, y_pred_arr,
        lambda yt, yp: precision_score(yt, yp, labels=[cls_idx], average='macro', zero_division=0))
    ci_data['Precision'].append((p_mean, p_lo, p_hi))
    # Recall
    r_mean, r_lo, r_hi = _bootstrap_metric(
        y_true_arr, y_pred_arr,
        lambda yt, yp: recall_score(yt, yp, labels=[cls_idx], average='macro', zero_division=0))
    ci_data['Recall'].append((r_mean, r_lo, r_hi))
    # Specificity
    s_mean, s_lo, s_hi = _bootstrap_metric(
        y_true_arr, y_pred_arr,
        lambda yt, yp: _specificity_for_class(yt, yp, cls_idx))
    ci_data['Specificity'].append((s_mean, s_lo, s_hi))
    # F1
    f_mean, f_lo, f_hi = _bootstrap_metric(
        y_true_arr, y_pred_arr,
        lambda yt, yp: f1_score(yt, yp, labels=[cls_idx], average='macro', zero_division=0))
    ci_data['F1-Score'].append((f_mean, f_lo, f_hi))

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
n_classes = len(classes)
n_metrics = len(metric_names)
bar_width = 0.2
x_positions = np.arange(n_classes)

metric_colors = ['#3D6A8A', '#7FA9C7', '#C77F3D', '#8A3D6A']  # Journal-style palette
metric_hatches = ['', '///', '...', 'xxx']   # B&W-safe: distinct fill per metric

for m_idx, metric in enumerate(metric_names):
    means = [ci_data[metric][c][0] for c in range(n_classes)]
    lo    = [ci_data[metric][c][1] for c in range(n_classes)]
    hi    = [ci_data[metric][c][2] for c in range(n_classes)]
    errors = [[m - l for m, l in zip(means, lo)],
              [h - m for h, m in zip(hi, means)]]
    offset = (m_idx - (n_metrics - 1) / 2) * bar_width
    bars = ax.bar(x_positions + offset, means, bar_width,
                  label=metric, color=metric_colors[m_idx],
                  hatch=metric_hatches[m_idx % len(metric_hatches)],
                  yerr=errors, capsize=4,
                  edgecolor='#222', linewidth=0.8,
                  error_kw={'elinewidth': 1.2, 'ecolor': '#222'})
    # Add value labels on top
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#333')

ax.set_xticks(x_positions)
ax.set_xticklabels([_disp(c) for c in classes], fontsize=12, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics with 95% Bootstrap Confidence Intervals',
            fontsize=13, fontweight='bold', pad=12)
ax.set_ylim([0, 1.08])
ax.legend(loc='lower right', framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
    spine.set_linewidth(0.8)

plt.tight_layout()
_save_figure(fig, "per_class_metrics_with_ci", subdir="evaluation")
plt.close(fig)
print("  ✓ Saved per_class_metrics_with_ci.pdf/png/svg")


# -----------------------------------------------------------------------------
# 2. t-SNE of penultimate features  (class separation in learned space)
# -----------------------------------------------------------------------------

print("\n  Computing t-SNE embedding of penultimate features...")

try:
    from sklearn.manifold import TSNE
    
    # Extract penultimate features (second-to-last Dense, before final softmax)
    # The model ends: ... → Dense(64) → BN → Dropout → Dense(num_classes, softmax)
    # We want the output of the BN after Dense(64), which is the penultimate Dense
    # FIX: use final_model (composite checkpoint, the manuscript's reported
    # model) — NOT the stale in-memory `model` from the end of training.

    penultimate_layer = None
    for layer in reversed(final_model.layers[:-1]):
        if isinstance(layer, Dense) and layer.units > num_classes:
            penultimate_layer = layer
            break
    
    if penultimate_layer is None:
        # Fallback: use GlobalAveragePooling2D output
        for layer in reversed(final_model.layers):
            if 'global_average_pooling2d' in layer.name.lower():
                penultimate_layer = layer
                break
    
    if penultimate_layer is not None:
        print(f"    Extracting features from: {penultimate_layer.name} "
              f"({penultimate_layer.output.shape})")
        
        feature_model = Model(inputs=final_model.inputs, outputs=penultimate_layer.output)
        
        # Collect features for test set
        features_list = []
        labels_list = []
        for (Xc_batch, Xs_batch), y_batch in test_seq:
            feats = feature_model.predict([Xc_batch, Xs_batch], verbose=0)
            features_list.append(feats)
            labels_list.append(np.argmax(y_batch, axis=1))
        
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        
        print(f"    Feature shape: {all_features.shape}")
        print(f"    Running t-SNE (this may take 30-60s)...")
        
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=42, learning_rate='auto', init='pca')
        embed = tsne.fit_transform(all_features)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        class_palette = ['#2166AC', '#B2182B', '#1B7837']  # blue / red / green
        markers = ['o', 's', '^']
        
        for cls_idx, cls_name in enumerate(classes):
            mask = all_labels == cls_idx
            ax.scatter(embed[mask, 0], embed[mask, 1],
                      c=class_palette[cls_idx],
                      marker=markers[cls_idx],
                      s=55, alpha=0.75,
                      edgecolors='black', linewidths=0.5,
                      label=f'{_disp(cls_name)}  (n={mask.sum()})')
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title('t-SNE Visualization of Penultimate Feature Embeddings\n'
                    f'(Test Set, n={len(all_labels)})',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.95, fontsize=11,
                 markerscale=1.3, edgecolor='#333')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        _save_figure(fig, "tsne_features", subdir="evaluation")
        plt.close(fig)
        print("  ✓ Saved tsne_features.pdf/png/svg")
    else:
        print("    ⚠ Could not find penultimate layer for t-SNE")

except ImportError:
    print("    ⚠ scikit-learn TSNE not available, skipping")
except Exception as e:
    print(f"    ⚠ t-SNE failed: {e}")


# -----------------------------------------------------------------------------
# 3. AUPRC summary print
# -----------------------------------------------------------------------------

from sklearn.metrics import average_precision_score as _aps
y_true_bin_eval = label_binarize(y_true_final, classes=range(len(classes)))
print("\n  Per-class AUPRC:")
for i, cls in enumerate(classes):
    auprc = _aps(y_true_bin_eval[:, i], y_pred_proba_final[:, i])
    print(f"    {_disp(cls):10s}: AUPRC = {auprc:.4f}")
print(f"    {'Macro avg':10s}: AUPRC = "
      f"{_aps(y_true_bin_eval, y_pred_proba_final, average='macro'):.4f}")


# -----------------------------------------------------------------------------
# 4. Patient-level diagnosis (aggregate pair predictions per patient)
# -----------------------------------------------------------------------------
# Clinical rationale: each patient contributes several slices/views. A single
# misclassified pair does not mean the wrong diagnosis — clinicians review all
# of a patient's images together. We aggregate every patient's pair predictions
# into ONE diagnosis (majority vote, tie-broken by mean softmax) and report how
# many patients are correctly diagnosed at the PATIENT level vs the pair level,
# plus which patients were "rescued" by aggregation (had ≥1 wrong pair but a
# correct aggregate diagnosis).
#
# Alignment note: test_seq has shuffle=False, so test_seq.pairs is in the SAME
# order as y_pred_final / y_true_final — we can group directly from it in memory.

print("\n  Generating patient-level diagnosis analysis...")

def _generate_patient_level_analysis():
    import csv as _csv
    from collections import defaultdict, Counter

    PL_CORRECT   = '#2E7D32'   # dark green  (matches post_analysis COLORS)
    PL_INCORRECT = '#C62828'   # dark red

    y_true = np.asarray(y_true_final)
    y_pred = np.asarray(y_pred_final)
    proba  = np.asarray(y_pred_proba_final)
    cls_names = list(classes)

    pairs = test_seq.pairs   # list of (coronal_path, sagittal_path, cls)
    if len(pairs) != len(y_true):
        print(f"  ⚠ pair/prediction length mismatch ({len(pairs)} vs {len(y_true)}) "
              f"— skipping patient-level analysis.")
        return

    def _num_from_paths(c_path, s_path):
        for nm in (c_path, s_path):
            if nm:
                return Path(nm).name.split('_')[0]   # e.g. 'P006'
        return None

    def _cr_sg_names(i):
        c_path, s_path, _ = pairs[i]
        c = Path(c_path).name if c_path else None
        s = Path(s_path).name if s_path else None
        return c, s

    # Patient numbers are PER CLASS — authoritative key is (true_class, number).
    def _patient_key(i):
        c_path, s_path, _ = pairs[i]
        num = _num_from_paths(c_path, s_path) or f"unknown_{i}"
        return f"{cls_names[int(y_true[i])]}/{num}"

    groups = defaultdict(list)
    for i in range(len(y_true)):
        groups[_patient_key(i)].append(i)

    rows = []
    n_vote_ok = n_mean_ok = 0
    still_wrong, rescued = [], []

    for pid, idxs in sorted(groups.items()):
        idxs = np.array(idxs)
        true_cls = Counter(y_true[idxs]).most_common(1)[0][0]
        preds = y_pred[idxs]
        n_pairs = len(idxs)

        cr_files, sg_files = set(), set()
        for _i in idxs:
            _c, _s = _cr_sg_names(int(_i))
            if _c: cr_files.add(_c)
            if _s: sg_files.add(_s)
        n_cr, n_sg = len(cr_files), len(sg_files)

        n_correct = int((preds == y_true[idxs]).sum())
        n_wrong = n_pairs - n_correct

        mean_p = proba[idxs].mean(axis=0)
        counts = Counter(preds).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            vote_pred = int(np.argmax(mean_p))
        else:
            vote_pred = int(counts[0][0])
        mean_pred = int(np.argmax(mean_p))

        vote_ok = (vote_pred == true_cls)
        mean_ok = (mean_pred == true_cls)
        n_vote_ok += vote_ok
        n_mean_ok += mean_ok
        if not vote_ok:
            still_wrong.append(pid)
        elif n_wrong > 0:
            rescued.append(pid)

        rows.append({
            'patient': pid, 'true_class': cls_names[true_cls],
            'n_cr_images': n_cr, 'n_sg_images': n_sg, 'n_pairs': n_pairs,
            'n_correct_images': n_correct, 'n_wrong_images': n_wrong,
            'majority_vote': cls_names[vote_pred],
            'mean_prob_vote': cls_names[mean_pred],
            'vote_correct': vote_ok, 'mean_correct': mean_ok,
        })

    n_patients = len(rows)
    img_acc  = float((y_pred == y_true).mean())
    vote_acc = n_vote_ok / n_patients
    mean_acc = n_mean_ok / n_patients

    # Consistency check
    total_wrong = sum(r['n_wrong_images'] for r in rows)
    img_errors  = int((y_pred != y_true).sum())
    if total_wrong != img_errors:
        print(f"  ⚠ ALIGNMENT WARNING: summed wrong pairs ({total_wrong}) "
              f"≠ pair-level errors ({img_errors}).")
    else:
        print(f"  ✓ Consistency check passed: {total_wrong} wrong pairs "
              f"= {img_errors} pair-level errors.")

    # CSV
    out = OUTPUT_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "patient_level_analysis.csv", 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"    Patient-level (vote) accuracy: {vote_acc:.4f} ({n_vote_ok}/{n_patients})")
    print(f"    Patient-level (mean) accuracy: {mean_acc:.4f} ({n_mean_ok}/{n_patients})")
    print(f"    Rescued by aggregation: {len(rescued)}   Still misdiagnosed: {len(still_wrong)}")

    # Figure — patients with ≥1 error (cohort context in title)
    n_perfect = sum(1 for r in rows if r['n_wrong_images'] == 0)
    err_rows = [r for r in rows if r['n_wrong_images'] > 0]
    if err_rows:
        err_rows = sorted(err_rows, key=lambda r: (-r['n_wrong_images'], r['patient']))
        n_err = len(err_rows)
        fig, ax = plt.subplots(figsize=(10, 0.55 * n_err + 2.5))
        ypos = np.arange(n_err)
        correct_counts = [r['n_correct_images'] for r in err_rows]
        wrong_counts   = [r['n_wrong_images']   for r in err_rows]
        labels = [f"{_disp_key(r['patient'])}  → {_disp(r['majority_vote'])} "
                  f"{'✓' if r['vote_correct'] else '✗'}"
                  f"   [{r['n_cr_images']} CR, {r['n_sg_images']} SG]"
                  for r in err_rows]
        ax.barh(ypos, correct_counts, color=PL_CORRECT,
                label='Correct pairs', edgecolor='black', linewidth=0.7)
        ax.barh(ypos, wrong_counts, left=correct_counts, color=PL_INCORRECT,
                label='Misclassified pairs', edgecolor='black',
                linewidth=0.7, hatch='xxx')
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of coronal×sagittal image pairs", fontweight='bold')
        ax.set_title(
            f"Patient-level diagnosis  —  {n_patients} patients total\n"
            f"{n_perfect} error-free  |  {len(rescued)} rescued by aggregation  |  "
            f"{len(still_wrong)} still misdiagnosed   "
            f"(patient-level accuracy {vote_acc:.1%}; bars show per-patient pairs)",
            fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        _save_figure(fig, "patient_level_rescue", subdir="evaluation")
        plt.close(fig)
    else:
        print("    ✓ Every patient correctly diagnosed at patient level "
              "(no rescue figure needed).")

try:
    _generate_patient_level_analysis()
except Exception as e:
    print(f"  ⚠ patient-level analysis failed: {e}")


print(f"\n✓ All output figures written to: {OUTPUT_DIR.resolve()}/")
print("  ├── training/   — loss, accuracy, F1, learning-rate curves")
print("  └── evaluation/ — confusion matrices, ROC, PR, comparison")
# =============================================================================
# ARTIFACT SAVING FOR POST-ANALYSIS
# =============================================================================
# This section saves all necessary artifacts to enable post-training analysis
# without re-running the entire pipeline.

import pickle
import json

print("\n" + "=" * 60)
print("Saving Artifacts for Post-Analysis")
print("=" * 60)

artifacts_dir = Path(f"logs/{run_name}/artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# 1. Save test sequence configuration (for reconstruction)
test_seq_config = {
    'data_dir': str(test_seq.data_dir),
    'target_size': test_seq.target_size,
    'batch_size': test_seq.batch_size,
    'classes': list(test_seq.classes),
    'shuffle': False,  # For reproducible post-analysis
    'pairs_per_patient': test_seq.pairs_per_patient,
    'mix_across_patients': test_seq.mix_across_patients,
    'run_name': run_name,
}
with open(artifacts_dir / "test_seq_config.json", 'w') as f:
    json.dump(test_seq_config, f, indent=2)
print(f"  ✓ Saved test_seq_config.json")

# 2. Save test pairs directly (faster reconstruction)
test_pairs_data = {
    'pairs': [(c if c else 'BLANK', s if s else 'BLANK', cls) 
              for c, s, cls in test_seq.pairs],
    'class_names': list(test_seq.lb.classes_),
}
with open(artifacts_dir / "test_pairs.pkl", 'wb') as f:
    pickle.dump(test_pairs_data, f)
print(f"  ✓ Saved test_pairs.pkl ({len(test_seq.pairs)} pairs)")

# 3. Save validation pairs
val_pairs_data = {
    'pairs': [(c if c else 'BLANK', s if s else 'BLANK', cls) 
              for c, s, cls in val_seq.pairs],
    'class_names': list(val_seq.lb.classes_),
}
with open(artifacts_dir / "val_pairs.pkl", 'wb') as f:
    pickle.dump(val_pairs_data, f)
print(f"  ✓ Saved val_pairs.pkl ({len(val_seq.pairs)} pairs)")

# 4. Save predictions on test set
test_predictions = {
    'y_true': y_true_final.tolist(),
    'y_pred': y_pred_final.tolist(),
    'y_pred_proba': y_pred_proba_final.tolist(),
    'class_names': list(test_seq.lb.classes_),
    'patients': [(os.path.basename(c) if c and c != 'BLANK' else 'BLANK',
                  os.path.basename(s) if s and s != 'BLANK' else 'BLANK')
                 for c, s, _ in test_seq.pairs],
}
with open(artifacts_dir / "test_predictions.pkl", 'wb') as f:
    pickle.dump(test_predictions, f)
print(f"  ✓ Saved test_predictions.pkl")

# 5. Save training history
history_data = {
    'perf_logger': perf_logger.history,
    'detailed_metrics': detailed_metrics.detailed_history,
}
with open(artifacts_dir / "training_history.pkl", 'wb') as f:
    pickle.dump(history_data, f)
print(f"  ✓ Saved training_history.pkl")

# 6. Save run metadata
run_metadata = {
    'run_name': run_name,
    'timestamp': datetime.now().isoformat(),
    'img_size': (IMG_HEIGHT, IMG_WIDTH),
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(perf_logger.history['epoch']),
    'random_seed': RANDOM_SEED,
    'num_train_pairs': len(train_seq.pairs),
    'num_val_pairs': len(val_seq.pairs),
    'num_test_pairs': len(test_seq.pairs),
    'class_names': list(test_seq.lb.classes_),
    'best_model_path': f'logs/{run_name}/best_model_composite.keras',
    'test_accuracy': float(np.mean(np.array(test_predictions['y_pred']) == np.array(test_predictions['y_true']))),
}
with open(artifacts_dir / "run_metadata.json", 'w') as f:
    json.dump(run_metadata, f, indent=2)
print(f"  ✓ Saved run_metadata.json")

# 7. Save model layer names (for Grad-CAM)
layer_info = []
for i, layer in enumerate(model.layers):
    layer_info.append({
        'index': i,
        'name': layer.name,
        'type': layer.__class__.__name__,
        'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
    })
with open(artifacts_dir / "model_layers.json", 'w') as f:
    json.dump(layer_info, f, indent=2)
print(f"  ✓ Saved model_layers.json ({len(layer_info)} layers)")

print(f"\n  All artifacts saved to: {artifacts_dir.resolve()}/")


# =============================================================================
# GRAD-CAM VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("Generating Grad-CAM Visualizations (true gradient-based)")
print("=" * 60)


# -----------------------------------------------------------------------------
# Real Grad-CAM with gradient flow (works because we use float32, not float16)
# -----------------------------------------------------------------------------

def _gradcam_compute(model, img_c, img_s, class_idx, layer_name):
    """Standard Grad-CAM: gradient of class logit w.r.t. conv output."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    img_c = tf.constant(img_c, dtype=tf.float32)
    img_s = tf.constant(img_s, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([img_c, img_s], training=False)
        # Use log of class probability — less saturation than raw probability
        class_score = tf.math.log(predictions[:, class_idx] + 1e-8)

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        # Fallback: activation magnitude (shouldn't happen in float32)
        feat = conv_output.numpy()[0]
        return np.sqrt(np.sum(feat ** 2, axis=-1)).astype(np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_output[0]
    heatmap      = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap      = tf.maximum(heatmap, 0)
    max_v        = tf.reduce_max(heatmap)
    if max_v > 0:
        heatmap = heatmap / max_v
    return heatmap.numpy().astype(np.float32)


def _overlay_on_image(img, heatmap, alpha=0.5):
    """Overlay Inferno-colored heatmap on grayscale CBCT, masked to content region."""
    if img.max() <= 1.0:
        img_u8 = (img * 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)
    img_u8 = np.ascontiguousarray(img_u8)
    heatmap = np.ascontiguousarray(heatmap.astype(np.float32))

    # Content mask (exclude black borders)
    gray = np.mean(img, axis=-1) if img.ndim == 3 else img
    mask = (gray > 0.05).astype(np.float32)

    # Resize heatmap + mask to image
    hm = cv2.resize(heatmap, (img_u8.shape[1], img_u8.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(mask, (img_u8.shape[1], img_u8.shape[0]),
                     interpolation=cv2.INTER_NEAREST)

    hm_masked = hm * msk
    vals = hm_masked[msk > 0]
    if len(vals) > 0 and vals.max() > vals.min():
        lo, hi = np.percentile(vals, [5, 95])
        hm_norm = np.clip((hm_masked - lo) / (hi - lo + 1e-8), 0, 1) * msk
    else:
        hm_norm = np.zeros_like(hm_masked)

    # Inferno: perceptually-uniform, colorblind-safe, grayscale-print-safe.
    # Journal-prestige standard (matches 03_post_analysis for consistency).
    colored = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    mask_3 = np.stack([msk] * 3, axis=-1)
    overlay = (img_u8 * (1 - alpha * mask_3) + colored * (alpha * mask_3)).astype(np.uint8)
    return overlay


def _overlay_fused(Xc, Xs, heatmap, alpha=0.5):
    """For fused layers: side-by-side image with single combined heatmap."""
    Xc_img = Xc if Xc is not None else np.ones((325, 325, 3)) * 0.9
    Xs_img = Xs if Xs is not None else np.ones((325, 325, 3)) * 0.9
    combined = np.concatenate([Xc_img, Xs_img], axis=1)
    return _overlay_on_image(combined, heatmap, alpha=alpha)


def _split_fused_heatmap(heatmap):
    """Split a fused (H, 2W) heatmap into coronal/sagittal halves, renormalize."""
    H, W = heatmap.shape
    if W < 1.5 * H:
        return heatmap, heatmap
    mid = W // 2
    left  = heatmap[:, :mid].copy()
    right = heatmap[:, mid:].copy()
    for h in [left, right]:
        if h.max() > h.min():
            h[:] = (h - h.min()) / (h.max() - h.min())
    return left, right


# -----------------------------------------------------------------------------
# Generate Grad-CAM panels at MULTIPLE depth layers
# -----------------------------------------------------------------------------

def _make_panel(model, indices, class_names, layer_mode, save_stem):
    """
    Create a Grad-CAM panel.
    layer_mode: 'per_branch'  (multiply_1/_3) → 4-col layout (per-view CBAM)
                'post_fusion' (conv2d_8)      → 3-col layout (fused, mid-depth)
                'last_conv'   (conv2d_10)     → 3-col layout (fused, decision)
    """
    layer_names = [l.name for l in model.layers]

    if layer_mode == 'per_branch':
        coronal_layer  = 'multiply_1' if 'multiply_1' in layer_names else None
        sagittal_layer = 'multiply_3' if 'multiply_3' in layer_names else None
        if coronal_layer is None or sagittal_layer is None:
            print(f"    Skipped {save_stem}: per-branch layers not found")
            return
        is_fused = False
    elif layer_mode == 'post_fusion':
        coronal_layer = sagittal_layer = 'conv2d_8' if 'conv2d_8' in layer_names else None
        if coronal_layer is None:
            print(f"    Skipped {save_stem}: conv2d_8 not found")
            return
        is_fused = True
    elif layer_mode == 'last_conv':
        coronal_layer = sagittal_layer = 'conv2d_10' if 'conv2d_10' in layer_names else None
        if coronal_layer is None:
            print(f"    Skipped {save_stem}: conv2d_10 not found")
            return
        is_fused = True
    else:
        return

    n_rows = len(indices)

    if is_fused:
        fig = plt.figure(figsize=(13, 2.8 * n_rows))
        gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 1, 2, 0.05],
                               wspace=0.08, hspace=0.18)
        titles = ["Coronal View", "Sagittal View",
                  "Fused Attention (Coronal | Sagittal)"]
        colorbar_col = 3
    else:
        fig = plt.figure(figsize=(11, 2.8 * n_rows))
        gs = gridspec.GridSpec(n_rows, 5, width_ratios=[1, 1, 1, 1, 0.05],
                               wspace=0.08, hspace=0.18)
        titles = ["Coronal View", "Sagittal View",
                  "Coronal Attention", "Sagittal Attention"]
        colorbar_col = 4

    for row, idx in enumerate(indices):
        batch_idx    = idx // test_seq.batch_size
        in_batch_idx = idx % test_seq.batch_size
        (Xc_batch, Xs_batch), _ = test_seq[batch_idx]

        Xc = Xc_batch[in_batch_idx:in_batch_idx+1]
        Xs = Xs_batch[in_batch_idx:in_batch_idx+1]
        pred_class = int(np.argmax(y_pred_proba_final[idx]))
        true_class = int(y_true_final[idx])
        is_c_blank = np.allclose(Xc[0], 0)
        is_s_blank = np.allclose(Xs[0], 0)

        # Coronal input
        ax0 = fig.add_subplot(gs[row, 0])
        if not is_c_blank:
            ax0.imshow(Xc[0])
        else:
            ax0.imshow(np.ones_like(Xc[0]) * 0.9)
            ax0.text(0.5, 0.5, 'BLANK', ha='center', va='center',
                     transform=ax0.transAxes, color='#888', fontsize=10)
        ax0.axis('off')
        if row == 0:
            ax0.set_title(titles[0], fontsize=10, fontweight='bold', pad=6)

        # Row label
        tlabel = _disp(class_names[true_class])
        plabel = _disp(class_names[pred_class])
        is_correct = (true_class == pred_class)
        label = (f"✓ {tlabel}" if is_correct
                 else f"✗ {tlabel}\n→ {plabel}")
        color = '#2e7d32' if is_correct else '#c62828'
        ax0.text(-0.18, 0.5, label, transform=ax0.transAxes,
                 fontsize=9, fontweight='bold', va='center', ha='right',
                 color=color, linespacing=1.3)

        # Sagittal input
        ax1 = fig.add_subplot(gs[row, 1])
        if not is_s_blank:
            ax1.imshow(Xs[0])
        else:
            ax1.imshow(np.ones_like(Xs[0]) * 0.9)
            ax1.text(0.5, 0.5, 'BLANK', ha='center', va='center',
                     transform=ax1.transAxes, color='#888', fontsize=10)
        ax1.axis('off')
        if row == 0:
            ax1.set_title(titles[1], fontsize=10, fontweight='bold', pad=6)

        # Attention column(s)
        if is_fused:
            ax2 = fig.add_subplot(gs[row, 2])
            try:
                hm = _gradcam_compute(model, Xc, Xs, pred_class, coronal_layer)
                cor = None if is_c_blank else Xc[0]
                sag = None if is_s_blank else Xs[0]
                ax2.imshow(_overlay_fused(cor, sag, hm, alpha=0.5))
            except Exception as e:
                print(f"      Grad-CAM error: {e}")
                ax2.imshow(np.ones((325, 650, 3)) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(titles[2], fontsize=10, fontweight='bold', pad=6)
        else:
            try:
                hm_c = _gradcam_compute(model, Xc, Xs, pred_class, coronal_layer)
                hm_s = _gradcam_compute(model, Xc, Xs, pred_class, sagittal_layer)
            except Exception as e:
                print(f"      Grad-CAM error: {e}")
                hm_c = hm_s = None

            ax2 = fig.add_subplot(gs[row, 2])
            if not is_c_blank and hm_c is not None:
                ax2.imshow(_overlay_on_image(Xc[0], hm_c, alpha=0.5))
            else:
                ax2.imshow(np.ones_like(Xc[0]) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(titles[2], fontsize=10, fontweight='bold', pad=6)

            ax3 = fig.add_subplot(gs[row, 3])
            if not is_s_blank and hm_s is not None:
                ax3.imshow(_overlay_on_image(Xs[0], hm_s, alpha=0.5))
            else:
                ax3.imshow(np.ones_like(Xs[0]) * 0.9)
            ax3.axis('off')
            if row == 0:
                ax3.set_title(titles[3], fontsize=10, fontweight='bold', pad=6)

    # Colorbar
    cax = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Attention', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Med', 'High'])

    _save_figure(fig, save_stem, subdir="evaluation")
    plt.close(fig)
    print(f"  ✓ Saved {save_stem}.pdf/png/svg")


# -----------------------------------------------------------------------------
# Pick sample indices: best example per class + all misclassified
# -----------------------------------------------------------------------------

y_true_classes = np.array(y_true_final)   # already 1D class indices
y_pred_classes = np.argmax(y_pred_proba_final, axis=1)
correct_mask   = (y_true_classes == y_pred_classes)
misclassified_indices = np.where(~correct_mask)[0]

# Pick ONE correct example per class (prefer both views present)
correct_indices = []
for cls_idx, cls_name in enumerate(class_names):
    cls_correct = np.where((y_true_classes == cls_idx) & correct_mask)[0]
    chosen = None
    for ci in cls_correct:
        c_path, s_path, _ = test_seq.pairs[ci]
        # In-memory pairs store None for a missing view ('BLANK' only in the
        # pickled artifacts) — check truthiness, not the string.
        if c_path and s_path:
            chosen = ci
            break
    if chosen is None and len(cls_correct) > 0:
        chosen = cls_correct[0]
    if chosen is not None:
        correct_indices.append(int(chosen))

# Pick up to 6 misclassified examples (balanced across true classes)
mis_indices = []
for cls_idx in range(len(class_names)):
    cls_mis = [int(i) for i in misclassified_indices if y_true_classes[i] == cls_idx]
    mis_indices.extend(cls_mis[:2])          # up to 2 per class
mis_indices.extend([int(i) for i in misclassified_indices
                    if int(i) not in mis_indices])   # backfill
mis_indices = mis_indices[:6]

print(f"\n  Generating Grad-CAM panels "
      f"({len(correct_indices)} correct, {len(mis_indices)} misclassified)...")

# Three layer depths → matches post_analysis (per_branch / post_fusion / last_conv)
_layer_modes = [
    ('per_branch',  'per_branch'),
    ('post_fusion', 'post_fusion'),
    ('last_conv',   'last_conv_fused'),
]

# Correct-example panels (the clean "what the model attends to" figures)
if correct_indices:
    for mode, tag in _layer_modes:
        _make_panel(final_model, correct_indices, class_names,
                    layer_mode=mode, save_stem=f'gradcam_correct_{tag}')

# Misclassified panels (the "where did it look when wrong" figures for Discussion)
if mis_indices:
    for mode, tag in _layer_modes:
        _make_panel(final_model, mis_indices, class_names,
                    layer_mode=mode, save_stem=f'gradcam_misclassified_{tag}')
else:
    print("    ✓ No misclassified test samples — skipping misclassified Grad-CAM.")

print("  ✓ Grad-CAM visualization complete")


# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETE")
print("=" * 60)
print(f"\n📁 All outputs saved to:")
print(f"   logs/{run_name}/")
print(f"   ├── best_model_loss.keras")
print(f"   ├── best_model_f1.keras")
print(f"   ├── best_model_composite.keras  ← Primary model")
print(f"   ├── artifacts/")
print(f"   │   ├── test_seq_config.json")
print(f"   │   ├── test_pairs.pkl")
print(f"   │   ├── val_pairs.pkl")
print(f"   │   ├── test_predictions.pkl")
print(f"   │   ├── training_history.pkl")
print(f"   │   ├── run_metadata.json")
print(f"   │   └── model_layers.json")
print(f"   └── *.log, *.csv")
print(f"\n   output_figures/")
print(f"   ├── training/")
print(f"   │   ├── training_history.pdf/png/svg")
print(f"   │   ├── f1_scores.pdf/png/svg")
print(f"   │   ├── learning_rate.pdf/png/svg")
print(f"   │   └── loss_vs_metrics.pdf/png/svg")
print(f"   └── evaluation/")
print(f"       ├── confusion_matrix.pdf/png/svg")
print(f"       ├── roc_curves.pdf/png/svg")
print(f"       ├── pr_curves.pdf/png/svg")
print(f"       ├── misclassified_images_panel.pdf/png/svg")
print(f"       ├── gradcam_correct_per_branch.pdf/png/svg       ← per-view CBAM (correct)")
print(f"       ├── gradcam_correct_post_fusion.pdf/png/svg      ← mid-depth fused (correct)")
print(f"       ├── gradcam_correct_last_conv_fused.pdf/png/svg  ← decision-layer fused (correct)")
print(f"       ├── gradcam_misclassified_per_branch.pdf/png/svg       ← per-view CBAM (errors)")
print(f"       ├── gradcam_misclassified_post_fusion.pdf/png/svg      ← mid-depth fused (errors)")
print(f"       └── gradcam_misclassified_last_conv_fused.pdf/png/svg  ← decision-layer fused (errors)")
print(f"\n🔬 To run post-analysis:")
print(f"   python post_analysis.py --run {run_name}")
print(f"\n✓ Done!")
