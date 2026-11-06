#!/usr/bin/env python3
"""
ODIN Post-Training Analysis — Publication Quality
===================================================
Generate paper-ready Grad-CAM visualizations from saved artifacts.

Usage:
    python 03_post_analysis.py --run run_20260520_153721
    python 03_post_analysis.py --run run_20260520_153721 --gradcam
    python 03_post_analysis.py --run run_20260520_153721 --tsne
    python 03_post_analysis.py --run run_20260520_153721 --all

K-fold usage (run names are namespaced under fold_k/):
    python 03_post_analysis.py --run fold_1/run_20260520_153721 --all \
        --dataset-root kfold_data/fold_1
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dense, Dropout, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Concatenate, Add, Multiply, Reshape)
from tensorflow.keras.regularizers import l2


# =============================================================================
# CUSTOM KERAS FUNCTIONS (required for model loading)
# =============================================================================

@register_keras_serializable()
def slice_coronal_gate(x):
    return x[:, 0:1]

@register_keras_serializable()
def slice_sagittal_gate(x):
    return x[:, 1:2]

@register_keras_serializable()
def reduce_mean_spatial(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable()
def reduce_max_spatial(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)


# =============================================================================
# MODEL ARCHITECTURE (mirror of training script, for float32 rebuild)
# =============================================================================

def _first_branch(name, input_shape, weight_decay):
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


def _cbam_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
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
    avg_pool_spatial = Lambda(reduce_mean_spatial)(x)
    max_pool_spatial = Lambda(reduce_max_spatial)(x)
    spatial_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                               kernel_regularizer=l2(1e-4))(spatial_features)
    output = Multiply()([x, spatial_attention])
    return output


def _gated_fusion_layer(coronal_features, sagittal_features):
    coronal_gap  = GlobalAveragePooling2D()(coronal_features)
    sagittal_gap = GlobalAveragePooling2D()(sagittal_features)
    combined_gap = Concatenate()([coronal_gap, sagittal_gap])
    gate_weights = Dense(2, activation='softmax', name='branch_gates')(combined_gap)
    coronal_gate  = Lambda(slice_coronal_gate)(gate_weights)
    sagittal_gate = Lambda(slice_sagittal_gate)(gate_weights)
    coronal_gate  = Reshape((1, 1, 1))(coronal_gate)
    sagittal_gate = Reshape((1, 1, 1))(sagittal_gate)
    coronal_gated  = Multiply()([coronal_features,  coronal_gate])
    sagittal_gated = Multiply()([sagittal_features, sagittal_gate])
    fused_features = Concatenate(axis=2)([coronal_gated, sagittal_gated])
    fused_filters  = coronal_features.shape[-1]
    fused_features = Conv2D(int(fused_filters), (1, 1), activation='relu')(fused_features)
    return fused_features


def build_model_float32(input_shape=(325, 325, 3), num_classes=3):
    """Build ODIN model in pure float32 for Grad-CAM compatibility."""
    weight_decay = 1e-4
    coronal_input,  coronal_features  = _first_branch("coronal",  input_shape, weight_decay)
    sagittal_input, sagittal_features = _first_branch("sagittal", input_shape, weight_decay)
    coronal_features  = _cbam_block(coronal_features,  ratio=16)
    sagittal_features = _cbam_block(sagittal_features, ratio=16)
    merged_features = _gated_fusion_layer(coronal_features, sagittal_features)
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
    x = Dropout(0.6)(x)
    x = Dense(64,  activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = Model(inputs=[coronal_input, sagittal_input], outputs=outputs)
    return model


# =============================================================================
# PUBLICATION-QUALITY CONFIGURATION
# =============================================================================

MANUSCRIPT_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'image.cmap': 'gray',
}
plt.rcParams.update(MANUSCRIPT_RC)

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

# ODIN color palette
COLORS = {
    'healthy': '#2166AC',  # Blue
    'lesion': '#B2182B',   # Red
    'mrc': '#1B7837',      # Green
    'correct': '#2E7D32',  # Dark green
    'incorrect': '#C62828', # Dark red
    'neutral': '#424242',  # Dark gray
}


# =============================================================================
# DATA LOADING
# =============================================================================

class PostAnalysisSequence(Sequence):
    """Load test data from saved pairs."""
    
    def __init__(self, pairs, class_names, data_dir, target_size=(325, 325), batch_size=16):
        self.pairs = pairs
        self.class_names = class_names
        self.target_size = target_size
        self.batch_size = batch_size
        self.lb = LabelBinarizer()
        self.lb.fit(class_names)
        
        # Test first valid path
        for c, s, _ in pairs[:10]:
            test_path = c if c and c != 'BLANK' else s
            if test_path and test_path != 'BLANK':
                exists = Path(test_path).exists()
                print(f"  Test path: {test_path}")
                print(f"  Exists: {exists}")
                if not exists:
                    print(f"  WARNING: Image paths not found! Check working directory.")
                break
    
    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        Xc_batch, Xs_batch, y_batch = [], [], []
        
        for coronal_path, sagittal_path, cls in batch_pairs:
            # Load coronal - paths are stored as working relative paths
            if coronal_path and coronal_path != 'BLANK':
                img_path = Path(coronal_path)
                if img_path.exists():
                    img_c = Image.open(img_path).convert('RGB')
                    img_c = img_c.resize(self.target_size)
                    img_c = np.array(img_c) / 255.0
                else:
                    img_c = np.zeros((*self.target_size, 3))
            else:
                img_c = np.zeros((*self.target_size, 3))
            
            # Load sagittal
            if sagittal_path and sagittal_path != 'BLANK':
                img_path = Path(sagittal_path)
                if img_path.exists():
                    img_s = Image.open(img_path).convert('RGB')
                    img_s = img_s.resize(self.target_size)
                    img_s = np.array(img_s) / 255.0
                else:
                    img_s = np.zeros((*self.target_size, 3))
            else:
                img_s = np.zeros((*self.target_size, 3))
            
            Xc_batch.append(img_c)
            Xs_batch.append(img_s)
            y_batch.append(self.lb.transform([cls])[0])
        
        return (np.array(Xc_batch), np.array(Xs_batch)), np.array(y_batch)


class ArtifactLoader:
    """Load saved training artifacts."""
    
    def __init__(self, run_name, logs_dir="logs"):
        self.run_name = run_name
        self.run_dir = Path(logs_dir) / run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts not found: {self.artifacts_dir}")
        
        print(f"Loading artifacts from: {self.artifacts_dir}")
        self._load_all()
    
    def _load_all(self):
        # Test sequence config
        with open(self.artifacts_dir / "test_seq_config.json") as f:
            self.config = json.load(f)
        
        # Test pairs
        with open(self.artifacts_dir / "test_pairs.pkl", 'rb') as f:
            data = pickle.load(f)
            self.test_pairs = data['pairs']
            self.class_names = data['class_names']
        
        # Predictions
        with open(self.artifacts_dir / "test_predictions.pkl", 'rb') as f:
            self.predictions = pickle.load(f)
        
        # Training history
        history_path = self.artifacts_dir / "training_history.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = None
        
        print(f"  ✓ Loaded {len(self.test_pairs)} test pairs")
        print(f"  ✓ Classes: {self.class_names}")
    
    def load_model(self, model_type='composite'):
        model_path = self.run_dir / f"best_model_{model_type}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        print(f"  ✓ Loaded ({len(model.layers)} layers)")
        return model

    def load_model_float32(self, model_type='composite'):
        """
        Load model weights into a float32 architecture.
        This enables proper Grad-CAM gradient flow (mixed_float16 blocks gradients).
        """
        model_path = self.run_dir / f"best_model_{model_type}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading float32 model from: {model_path}")
        
        # Step 1: Load the saved mixed_float16 model to get weights
        original_model = load_model(model_path)
        
        # Step 2: Build a fresh float32 model with identical architecture
        # Disable mixed precision policy
        tf.keras.mixed_precision.set_global_policy('float32')
        float32_model = build_model_float32()
        
        # Step 3: Transfer weights layer by layer
        transferred = 0
        skipped = 0
        for orig_layer, new_layer in zip(original_model.layers, float32_model.layers):
            try:
                orig_weights = orig_layer.get_weights()
                if orig_weights:
                    # Convert weights to float32
                    float32_weights = [w.astype(np.float32) for w in orig_weights]
                    new_layer.set_weights(float32_weights)
                    transferred += 1
            except Exception as e:
                skipped += 1
        
        print(f"  ✓ Float32 model built ({len(float32_model.layers)} layers)")
        print(f"  ✓ Weights transferred: {transferred} layers, skipped: {skipped}")
        print(f"  ✓ All computations now in float32 → Grad-CAM will work!")
        
        # Keep the global policy at float32 — the entire ODIN pipeline now
        # runs in pure float32 (mixed_float16 was disabled in training because
        # it blocks Grad-CAM gradient flow). Restoring mixed_float16 here
        # would silently re-break gradients for everything that follows.
        tf.keras.mixed_precision.set_global_policy('float32')
        
        return float32_model
    
    def create_test_sequence(self, batch_size=16, data_dir_override=None):
        pairs = []
        for c, s, cls in self.test_pairs:
            c = None if c == 'BLANK' else c
            s = None if s == 'BLANK' else s
            pairs.append((c, s, cls))
        
        # Use override if provided, otherwise use config
        data_dir = data_dir_override or self.config.get('data_dir', '.')
        
        return PostAnalysisSequence(
            pairs=pairs,
            class_names=self.class_names,
            data_dir=data_dir,
            target_size=tuple(self.config.get('target_size', (325, 325))),
            batch_size=batch_size
        )
    
    def get_misclassified_indices(self):
        y_true = np.array(self.predictions['y_true'])
        y_pred = np.array(self.predictions['y_pred'])
        return np.where(y_true != y_pred)[0]
    
    def get_correct_indices(self):
        y_true = np.array(self.predictions['y_true'])
        y_pred = np.array(self.predictions['y_pred'])
        return np.where(y_true == y_pred)[0]
    
    def get_class_indices(self, class_name, correct_only=True):
        class_idx = self.class_names.index(class_name)
        y_true = np.array(self.predictions['y_true'])
        y_pred = np.array(self.predictions['y_pred'])
        
        mask = y_true == class_idx
        if correct_only:
            mask &= (y_true == y_pred)
        return np.where(mask)[0]


# =============================================================================
# Visualization: Post-CBAM activation magnitude per branch
# =============================================================================

# Global layer choice — set by main()
DEFAULT_LAYER = 'per_branch'
LAYER_CHOICE = DEFAULT_LAYER


def is_fused_mode():
    """True when the chosen layer is after gated fusion (single combined attention)."""
    return LAYER_CHOICE in ('post_fusion', 'last_conv')


def find_gradcam_layers(model):
    """
    Choose layers based on LAYER_CHOICE:
      per_branch: multiply_1/3  → separate per-view CBAM attention
      post_fusion: conv2d_8     → after gated fusion (fused view)
      last_conv: conv2d_10      → deepest, most class-specific (fused view)
    """
    layer_names = [l.name for l in model.layers]
    
    if LAYER_CHOICE == 'per_branch':
        print("\n  Mode: PER-BRANCH (CBAM outputs) — separate attention per view")
        coronal_layer  = 'multiply_1' if 'multiply_1' in layer_names else 'conv2d_2'
        sagittal_layer = 'multiply_3' if 'multiply_3' in layer_names else 'conv2d_5'
    elif LAYER_CHOICE == 'post_fusion':
        print("\n  Mode: POST-FUSION (conv2d_8) — what survives gating")
        coronal_layer = sagittal_layer = 'conv2d_8'
    elif LAYER_CHOICE == 'last_conv':
        print("\n  Mode: LAST CONV (conv2d_10) — deepest, most class-specific")
        coronal_layer = sagittal_layer = 'conv2d_10'
    else:
        coronal_layer  = 'multiply_1'
        sagittal_layer = 'multiply_3'
    
    print(f"    Coronal:  {coronal_layer} → {model.get_layer(coronal_layer).output.shape}")
    print(f"    Sagittal: {sagittal_layer} → {model.get_layer(sagittal_layer).output.shape}")
    
    return coronal_layer, sagittal_layer


def compute_gradcam(model, img_coronal, img_sagittal, class_idx, layer_name, debug=False):
    """
    TRUE gradient-based Grad-CAM (Selvaraju et al., 2017).

    Now that the model runs in pure float32 (mixed_float16 blocked gradient
    flow through the Lambda/CBAM/gating layers), real Grad-CAM works:

      1. Forward pass to get target-layer feature maps A^k and class logit y^c.
      2. Gradient of the class logit w.r.t. the feature maps:  ∂y^c/∂A^k.
      3. Global-average-pool the gradients → channel importance α_k^c.
      4. ReLU(Σ_k α_k^c · A^k) → class-discriminative localization map.

    IMPORTANT: gradients are taken w.r.t. the PRE-SOFTMAX logit, not the
    softmax probability. This model is highly confident (many predictions at
    p≈1.0); differentiating the saturated softmax would give ~zero gradients
    and blank maps. The logit is reconstructed as penultimate · W + b from the
    final Dense layer's weights so the gradient signal stays informative.

    The map is CLASS-SPECIFIC: pass class_idx to see why the model predicted
    that class (the panels pass the predicted class).
    """
    if layer_name is None:
        return np.zeros((81, 81))
    try:
        # Reconstruct pre-softmax logits from the final Dense layer.
        final_dense = model.layers[-1]
        dense_w, dense_b = final_dense.get_weights()
        dense_w = tf.constant(dense_w, dtype=tf.float32)
        dense_b = tf.constant(dense_b, dtype=tf.float32)

        # Model that emits (target feature maps, penultimate features) in one pass.
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.layers[-2].output]
        )

        img_c = tf.constant(img_coronal, dtype=tf.float32)
        img_s = tf.constant(img_sagittal, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_output, penult = grad_model([img_c, img_s], training=False)
            tape.watch(conv_output)
            logits = tf.matmul(penult, dense_w) + dense_b      # (1, num_classes)
            ci = int(tf.argmax(logits[0])) if class_idx is None else int(class_idx)
            class_score = logits[:, ci]

        grads = tape.gradient(class_score, conv_output)         # (1, H, W, C)
        if grads is None:
            raise ValueError("Gradient is None — gradient flow broken "
                             "(is the model float32?)")

        # Channel importance weights = GAP of gradients
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))         # (C,)
        cam = tf.reduce_sum(conv_output[0] * weights, axis=-1)  # (H, W)
        cam = tf.nn.relu(cam)                                   # keep positive evidence
        heatmap = cam.numpy().astype(np.float32)

        if debug:
            H, W = heatmap.shape
            is_fused = W >= 1.5 * H
            print(f"    DEBUG [{layer_name}] class={ci}: feat={conv_output.shape[1:]}, "
                  f"fused={is_fused}, grad|max|={float(tf.reduce_max(tf.abs(grads))):.2e}")
            print(f"    DEBUG: cam min={heatmap.min():.4f} max={heatmap.max():.4f} "
                  f"mean={heatmap.mean():.4f}")

        # Normalize to 0-1
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap.astype(np.float32)

    except Exception as e:
        print(f"    ERROR [{layer_name}]: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((81, 81))


def compute_gradcam_split(model, img_coronal, img_sagittal, class_idx,
                          layer_name, debug=False):
    """
    For per-branch layers (multiply_1/3): heatmap is for ONE view directly.
    For post-fusion layers (conv2d_8/9/10): the real Grad-CAM map spans the
      spatially-concatenated fused tensor (H, W*2). Split it:
        Left half  = coronal contribution
        Right half = sagittal contribution
    Each half is re-normalized independently for per-view contrast.
    """
    heatmap = compute_gradcam(model, img_coronal, img_sagittal,
                              class_idx, layer_name, debug=debug)
    
    if heatmap.size == 0:
        return heatmap, heatmap
    
    # If width > height by ~2x, this is a fused (spatially concatenated) layer
    H, W = heatmap.shape
    if W >= 1.5 * H:
        mid = W // 2
        coronal_heatmap  = heatmap[:, :mid]
        sagittal_heatmap = heatmap[:, mid:]
        
        # Re-normalize each half independently for better per-view contrast
        for hm in [coronal_heatmap, sagittal_heatmap]:
            if hm.max() > hm.min():
                hm[:] = (hm - hm.min()) / (hm.max() - hm.min())
        
        if debug:
            print(f"    DEBUG: Split fused ({H}×{W}) → coronal {coronal_heatmap.shape}, "
                  f"sagittal {sagittal_heatmap.shape}")
        
        return coronal_heatmap, sagittal_heatmap
    
    # Per-branch layer — same heatmap for both
    return heatmap, heatmap


def compute_cbam_attention(model, img_coronal, img_sagittal, layer_name, debug=False):
    return compute_gradcam(model, img_coronal, img_sagittal, 0, layer_name, debug)


def compute_per_view_attention(model, Xc, Xs, coronal_layer, sagittal_layer,
                                class_idx=0, debug=False):
    """
    Returns (coronal_heatmap, sagittal_heatmap).
    
    If coronal_layer == sagittal_layer (fused layer like conv2d_8/10):
      Compute ONE heatmap, split into halves.
    Else (per-branch like multiply_1 vs multiply_3):
      Extract each layer separately for its own view.
    """
    if coronal_layer == sagittal_layer:
        # Fused layer — split spatially
        return compute_gradcam_split(model, Xc, Xs, class_idx, coronal_layer, debug=debug)
    else:
        # Per-branch — extract each separately
        coronal_hm  = compute_gradcam(model, Xc, Xs, class_idx, coronal_layer, debug=debug)
        sagittal_hm = compute_gradcam(model, Xc, Xs, class_idx, sagittal_layer, debug=debug)
        return coronal_hm, sagittal_hm


def compute_fused_attention(model, Xc, Xs, layer_name, class_idx=0, debug=False):
    """
    For fused layers, return ONE combined heatmap (full H × W including both branches).
    This represents the model's unified post-fusion attention used for the final decision.
    """
    return compute_gradcam(model, Xc, Xs, class_idx, layer_name, debug=debug)


def overlay_heatmap_fused(Xc, Xs, heatmap_fused, alpha=0.5):
    """
    For fused layers: create a side-by-side image of [coronal | sagittal]
    with the fused heatmap (W = 2H shape) overlaid across the full width.
    Returns a single combined image.
    """
    # Stack coronal and sagittal horizontally
    Xc_img = Xc if Xc is not None else np.ones((325, 325, 3)) * 0.9
    Xs_img = Xs if Xs is not None else np.ones((325, 325, 3)) * 0.9
    
    # Ensure same height
    h = max(Xc_img.shape[0], Xs_img.shape[0])
    if Xc_img.shape[0] != h:
        Xc_img = cv2.resize(Xc_img, (Xc_img.shape[1], h))
    if Xs_img.shape[0] != h:
        Xs_img = cv2.resize(Xs_img, (Xs_img.shape[1], h))
    
    # Side-by-side concatenation (matches the architecture's spatial concat)
    combined_img = np.concatenate([Xc_img, Xs_img], axis=1)
    
    # Overlay the fused heatmap across the full combined image
    overlay, _ = overlay_heatmap(combined_img, heatmap_fused, alpha=alpha)
    return overlay


def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlay heatmap with Inferno colormap, masking black borders."""
    # Ensure img is uint8
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    img_uint8 = np.ascontiguousarray(img_uint8)
    heatmap = np.ascontiguousarray(heatmap.astype(np.float32))
    
    # Create mask from image - where actual content is (not black borders)
    img_gray = np.mean(img, axis=-1) if len(img.shape) == 3 else img
    content_mask = (img_gray > 0.05).astype(np.float32)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
    
    # Resize mask
    content_mask_resized = cv2.resize(content_mask.astype(np.float32),
                                       (img_uint8.shape[1], img_uint8.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
    
    # Apply mask - zero out attention outside content area
    heatmap_masked = heatmap_resized * content_mask_resized
    
    # Percentile normalization on content region for better contrast
    content_values = heatmap_masked[content_mask_resized > 0]
    if len(content_values) > 0 and content_values.max() > content_values.min():
        p_low = np.percentile(content_values, 5)
        p_high = np.percentile(content_values, 95)
        heatmap_norm = np.clip((heatmap_masked - p_low) / (p_high - p_low + 1e-8), 0, 1)
        heatmap_norm = heatmap_norm * content_mask_resized
    else:
        heatmap_norm = np.zeros_like(heatmap_masked)
    
    # Apply Inferno colormap (perceptually uniform, journal-prestige)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Fixed alpha blending only within content mask
    mask_3ch = np.stack([content_mask_resized] * 3, axis=-1)
    overlay = (img_uint8.astype(np.float32) * (1 - alpha * mask_3ch) +
               heatmap_colored.astype(np.float32) * (alpha * mask_3ch)).astype(np.uint8)
    
    return overlay, heatmap_norm


# =============================================================================
# PUBLICATION-QUALITY FIGURES
# =============================================================================

def save_figure(fig, output_dir, stem):
    """Save in PDF, PNG, SVG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(output_dir / f"{stem}.{fmt}", format=fmt, dpi=300, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  ✓ Saved {stem}.pdf/png/svg")


def generate_paper_gradcam_panel(loader, model, test_seq, output_dir):
    """
    Generate publication-quality Grad-CAM panel.
    Layout: 3 rows (one per class) × 4 columns (CR, SG, CR-CAM, SG-CAM)
    """
    print("\n" + "=" * 60)
    print("Generating Paper-Quality Grad-CAM Panel")
    print("=" * 60)
    
    coronal_layer, sagittal_layer = find_gradcam_layers(model)
    
    # Select best examples: one correct per class with both views
    selected = []
    for class_name in loader.class_names:
        indices = loader.get_class_indices(class_name, correct_only=True)
        
        for idx in indices:
            c_path, s_path, _ = loader.test_pairs[idx]
            if c_path != 'BLANK' and s_path != 'BLANK':
                selected.append((idx, class_name))
                break
        else:
            if len(indices) > 0:
                selected.append((indices[0], class_name))
    
    n_rows = len(selected)
    fused_mode = is_fused_mode()
    
    if fused_mode:
        # 3 cols: Coronal | Sagittal | Fused Attention (wider, spans both views)
        fig = plt.figure(figsize=(12, 2.8 * n_rows))
        gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 1, 2, 0.05],
                               wspace=0.08, hspace=0.15)
        attention_titles = ["Coronal View", "Sagittal View", "Fused Attention (Coronal | Sagittal)"]
    else:
        # 4 cols: Coronal | Sagittal | Coronal Attention | Sagittal Attention
        fig = plt.figure(figsize=(10, 2.8 * n_rows))
        gs = gridspec.GridSpec(n_rows, 5, width_ratios=[1, 1, 1, 1, 0.05],
                               wspace=0.08, hspace=0.15)
        attention_titles = ["Coronal View", "Sagittal View",
                            "Coronal Attention", "Sagittal Attention"]
    
    for row, (idx, class_name) in enumerate(selected):
        batch_idx = idx // test_seq.batch_size
        in_batch_idx = idx % test_seq.batch_size
        (Xc_batch, Xs_batch), _ = test_seq[batch_idx]
        
        Xc = Xc_batch[in_batch_idx:in_batch_idx+1]
        Xs = Xs_batch[in_batch_idx:in_batch_idx+1]
        pred_class = loader.predictions['y_pred'][idx]
        
        is_coronal_blank = np.allclose(Xc[0], 0)
        is_sagittal_blank = np.allclose(Xs[0], 0)
        
        # Coronal input
        ax0 = fig.add_subplot(gs[row, 0])
        if not is_coronal_blank:
            ax0.imshow(Xc[0])
        else:
            ax0.imshow(np.ones_like(Xc[0]) * 0.9)
            ax0.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                    transform=ax0.transAxes, color='gray')
        ax0.axis('off')
        if row == 0:
            ax0.set_title(attention_titles[0], fontsize=11, fontweight='bold', pad=8)
        
        ax0.text(-0.15, 0.5, _disp(class_name), transform=ax0.transAxes,
                fontsize=12, fontweight='bold', va='center', ha='right',
                color=COLORS.get(class_name, COLORS['neutral']))
        
        # Sagittal input
        ax1 = fig.add_subplot(gs[row, 1])
        if not is_sagittal_blank:
            ax1.imshow(Xs[0])
        else:
            ax1.imshow(np.ones_like(Xs[0]) * 0.9)
            ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                    transform=ax1.transAxes, color='gray')
        ax1.axis('off')
        if row == 0:
            ax1.set_title(attention_titles[1], fontsize=11, fontweight='bold', pad=8)
        
        debug = (row == 0)
        
        if fused_mode:
            # ONE attention column spanning both views
            ax2 = fig.add_subplot(gs[row, 2])
            try:
                fused_hm = compute_fused_attention(model, Xc, Xs,
                                                    coronal_layer, pred_class, debug=debug)
                # Use either view as background (or both if both available)
                cor = None if is_coronal_blank else Xc[0]
                sag = None if is_sagittal_blank else Xs[0]
                if cor is None and sag is None:
                    ax2.imshow(np.ones((325, 650, 3)) * 0.9)
                else:
                    overlay = overlay_heatmap_fused(cor, sag, fused_hm, alpha=0.5)
                    ax2.imshow(overlay)
            except Exception as e:
                print(f"    Warning (fused): {e}")
                ax2.imshow(np.ones((325, 650, 3)) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(attention_titles[2], fontsize=11, fontweight='bold', pad=8)
            colorbar_col = 3
        else:
            # Two attention columns (per-branch mode)
            try:
                ch, sh = compute_per_view_attention(model, Xc, Xs,
                                                     coronal_layer, sagittal_layer,
                                                     class_idx=pred_class, debug=debug)
                coronal_heatmap  = ch if not is_coronal_blank else None
                sagittal_heatmap = sh if not is_sagittal_blank else None
            except Exception as e:
                print(f"    Warning: {e}")
                coronal_heatmap = sagittal_heatmap = None
            
            ax2 = fig.add_subplot(gs[row, 2])
            if not is_coronal_blank and coronal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xc[0], coronal_heatmap, alpha=0.5)
                    ax2.imshow(overlay)
                except Exception:
                    ax2.imshow(Xc[0])
            else:
                ax2.imshow(np.ones_like(Xc[0]) * 0.9)
                ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                        transform=ax2.transAxes, color='gray')
            ax2.axis('off')
            if row == 0:
                ax2.set_title(attention_titles[2], fontsize=11, fontweight='bold', pad=8)
            
            ax3 = fig.add_subplot(gs[row, 3])
            if not is_sagittal_blank and sagittal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xs[0], sagittal_heatmap, alpha=0.5)
                    ax3.imshow(overlay)
                except Exception:
                    ax3.imshow(Xs[0])
            else:
                ax3.imshow(np.ones_like(Xs[0]) * 0.9)
                ax3.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                        transform=ax3.transAxes, color='gray')
            ax3.axis('off')
            if row == 0:
                ax3.set_title(attention_titles[3], fontsize=11, fontweight='bold', pad=8)
            colorbar_col = 4
    
    # Colorbar
    cax = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Attention\nIntensity', fontsize=9, labelpad=5)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Med', 'High'])
    
    save_figure(fig, output_dir, "gradcam_panel_correct")
    plt.close(fig)


def generate_misclassified_gradcam(loader, model, test_seq, output_dir):
    """Generate detailed Grad-CAM for misclassified samples."""
    print("\nGenerating Misclassified Sample Analysis...")
    
    coronal_layer, sagittal_layer = find_gradcam_layers(model)
    misclassified = loader.get_misclassified_indices()
    fused_mode = is_fused_mode()
    
    if len(misclassified) == 0:
        print("  ✓ No misclassified samples — perfect accuracy!")
        return
    
    print(f"  Found {len(misclassified)} misclassified samples")

    # Select up to 4 cases, balanced across available views:
    # 2 with coronal present and 2 with sagittal present (all errors here are
    # single-view), filling from leftovers if one view is under-represented.
    def _pick_balanced(indices, target=4):
        cor_only, sag_only, both = [], [], []
        for i in indices:
            c, s, _ = loader.test_pairs[i]
            cb, sb = (c == 'BLANK'), (s == 'BLANK')
            if not cb and sb:
                cor_only.append(i)
            elif cb and not sb:
                sag_only.append(i)
            elif not cb and not sb:
                both.append(i)
        half = target // 2
        sel = cor_only[:half] + sag_only[:half]
        if len(sel) < target:                      # backfill if a view is scarce
            for i in cor_only[half:] + sag_only[half:] + both:
                if i not in sel and len(sel) < target:
                    sel.append(i)
        return sel[:target]

    selected = _pick_balanced(misclassified, target=4)
    n_samples = len(selected)
    n_cor = sum(1 for i in selected if loader.test_pairs[i][0] != 'BLANK')
    n_sag = sum(1 for i in selected if loader.test_pairs[i][1] != 'BLANK')
    print(f"  Showing {n_samples} of {len(misclassified)} "
          f"({n_cor} coronal-present, {n_sag} sagittal-present)")
    
    if fused_mode:
        # Two attention columns: one explaining the TRUE class, one the PREDICTED
        # class — so the reader sees what supports the correct diagnosis vs. what
        # misled the model.
        fig = plt.figure(figsize=(18, 2.7 * n_samples + 0.6))
        gs = gridspec.GridSpec(n_samples, 5, width_ratios=[1, 1, 2, 2, 0.05],
                               wspace=0.08, hspace=0.30)
        col_titles = ["Coronal View", "Sagittal View",
                      "Attention for TRUE class\n(what supports correct diagnosis)",
                      "Attention for PREDICTED class\n(what the model relied on)"]
        colorbar_col = 4
    else:
        fig = plt.figure(figsize=(12, 3.2 * n_samples))
        gs = gridspec.GridSpec(n_samples, 5, width_ratios=[1, 1, 1, 1, 0.05],
                               wspace=0.08, hspace=0.2)
        col_titles = ["Coronal View", "Sagittal View",
                      "Coronal Attention", "Sagittal Attention"]
        colorbar_col = 4
    
    for row, idx in enumerate(selected):
        batch_idx = idx // test_seq.batch_size
        in_batch_idx = idx % test_seq.batch_size
        (Xc_batch, Xs_batch), _ = test_seq[batch_idx]
        
        Xc = Xc_batch[in_batch_idx:in_batch_idx+1]
        Xs = Xs_batch[in_batch_idx:in_batch_idx+1]
        
        true_label = loader.class_names[loader.predictions['y_true'][idx]]
        pred_label = loader.class_names[loader.predictions['y_pred'][idx]]
        pred_class = loader.predictions['y_pred'][idx]
        probs = loader.predictions['y_pred_proba'][idx]
        
        is_coronal_blank = np.allclose(Xc[0], 0)
        is_sagittal_blank = np.allclose(Xs[0], 0)
        
        # Coronal
        ax0 = fig.add_subplot(gs[row, 0])
        if not is_coronal_blank:
            ax0.imshow(Xc[0])
        else:
            ax0.imshow(np.ones_like(Xc[0]) * 0.9)
            ax0.text(0.5, 0.5, 'BLANK', ha='center', va='center', fontsize=11,
                    transform=ax0.transAxes, color='#666', fontweight='bold')
        ax0.axis('off')
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=8)
        
        label_text = f"True: {_disp(true_label)}\n→ Pred: {_disp(pred_label)}"
        ax0.text(-0.18, 0.5, label_text, transform=ax0.transAxes,
                fontsize=10, fontweight='bold', va='center', ha='right',
                color=COLORS['incorrect'], linespacing=1.4)
        
        # Sagittal
        ax1 = fig.add_subplot(gs[row, 1])
        if not is_sagittal_blank:
            ax1.imshow(Xs[0])
        else:
            ax1.imshow(np.ones_like(Xs[0]) * 0.9)
            ax1.text(0.5, 0.5, 'BLANK', ha='center', va='center', fontsize=11,
                    transform=ax1.transAxes, color='#666', fontweight='bold')
        ax1.axis('off')
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=8)
        
        if fused_mode:
            true_class = loader.predictions['y_true'][idx]
            cor = None if is_coronal_blank else Xc[0]
            sag = None if is_sagittal_blank else Xs[0]

            # Column 2 — attention for the TRUE class
            ax2 = fig.add_subplot(gs[row, 2])
            try:
                hm_true = compute_fused_attention(model, Xc, Xs,
                                                  coronal_layer, true_class)
                if cor is None and sag is None:
                    ax2.imshow(np.ones((325, 650, 3)) * 0.9)
                else:
                    ax2.imshow(overlay_heatmap_fused(cor, sag, hm_true, alpha=0.5))
            except Exception:
                ax2.imshow(np.ones((325, 650, 3)) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(col_titles[2], fontsize=10, fontweight='bold', pad=10)
            # Per-row class label as a caption BELOW the panel (green = correct)
            ax2.text(0.5, -0.04, f"{_disp(true_label)}  (P={probs[true_class]:.2f})",
                     transform=ax2.transAxes, fontsize=9, fontweight='bold',
                     va='top', ha='center', color=COLORS['correct'])

            # Column 3 — attention for the PREDICTED (wrong) class
            ax3 = fig.add_subplot(gs[row, 3])
            try:
                hm_pred = compute_fused_attention(model, Xc, Xs,
                                                  coronal_layer, pred_class)
                if cor is None and sag is None:
                    ax3.imshow(np.ones((325, 650, 3)) * 0.9)
                else:
                    ax3.imshow(overlay_heatmap_fused(cor, sag, hm_pred, alpha=0.5))
            except Exception:
                ax3.imshow(np.ones((325, 650, 3)) * 0.9)
            ax3.axis('off')
            if row == 0:
                ax3.set_title(col_titles[3], fontsize=10, fontweight='bold', pad=10)
            # Per-row class label as a caption BELOW the panel (red = model's pick)
            ax3.text(0.5, -0.04, f"{_disp(pred_label)}  (P={probs[pred_class]:.2f})",
                     transform=ax3.transAxes, fontsize=9, fontweight='bold',
                     va='top', ha='center', color=COLORS['incorrect'])
        else:
            # Per-branch attention (2 columns)
            coronal_heatmap, sagittal_heatmap = None, None
            try:
                ch, sh = compute_per_view_attention(model, Xc, Xs,
                                                     coronal_layer, sagittal_layer,
                                                     class_idx=pred_class)
                if not is_coronal_blank:  coronal_heatmap  = ch
                if not is_sagittal_blank: sagittal_heatmap = sh
            except Exception:
                pass
            
            ax2 = fig.add_subplot(gs[row, 2])
            if not is_coronal_blank and coronal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xc[0], coronal_heatmap, alpha=0.5)
                    ax2.imshow(overlay)
                except:
                    ax2.imshow(Xc[0])
            else:
                ax2.imshow(np.ones_like(Xc[0]) * 0.9)
                ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                        transform=ax2.transAxes, color='gray')
            ax2.axis('off')
            if row == 0:
                ax2.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=8)
            
            ax3 = fig.add_subplot(gs[row, 3])
            if not is_sagittal_blank and sagittal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xs[0], sagittal_heatmap, alpha=0.5)
                    ax3.imshow(overlay)
                except:
                    ax3.imshow(Xs[0])
            else:
                ax3.imshow(np.ones_like(Xs[0]) * 0.9)
                ax3.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                        transform=ax3.transAxes, color='gray')
            ax3.axis('off')
            if row == 0:
                ax3.set_title(col_titles[3], fontsize=11, fontweight='bold', pad=8)
            
            prob_text = f"P({pred_label})={probs[pred_class]:.2f}"
            ax3.text(1.02, 0.5, prob_text, transform=ax3.transAxes,
                    fontsize=8, va='center', ha='left', color='#666')
    
    # Colorbar
    cax = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Attention', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Med', 'High'])
    
    fig.suptitle(
        f"Misclassified Samples — {n_samples} of {len(misclassified)} shown "
        f"({n_cor} coronal-present, {n_sag} sagittal-present)\n"
        f"left attention = TRUE class (green),  right = PREDICTED class (red)",
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.06, right=0.97)
    
    save_figure(fig, output_dir, "gradcam_misclassified")
    plt.close(fig)
    
    # Individual figures
    indiv_dir = Path(output_dir) / "gradcam_individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(misclassified):
        batch_idx = idx // test_seq.batch_size
        in_batch_idx = idx % test_seq.batch_size
        (Xc_batch, Xs_batch), _ = test_seq[batch_idx]
        
        Xc = Xc_batch[in_batch_idx:in_batch_idx+1]
        Xs = Xs_batch[in_batch_idx:in_batch_idx+1]
        
        true_label = loader.class_names[loader.predictions['y_true'][idx]]
        pred_label = loader.class_names[loader.predictions['y_pred'][idx]]
        pred_class = loader.predictions['y_pred'][idx]
        probs = loader.predictions['y_pred_proba'][idx]

        # Identify the patient/file for this misclassified pair.
        # Pairs are (coronal_path, sagittal_path, cls); one view may be BLANK.
        c_path, s_path, _ = loader.test_pairs[idx]
        c_name = os.path.basename(c_path) if c_path and c_path != 'BLANK' else None
        s_name = os.path.basename(s_path) if s_path and s_path != 'BLANK' else None
        sample_name = c_name or s_name or f"sample_{idx}"
        file_stem = os.path.splitext(sample_name)[0]
        # Sanitize for use in a filename (no spaces/slashes)
        safe_stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "_"
                            for ch in file_stem)
        
        is_coronal_blank = np.allclose(Xc[0], 0)
        is_sagittal_blank = np.allclose(Xs[0], 0)
        
        if fused_mode:
            # 1 row, 3 cols layout: Coronal | Sagittal | Fused Attention (wide)
            fig = plt.figure(figsize=(14, 4))
            gs_i = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2], wspace=0.1)
            
            axc = fig.add_subplot(gs_i[0])
            axc.imshow(Xc[0] if not is_coronal_blank else np.ones_like(Xc[0]) * 0.9)
            axc.set_title("Coronal" + (" (BLANK)" if is_coronal_blank else ""), fontsize=11)
            axc.axis('off')
            
            axs = fig.add_subplot(gs_i[1])
            axs.imshow(Xs[0] if not is_sagittal_blank else np.ones_like(Xs[0]) * 0.9)
            axs.set_title("Sagittal" + (" (BLANK)" if is_sagittal_blank else ""), fontsize=11)
            axs.axis('off')
            
            axa = fig.add_subplot(gs_i[2])
            try:
                fused_hm = compute_fused_attention(model, Xc, Xs,
                                                    coronal_layer, pred_class)
                cor = None if is_coronal_blank else Xc[0]
                sag = None if is_sagittal_blank else Xs[0]
                if cor is None and sag is None:
                    axa.imshow(np.ones((325, 650, 3)) * 0.9)
                else:
                    overlay = overlay_heatmap_fused(cor, sag, fused_hm, alpha=0.5)
                    axa.imshow(overlay)
            except Exception:
                axa.imshow(np.ones((325, 650, 3)) * 0.9)
            axa.set_title("Fused Attention (Coronal | Sagittal)", fontsize=11)
            axa.axis('off')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            
            axes[0, 0].imshow(Xc[0] if not is_coronal_blank else np.ones_like(Xc[0]) * 0.9)
            axes[0, 0].set_title("Coronal" + (" (BLANK)" if is_coronal_blank else ""), fontsize=11)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(Xs[0] if not is_sagittal_blank else np.ones_like(Xs[0]) * 0.9)
            axes[0, 1].set_title("Sagittal" + (" (BLANK)" if is_sagittal_blank else ""), fontsize=11)
            axes[0, 1].axis('off')
            
            coronal_heatmap, sagittal_heatmap = None, None
            try:
                ch, sh = compute_per_view_attention(model, Xc, Xs,
                                                     coronal_layer, sagittal_layer,
                                                     class_idx=pred_class)
                if not is_coronal_blank:  coronal_heatmap  = ch
                if not is_sagittal_blank: sagittal_heatmap = sh
            except Exception:
                pass
            
            if not is_coronal_blank and coronal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xc[0], coronal_heatmap, alpha=0.5)
                    axes[1, 0].imshow(overlay)
                except:
                    axes[1, 0].imshow(Xc[0])
            else:
                axes[1, 0].imshow(np.ones_like(Xc[0]) * 0.9)
            axes[1, 0].set_title("Coronal Attention", fontsize=11)
            axes[1, 0].axis('off')
            
            if not is_sagittal_blank and sagittal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xs[0], sagittal_heatmap, alpha=0.5)
                    axes[1, 1].imshow(overlay)
                except:
                    axes[1, 1].imshow(Xs[0])
            else:
                axes[1, 1].imshow(np.ones_like(Xs[0]) * 0.9)
            axes[1, 1].set_title("Sagittal Attention", fontsize=11)
            axes[1, 1].axis('off')
        
        prob_str = " | ".join([f"{_disp(c)}: {p:.2f}" for c, p in zip(loader.class_names, probs)])
        fig.suptitle(f"True: {_disp(true_label)} → Pred: {_disp(pred_label)}\n{prob_str}",
                    fontsize=12, fontweight='bold', color=COLORS['incorrect'])
        
        plt.tight_layout()
        
        for fmt in ['pdf', 'png']:
            fig.savefig(indiv_dir / f"misclassified_{i+1:02d}_{safe_stem}.{fmt}",
                       format=fmt, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  ✓ Saved {len(misclassified)} individual figures")


def generate_combined_panel(loader, model, test_seq, output_dir):
    """Generate combined panel: correct + misclassified."""
    print("\nGenerating Combined Grad-CAM Panel...")
    
    coronal_layer, sagittal_layer = find_gradcam_layers(model)
    fused_mode = is_fused_mode()
    
    samples = []
    
    # One correct per class
    for class_name in loader.class_names:
        indices = loader.get_class_indices(class_name, correct_only=True)
        for idx in indices:
            c_path, s_path, _ = loader.test_pairs[idx]
            if c_path != 'BLANK' and s_path != 'BLANK':
                samples.append((idx, class_name, True))
                break
        else:
            if len(indices) > 0:
                samples.append((indices[0], class_name, True))
    
    # Add misclassified
    for idx in loader.get_misclassified_indices()[:3]:
        true_label = loader.class_names[loader.predictions['y_true'][idx]]
        samples.append((idx, true_label, False))
    
    n_rows = len(samples)
    
    if fused_mode:
        fig = plt.figure(figsize=(13, 2.6 * n_rows))
        gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 1, 2, 0.05],
                               wspace=0.08, hspace=0.18)
        col_titles = ["Coronal", "Sagittal", "Fused Attention (Coronal | Sagittal)"]
        colorbar_col = 3
    else:
        fig = plt.figure(figsize=(11, 2.6 * n_rows))
        gs = gridspec.GridSpec(n_rows, 5, width_ratios=[1, 1, 1, 1, 0.05],
                               wspace=0.08, hspace=0.18)
        col_titles = ["Coronal", "Sagittal", "Coronal Attention", "Sagittal Attention"]
        colorbar_col = 4
    
    for row, (idx, class_name, is_correct) in enumerate(samples):
        batch_idx = idx // test_seq.batch_size
        in_batch_idx = idx % test_seq.batch_size
        (Xc_batch, Xs_batch), _ = test_seq[batch_idx]
        
        Xc = Xc_batch[in_batch_idx:in_batch_idx+1]
        Xs = Xs_batch[in_batch_idx:in_batch_idx+1]
        
        true_label = loader.class_names[loader.predictions['y_true'][idx]]
        pred_label = loader.class_names[loader.predictions['y_pred'][idx]]
        pred_class = loader.predictions['y_pred'][idx]
        
        is_coronal_blank = np.allclose(Xc[0], 0)
        is_sagittal_blank = np.allclose(Xs[0], 0)
        
        # Coronal
        ax0 = fig.add_subplot(gs[row, 0])
        if not is_coronal_blank:
            ax0.imshow(Xc[0])
        else:
            ax0.imshow(np.ones_like(Xc[0]) * 0.9)
            ax0.text(0.5, 0.5, 'BLANK', ha='center', va='center',
                   transform=ax0.transAxes, color='#888', fontsize=10)
        ax0.axis('off')
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=10, fontweight='bold', pad=6)
        
        # Row label
        if is_correct:
            label = f"✓ {_disp(true_label)}"
            color = COLORS['correct']
        else:
            label = f"✗ True: {_disp(true_label)}\n→ Pred: {_disp(pred_label)}"
            color = COLORS['incorrect']
        
        ax0.text(-0.18, 0.5, label, transform=ax0.transAxes,
                fontsize=9, fontweight='bold', va='center', ha='right',
                color=color, linespacing=1.3)
        
        # Sagittal
        ax1 = fig.add_subplot(gs[row, 1])
        if not is_sagittal_blank:
            ax1.imshow(Xs[0])
        else:
            ax1.imshow(np.ones_like(Xs[0]) * 0.9)
            ax1.text(0.5, 0.5, 'BLANK', ha='center', va='center',
                   transform=ax1.transAxes, color='#888', fontsize=10)
        ax1.axis('off')
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=10, fontweight='bold', pad=6)
        
        if fused_mode:
            ax2 = fig.add_subplot(gs[row, 2])
            try:
                fused_hm = compute_fused_attention(model, Xc, Xs,
                                                    coronal_layer, pred_class)
                cor = None if is_coronal_blank else Xc[0]
                sag = None if is_sagittal_blank else Xs[0]
                if cor is None and sag is None:
                    ax2.imshow(np.ones((325, 650, 3)) * 0.9)
                else:
                    overlay = overlay_heatmap_fused(cor, sag, fused_hm, alpha=0.5)
                    ax2.imshow(overlay)
            except Exception:
                ax2.imshow(np.ones((325, 650, 3)) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(col_titles[2], fontsize=10, fontweight='bold', pad=6)
        else:
            coronal_heatmap, sagittal_heatmap = None, None
            try:
                ch, sh = compute_per_view_attention(model, Xc, Xs,
                                                     coronal_layer, sagittal_layer,
                                                     class_idx=pred_class)
                if not is_coronal_blank:  coronal_heatmap  = ch
                if not is_sagittal_blank: sagittal_heatmap = sh
            except Exception:
                pass
            
            ax2 = fig.add_subplot(gs[row, 2])
            if not is_coronal_blank and coronal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xc[0], coronal_heatmap, alpha=0.5)
                    ax2.imshow(overlay)
                except:
                    ax2.imshow(Xc[0])
            else:
                ax2.imshow(np.ones_like(Xc[0]) * 0.9)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(col_titles[2], fontsize=10, fontweight='bold', pad=6)
            
            ax3 = fig.add_subplot(gs[row, 3])
            if not is_sagittal_blank and sagittal_heatmap is not None:
                try:
                    overlay, _ = overlay_heatmap(Xs[0], sagittal_heatmap, alpha=0.5)
                    ax3.imshow(overlay)
                except:
                    ax3.imshow(Xs[0])
            else:
                ax3.imshow(np.ones_like(Xs[0]) * 0.9)
            ax3.axis('off')
            if row == 0:
                ax3.set_title(col_titles[3], fontsize=10, fontweight='bold', pad=6)
    
    # Colorbar
    cax = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Attention', fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    
    fig.suptitle("Grad-CAM Attention Visualization", fontsize=12, fontweight='bold', y=1.0)
    
    save_figure(fig, output_dir, "gradcam_combined_panel")
    plt.close(fig)


def generate_tsne_embedding(loader, model, test_seq, output_dir,
                            perplexity=30, max_iter=1000):
    """
    Option A — regenerate the t-SNE embedding of penultimate (64-D) features
    without retraining. Loads the already-trained model, extracts the 64-unit
    dense embedding for every test pair, projects to 2-D with t-SNE, and colours
    points by true class (marker shape encodes correct vs misclassified).

    Uses max_iter (sklearn ≥1.5) instead of the removed n_iter argument — this
    is the fix for the '⚠ t-SNE failed: unexpected keyword argument n_iter'
    error from the training run.
    """
    from sklearn.manifold import TSNE

    print("\n" + "=" * 60)
    print("Generating t-SNE Embedding of Penultimate Features")
    print("=" * 60)

    # Find the 64-unit penultimate embedding layer robustly.
    embed_layer = None
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if (layer.__class__.__name__ == 'Dense'
                and len(shape) == 2 and shape[-1] == 64):
            embed_layer = layer.name
            break
    if embed_layer is None:
        # Fallback: layer feeding the final Dense (dropout/penultimate)
        embed_layer = model.layers[-2].name
    print(f"    Extracting features from: {embed_layer} "
          f"({model.get_layer(embed_layer).output.shape})")

    feature_model = Model(inputs=model.inputs,
                          outputs=model.get_layer(embed_layer).output)

    # Extract features across the whole test set
    feats = []
    for i in range(len(test_seq)):
        (Xc_batch, Xs_batch), _ = test_seq[i]
        f = feature_model([tf.constant(Xc_batch, dtype=tf.float32),
                           tf.constant(Xs_batch, dtype=tf.float32)],
                          training=False)
        feats.append(np.asarray(f))
    all_features = np.concatenate(feats, axis=0)
    print(f"    Feature shape: {all_features.shape}")

    y_true = np.array(loader.predictions['y_true'])[:len(all_features)]
    y_pred = np.array(loader.predictions['y_pred'])[:len(all_features)]

    # perplexity must be < n_samples
    perplexity = min(perplexity, max(5, (len(all_features) - 1) // 3))
    print(f"    Running t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                random_state=42, learning_rate='auto', init='pca')
    embed = tsne.fit_transform(all_features)

    # Plot: colour AND marker shape by true class (so classes stay separable
    # in grayscale); misclassified points use a black-edged 'X'.
    fig, ax = plt.subplots(figsize=(7, 6))
    correct = (y_true == y_pred)
    class_markers = ['o', 's', '^', 'D', 'v', 'P']   # one distinct shape per class
    for ci, cls in enumerate(loader.class_names):
        col = COLORS.get(cls, COLORS['neutral'])
        mk = class_markers[ci % len(class_markers)]
        m_ok = (y_true == ci) & correct
        m_no = (y_true == ci) & (~correct)
        ax.scatter(embed[m_ok, 0], embed[m_ok, 1], s=42, c=col, marker=mk,
                   edgecolors='white', linewidths=0.5, alpha=0.85,
                   label=f"{_disp(cls)} (correct)")
        if m_no.any():
            ax.scatter(embed[m_no, 0], embed[m_no, 1], s=130, c=col,
                       marker='X', edgecolors='black', linewidths=1.4,
                       label=f"{_disp(cls)} (misclassified)")

    ax.set_xlabel("t-SNE 1", fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontweight='bold')
    ax.set_title("t-SNE of Penultimate (64-D) Features — Test Set",
                 fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    save_figure(fig, output_dir, "tsne_penultimate_features")
    plt.close(fig)
    print(f"  ✓ t-SNE embedding saved.")


def generate_patient_level_analysis(loader, output_dir):
    """
    Aggregate image/pair-level predictions to a PATIENT-LEVEL diagnosis.

    Clinical rationale: each patient contributes several images (different
    slices / views). A single misclassified image does not mean the wrong
    diagnosis — in practice the clinician reviews all of a patient's images
    together. This routine aggregates every patient's image predictions into
    one diagnosis (by majority vote and, separately, by mean softmax
    probability) and reports how many patients are correctly diagnosed at the
    patient level versus the image level. It also lists which patients (if any)
    remain misclassified after aggregation, and which were "rescued" — i.e.
    had ≥1 wrong image but a correct aggregated diagnosis.
    """
    import csv
    from collections import defaultdict, Counter

    y_true = np.array(loader.predictions['y_true'])
    y_pred = np.array(loader.predictions['y_pred'])
    proba  = np.array(loader.predictions['y_pred_proba'])
    class_names = loader.class_names

    # Patient ID source MUST be aligned with the prediction arrays.
    # test_predictions.pkl stores a 'patients' field built in the SAME order as
    # y_true/y_pred, so it is guaranteed aligned. test_pairs.pkl is a separate
    # file and may be in a different order — using it would scramble the groups
    # and massively inflate the apparent error count.
    patients_field = loader.predictions.get('patients', None)

    def _num_from_names(c_name, s_name):
        for nm in (c_name, s_name):
            if nm and nm != 'BLANK':
                return os.path.basename(nm).split('_')[0]   # e.g. 'P006'
        return None

    def patient_number(idx):
        if patients_field is not None and idx < len(patients_field):
            c_name, s_name = patients_field[idx]
            num = _num_from_names(c_name, s_name)
            if num:
                return num
        c, s, _ = loader.test_pairs[idx]
        return _num_from_names(c, s) or f"unknown_{idx}"

    def cr_sg_names(idx):
        """Return (coronal_name, sagittal_name) for an index, or (None, None)."""
        if patients_field is not None and idx < len(patients_field):
            c_name, s_name = patients_field[idx]
        else:
            c_name, s_name, _ = loader.test_pairs[idx]
        c_name = None if (not c_name or c_name == 'BLANK') else os.path.basename(c_name)
        s_name = None if (not s_name or s_name == 'BLANK') else os.path.basename(s_name)
        return c_name, s_name

    if patients_field is None:
        print("  ⚠ WARNING: test_predictions.pkl has no 'patients' field; "
              "falling back to test_pairs.pkl ordering (may be misaligned).")

    # IMPORTANT: patient numbers are assigned PER CLASS in this dataset
    # (class is encoded in the path/filename), so the same number can denote
    # different patients across classes. The authoritative patient key is
    # therefore (true_class, number) — e.g. 'lesion/P011' ≠ 'mrc/P011'.
    # Keying on the number alone merges distinct patients and corrupts the vote.
    def patient_key(idx):
        return f"{class_names[int(y_true[idx])]}/{patient_number(idx)}"

    groups = defaultdict(list)
    numbers_seen = defaultdict(set)   # number → set of classes (to detect reuse)
    for idx in range(len(y_true)):
        groups[patient_key(idx)].append(idx)
        numbers_seen[patient_number(idx)].add(class_names[int(y_true[idx])])

    reused = {num: sorted(cls) for num, cls in numbers_seen.items() if len(cls) > 1}
    if reused:
        print(f"  ℹ Patient numbers reused across classes (treated as distinct "
              f"patients): {len(reused)}")
        for num, cls in sorted(reused.items()):
            print(f"      {num}: {', '.join(cls)}")

    rows = []
    n_vote_ok = n_mean_ok = 0
    still_wrong, rescued = [], []

    for pid, idxs in sorted(groups.items()):
        idxs = np.array(idxs)
        true_cls = Counter(y_true[idxs]).most_common(1)[0][0]
        preds = y_pred[idxs]
        n_imgs = len(idxs)
        # Count UNIQUE coronal / sagittal source images for this patient
        # (the pair count above is the CR×SG combinatorial product, which is
        # much larger than the actual number of slices). These give the honest
        # per-patient image counts.
        cr_files, sg_files = set(), set()
        for _i in idxs:
            _c, _s = cr_sg_names(int(_i))
            if _c:
                cr_files.add(_c)
            if _s:
                sg_files.add(_s)
        n_cr = len(cr_files)
        n_sg = len(sg_files)
        # Per-image correctness uses each image's OWN true label (sums to the
        # real image-level error count, i.e. 10 total — never more).
        n_correct_imgs = int((preds == y_true[idxs]).sum())
        n_wrong_imgs = n_imgs - n_correct_imgs

        # Majority vote (tie-break by mean probability)
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
        elif n_wrong_imgs > 0:
            rescued.append(pid)

        rows.append({
            'patient': pid,
            'true_class': class_names[true_cls],
            'n_cr_images': n_cr,
            'n_sg_images': n_sg,
            'n_pairs': n_imgs,
            'n_correct_images': n_correct_imgs,
            'n_wrong_images': n_wrong_imgs,
            'majority_vote': class_names[vote_pred],
            'mean_prob_vote': class_names[mean_pred],
            'vote_correct': vote_ok,
            'mean_correct': mean_ok,
        })

    n_patients = len(rows)
    img_acc = float((y_pred == y_true).mean())
    vote_acc = n_vote_ok / n_patients
    mean_acc = n_mean_ok / n_patients

    # Sanity check: total wrong images across patients MUST equal the
    # image-level error count. If not, patient IDs are misaligned with
    # predictions and the grouping is corrupt.
    total_wrong_imgs = sum(r['n_wrong_images'] for r in rows)
    img_errors = int((y_pred != y_true).sum())
    if total_wrong_imgs != img_errors:
        print(f"  ⚠ ALIGNMENT WARNING: summed per-patient wrong images "
              f"({total_wrong_imgs}) ≠ image-level errors ({img_errors}). "
              f"Patient grouping may be misaligned with predictions.")
    else:
        print(f"  ✓ Consistency check passed: {total_wrong_imgs} wrong images "
              f"= {img_errors} image-level errors.")

    # ── Write CSV ────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "patient_level_analysis.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PATIENT-LEVEL DIAGNOSIS (aggregated across each patient's images)")
    print("=" * 60)
    print(f"  Images (pairs):        {len(y_true)}")
    print(f"  Unique patients:       {n_patients}  (keyed by class+number; "
          f"per-class numbering means same number across classes = different patients)")
    print(f"  Error-free patients:   {sum(1 for r in rows if r['n_wrong_images']==0)}/{n_patients}")
    print(f"  Image-level accuracy:  {img_acc:.4f} "
          f"({int((y_pred==y_true).sum())}/{len(y_true)})")
    print(f"  Patient-level (vote):  {vote_acc:.4f} ({n_vote_ok}/{n_patients})")
    print(f"  Patient-level (mean):  {mean_acc:.4f} ({n_mean_ok}/{n_patients})")
    print(f"  Patients rescued by aggregation (had ≥1 wrong image, "
          f"correct diagnosis): {len(rescued)}")
    if rescued:
        print(f"    {', '.join(rescued)}")
    if still_wrong:
        print(f"  Patients still misdiagnosed after aggregation: "
              f"{len(still_wrong)} → {', '.join(still_wrong)}")
    else:
        print(f"  ✓ Every patient is correctly diagnosed at the patient level.")

    # ── Figure: patients with errors (cohort context is in the title) ────────
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

        ax.barh(ypos, correct_counts, color=COLORS['correct'],
                label='Correct pairs', edgecolor='black', linewidth=0.7)
        ax.barh(ypos, wrong_counts, left=correct_counts,
                color=COLORS['incorrect'], label='Misclassified pairs',
                edgecolor='black', linewidth=0.7, hatch='xxx')

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
        save_figure(fig, out, "patient_level_rescue")
        plt.close(fig)

    print(f"  ✓ Saved patient_level_analysis.csv and patient_level_rescue figure")
    return rows


def generate_dataset_composition(dataset_root, output_dir, loader=None):
    """
    Describe the dataset across all three splits (train / val / test):
      • number of distinct patients per class (class-aware: a patient number is
        unique only within its class), and
      • number of coronal (CR) and sagittal (SG) images.

    Counts are taken by walking the image files on disk (not the training
    pairs, which are combinatorial and would massively overcount). Filenames
    follow  P{num}_{class}_{index}_{view}.jpg  so class/view/patient are parsed
    directly from the name and are robust to folder-name casing.
    """
    import glob
    from collections import defaultdict

    splits = ['train', 'val', 'test']
    # Folder-name aliases — the val split is often named differently on disk.
    split_aliases = {
        'train': ['train', 'training', 'tr'],
        'val':   ['val', 'valid', 'validation', 'dev', 'validate'],
        'test':  ['test', 'testing', 'te', 'holdout', 'eval'],
    }
    classes = ['healthy', 'lesion', 'mrc']
    root = Path(dataset_root)

    print("\n" + "=" * 60)
    print(f"DATASET COMPOSITION  (root: {root})")
    print("=" * 60)

    def _blank_stats():
        return {c: {'cr': 0, 'sg': 0, 'patients': set()} for c in classes}

    def _tally(stat, name):
        """Parse one filename into the stats dict."""
        stem = os.path.splitext(os.path.basename(name))[0]
        parts = stem.split('_')
        if len(parts) < 4:
            return
        num, cls, view = parts[0], parts[1].lower(), parts[-1].upper()
        if cls not in classes:
            return
        if 'CR' in view:
            stat[cls]['cr'] += 1
        elif 'SG' in view:
            stat[cls]['sg'] += 1
        stat[cls]['patients'].add(num)

    def _find_split_dir(canonical):
        """Locate the on-disk folder for a split, trying known aliases."""
        if not root.exists():
            return None
        existing = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
        for alias in split_aliases[canonical]:
            if alias in existing:
                return existing[alias]
        return None

    def _stats_from_dir(sp_dir):
        stat = _blank_stats()
        files = glob.glob(str(sp_dir / "**" / "*.jpg"), recursive=True) \
              + glob.glob(str(sp_dir / "**" / "*.png"), recursive=True)
        for f in files:
            _tally(stat, f)
        return stat, len(files)

    def _stats_from_pairs(pairs):
        """Derive composition from saved pairs by de-duplicating image paths
        (pairs are combinatorial, so the same image appears many times)."""
        stat = _blank_stats()
        seen = set()
        for c, s, _ in pairs:
            for p in (c, s):
                if p and p != 'BLANK' and p not in seen:
                    seen.add(p)
                    _tally(stat, p)
        return stat, len(seen)

    # Optional pair-based fallbacks (val_pairs.pkl, test_pairs from loader)
    pair_fallback = {}
    if loader is not None:
        pair_fallback['test'] = loader.test_pairs
        vp = Path(loader.artifacts_dir) / "val_pairs.pkl"
        if vp.exists():
            with open(vp, 'rb') as f:
                pair_fallback['val'] = pickle.load(f)['pairs']

    stats = {}
    for sp in splits:
        sp_dir = _find_split_dir(sp)
        if sp_dir is not None:
            st, nfiles = _stats_from_dir(sp_dir)
            if nfiles == 0 and sp in pair_fallback:
                st, nfiles = _stats_from_pairs(pair_fallback[sp])
                print(f"  • {sp}: directory empty — derived from saved pairs "
                      f"({nfiles} unique images)")
            else:
                print(f"  • {sp}: read from {sp_dir} ({nfiles} images)")
        elif sp in pair_fallback:
            st, nfiles = _stats_from_pairs(pair_fallback[sp])
            print(f"  • {sp}: no '{sp}/' folder found — derived from saved pairs "
                  f"({nfiles} unique images)")
        else:
            st = _blank_stats()
            print(f"  • {sp}: not found on disk and no saved pairs — reported as 0. "
                  f"Pass --dataset-root or run where this split exists.")
        stats[sp] = st

    # ── Print + collect rows ─────────────────────────────────────────────────
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    import csv
    rows = []
    print(f"\n  {'Split':<6} {'Class':<8} {'Patients':>9} {'Coronal':>8} "
          f"{'Sagittal':>9} {'Images':>7}")
    print("  " + "-" * 52)
    split_tot = defaultdict(lambda: {'patients': 0, 'cr': 0, 'sg': 0})
    for sp in splits:
        for c in classes:
            d = stats[sp][c]
            npat, ncr, nsg = len(d['patients']), d['cr'], d['sg']
            rows.append({'split': sp, 'class': c, 'patients': npat,
                         'coronal': ncr, 'sagittal': nsg, 'images': ncr + nsg})
            split_tot[sp]['patients'] += npat
            split_tot[sp]['cr'] += ncr
            split_tot[sp]['sg'] += nsg
            print(f"  {sp:<6} {c:<8} {npat:>9} {ncr:>8} {nsg:>9} {ncr+nsg:>7}")
        t = split_tot[sp]
        print(f"  {sp:<6} {'TOTAL':<8} {t['patients']:>9} {t['cr']:>8} "
              f"{t['sg']:>9} {t['cr']+t['sg']:>7}")
        print("  " + "-" * 52)

    with open(out / "dataset_composition.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── Figure: two panels ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(splits)); width = 0.25
    cls_colors = [COLORS.get(c, COLORS['neutral']) for c in classes]
    # Hatch patterns give a second (grayscale-safe) channel alongside colour.
    cls_hatches = ['', '///', '...']        # healthy, lesion, mrc
    view_hatches = ['', 'xxx']              # Coronal, Sagittal

    # Panel 1 — patients per class per split
    for i, c in enumerate(classes):
        vals = [len(stats[sp][c]['patients']) for sp in splits]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=_disp(c),
                       color=cls_colors[i], hatch=cls_hatches[i],
                       edgecolor='black', linewidth=0.7)
        ax1.bar_label(bars, fontsize=8, padding=2)
    ax1.set_xticks(x); ax1.set_xticklabels([s.capitalize() for s in splits])
    ax1.set_ylabel("Number of patients", fontweight='bold')
    ax1.set_title("Distinct patients per class\n(patient IDs unique within class)",
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel 2 — coronal vs sagittal images per split
    cr_vals = [split_tot[sp]['cr'] for sp in splits]
    sg_vals = [split_tot[sp]['sg'] for sp in splits]
    b1 = ax2.bar(x - width / 2, cr_vals, width, label='Coronal (CR)',
                 color=COLORS.get('healthy', '#4878CF'), hatch=view_hatches[0],
                 edgecolor='black', linewidth=0.7)
    b2 = ax2.bar(x + width / 2, sg_vals, width, label='Sagittal (SG)',
                 color=COLORS.get('lesion', '#C44E52'), hatch=view_hatches[1],
                 edgecolor='black', linewidth=0.7)
    ax2.bar_label(b1, fontsize=8, padding=2)
    ax2.bar_label(b2, fontsize=8, padding=2)
    ax2.set_xticks(x); ax2.set_xticklabels([s.capitalize() for s in splits])
    ax2.set_ylabel("Number of images", fontweight='bold')
    ax2.set_title("Coronal vs sagittal images per split", fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle("Dataset Composition Across Splits", fontweight='bold', fontsize=14)
    fig.tight_layout()
    save_figure(fig, out, "dataset_composition")
    plt.close(fig)

    tot_pat = sum(split_tot[sp]['patients'] for sp in splits)
    print(f"\n  Total distinct patients (all splits): {tot_pat}")
    print(f"  ✓ Saved dataset_composition.csv and figure")
    return rows


def print_summary(loader):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    y_true = np.array(loader.predictions['y_true'])
    y_pred = np.array(loader.predictions['y_pred'])
    
    accuracy = (y_true == y_pred).mean()
    n_misclassified = (y_true != y_pred).sum()
    
    print(f"\n  Run: {loader.run_name}")
    print(f"  Test samples: {len(y_true)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Misclassified: {n_misclassified}")
    
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(loader.class_names):
        mask = y_true == i
        cls_acc = (y_pred[mask] == i).mean() if mask.sum() > 0 else 0
        print(f"    {cls}: {cls_acc:.4f} ({mask.sum()} samples)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ODIN Post-Training Analysis — Publication Quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python post_analysis.py --run run_20260516_164902
  python post_analysis.py --run run_20260516_164902 --gradcam
  python post_analysis.py --run run_20260516_164902 --all
        """
    )
    
    parser.add_argument('--run', required=True, help='Run name')
    parser.add_argument('--logs-dir', default='logs', help='Logs directory')
    parser.add_argument('--output-dir', default='output_figures/post_analysis',
                       help='Output directory')
    parser.add_argument('--data-dir', default=None, 
                       help='Data directory containing images (auto-detected if not specified)')
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM panels')
    parser.add_argument('--misclassified', action='store_true', help='Analyze misclassified')
    parser.add_argument('--tsne', action='store_true',
                       help='Regenerate t-SNE embedding of penultimate features (Option A)')
    parser.add_argument('--patient', action='store_true',
                       help='Patient-level diagnosis aggregation (image→patient majority/mean)')
    parser.add_argument('--dataset', action='store_true',
                       help='Dataset composition across train/val/test (patients + CR/SG counts)')
    parser.add_argument('--dataset-root', default='train_test_val_data',
                       help='Folder containing train/ val/ test/ (for --dataset)')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--model-type', default='composite',
                       choices=['loss', 'f1', 'composite'], help='Model checkpoint')
    parser.add_argument('--layer', default=DEFAULT_LAYER,
                       choices=['per_branch', 'post_fusion', 'last_conv'],
                       help='Which layer to visualize (single mode). Omit with '
                            '--all to cover ALL three depths: '
                            'per_branch=multiply_1/3 (CBAM per-view), '
                            'post_fusion=conv2d_8 (after gating), '
                            'last_conv=conv2d_10 (deepest, most class-specific)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ODIN POST-TRAINING ANALYSIS")
    print("Publication-Quality Grad-CAM Visualization")
    print("=" * 60)
    
    loader = ArtifactLoader(args.run, args.logs_dir)
    print_summary(loader)

    global LAYER_CHOICE
    run_all = args.all or not (args.gradcam or args.misclassified or args.tsne
                               or args.patient or args.dataset)

    model = loader.load_model(args.model_type)
    test_seq = loader.create_test_sequence(data_dir_override=args.data_dir)

    base_dir = Path(args.output_dir) / args.run

    # Which layer modes to cover for Grad-CAM.
    # --all (or no specific flag) covers ALL THREE depths: per-branch CBAM,
    # post-fusion, and last conv. A specific --layer narrows it to one.
    if run_all or args.gradcam or args.misclassified:
        if args.all or (run_all and args.layer == DEFAULT_LAYER):
            layer_modes = ['per_branch', 'post_fusion', 'last_conv']
        else:
            layer_modes = [args.layer]

        for mode in layer_modes:
            LAYER_CHOICE = mode
            output_dir = base_dir / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "#" * 60)
            print(f"# LAYER MODE: {mode}  →  {output_dir}")
            print("#" * 60)

            if run_all or args.gradcam:
                generate_paper_gradcam_panel(loader, model, test_seq, output_dir)
                generate_combined_panel(loader, model, test_seq, output_dir)
            if run_all or args.misclassified:
                generate_misclassified_gradcam(loader, model, test_seq, output_dir)

    # t-SNE is layer-independent (penultimate 64-D features) — run once.
    if run_all or args.tsne:
        tsne_dir = base_dir / "tsne"
        tsne_dir.mkdir(parents=True, exist_ok=True)
        generate_tsne_embedding(loader, model, test_seq, tsne_dir)

    # Patient-level aggregation is prediction-based (layer-independent) — run once.
    if run_all or args.patient:
        patient_dir = base_dir / "patient_level"
        patient_dir.mkdir(parents=True, exist_ok=True)
        generate_patient_level_analysis(loader, patient_dir)

    # Dataset composition walks the image files on disk — run once.
    if run_all or args.dataset:
        dataset_dir = base_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset_composition(args.dataset_root, dataset_dir, loader)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\n📁 Figures saved under: {base_dir}/")
    print("   ├── per_branch/   — per-view CBAM Grad-CAM")
    print("   ├── post_fusion/  — fused (post-gating) Grad-CAM")
    print("   ├── last_conv/    — deepest conv Grad-CAM (most class-specific)")
    print("   │     ├── gradcam_panel_correct.pdf/png/svg")
    print("   │     ├── gradcam_combined_panel.pdf/png/svg")
    print("   │     └── gradcam_misclassified.pdf/png/svg")
    print("   ├── tsne/         — tsne_penultimate_features.pdf/png/svg")
    print("   ├── patient_level/ — patient_level_analysis.csv + rescue figure")
    print("   └── dataset/      — dataset_composition.csv + figure (train/val/test)")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
