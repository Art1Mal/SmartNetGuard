# SmartNetGuard — Level 2 (L2) Classifier  
Dual-Input Conv1D Window Classifier with L1 Embeddings & Temperature Calibration

## Overview

This module implements the **Level 2 (L2) classifier** in the SmartNetGuard pipeline.

L2 is a **dual-input window classifier** that combines:

1. **Time-series of engineered L2 features** (WINDOW_SIZE × F) — processed by Conv1D blocks.  
2. **Embedding from L1 AutoEncoder** (bottleneck / En4) — computed for the same window.

The output is a **multi-class prediction** over canonical attack classes:

- `volumetric_flood`  
- `http_flood`  
- `bot`  
- `portscan`  

The script performs full offline training, evaluation, and calibration of L2, using a trained L1 run for embeddings and SLA verification.

---

## Data & Inputs

The script expects two **Parquet** files for L2 and one **L1 run** path:

- `trainval` — Parquet with **row-wise time series** for training + validation, including:
  - L1 base features (BASE7)
  - additional fields (e.g. bwd features, extra columns)
  - a **`label`** column with string attack labels

- `test` — Parquet with the same structure (rows by time, `label` present).

- `l1_run` — either:
  - path to **L1 run directory** (`run_*`), containing:
    - `SmartNetGuard_DeepConv1D_AE.keras` or `finetuned/SmartNetGuard_DeepConv1D_AE_finetuned.keras`
    - `preprocessing_config.json`
  - or a **direct path** to the L1 `.keras` model (config is taken from its parent run directory).

Default paths in the script:

```python
DEFAULT_TRAINVAL_PATH = r"...\for_l2\L2_trainval_ready_no_bruteforce.parquet"
DEFAULT_TEST_PATH     = r"...\for_l2\L2_test_ready_no_bruteforce.parquet"
DEFAULT_OUTPUT_DIR    = r"...\L2_experiments\l2_dual_input_L2only"
DEFAULT_L1_RUN        = r"...\L1_ready_for_working\run_04_09_2025_08-23-48\..."

Replace ... with your real paths before running.

Role of L2 in SmartNetGuard
* L2 works after L1:
  * L1 detects anomalies and provides window embeddings.
  * L2 classifies anomalous (or all) windows into attack types.
* L2 is L2-only clean in this file:
  * No Zero-Day gateway, no OOD, no event stitching, no KPI orchestration.
  * Pure supervised window classification with calibrated probabilities.
This module is intended as a clean training/evaluation engine for the classifier itself.

Canonical Classes & Label Mapping
The classifier uses a fixed canonical class order:
  CANONICAL_CLASS_ORDER = ["volumetric_flood", "http_flood", "bot", "portscan"]
String labels in the Parquet files are normalized via canonicalize_label() and CANON_LABEL_ALIASES.
Examples:
* ddos_volumetric_flood → volumetric_flood
* dos_http_flood → http_flood
* network_portscan → portscan
* auth_bruteforce → bruteforce (filtered out in this L2-only version)
Only rows whose labels are in CANONICAL_CLASS_ORDER are kept for training and testing.

Features & Feature Engineering
Base Features (visible even for encrypted traffic)
The L2 pipeline is built on top of the same BASE7 as L1:
BASE_FEATURE_NAMES = [
    "flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_mean", "flow_iat_mean", "flow_pkts_per_sec"
]
Optional backward (bwd) candidates:
BWD_BASE_CANDIDATES = [
    "tot_bwd_pkts", "totlen_bwd_pkts", "bwd_pkt_len_max",
    "bwd_pkt_len_mean", "bwd_iat_mean"
]

Derived Features for L2
compute_derived_features() adds additional engineered features such as:
* fwd_bytes_per_sec = totlen_fwd_pkts / flow_duration
* fwd_bytes_per_pkt = totlen_fwd_pkts / tot_fwd_pkts
* pktlen_max_over_mean = fwd_pkt_len_max / fwd_pkt_len_mean
* consistency = flow_pkts_per_sec * flow_iat_mean (sanity check ≈ 1 for stable flows)
* fwd_pkts_per_sec = tot_fwd_pkts / flow_duration
* payload_sparsity
* log_tot_fwd_pkts, log_totlen_fwd_pkts
If bwd fields are present, symmetric features and ratios are computed:
* bwd_bytes_per_sec, bwd_bytes_per_pkt, bwd_pkts_per_sec
* ratio_bytes_per_sec_bwd_fwd, ratio_pkts_per_sec_bwd_fwd, ratio_bytes_per_pkt_bwd_fwd
get_all_feature_names() returns the final feature list used for L2 training, always starting with core forward features and appending bwd/ratio features if available.

Windows & Cleanliness
Data is row-wise by time. The script builds windows of length WINDOW_SIZE:
 WINDOW_SIZE = 112
Window statistics (rolling_label_stats()):
* For each time window:
  * dom = dominant class ID within the window
  * purity = fraction of rows belonging to the dominant class in that window
Purity is used for:
* Train sampling (pure vs mixed windows)
* Val/Test strict vs mixed splits

Train/Val/Test Splits
1. Train/Val by Class
split_train_val_per_class() splits the trainval Parquet into:
* df_train — training rows
* df_val — validation rows
Split is done separately for each class using TRAIN_FRAC_PER_CLASS (default 0.80).
2. Train Window Sampling
sample_train_windows_random_starts():
* Computes dominant_class, purity for sliding windows over df_train.
* For each canonical class:
  * Samples a target number of windows:
    * TRAIN_TARGET_PER_CLASS_MAP = {"volumetric_flood": 7000, "http_flood": 9000, "bot": 10000, "portscan": 10000}
  * Combines:
    * pure windows (purity == 1.0)
    * mixed windows (purity >= TRAIN_MIX_MINFRAC, default 0.75, with higher thresholds for some classes)
  * Uses class-specific overrides:
    * PURE_FRACTION_MAP, MIX_MINFRAC_MAP for bot and http_flood.
Output:
* Xtr_ts — time-series windows for training, shape (B, WINDOW_SIZE, F)
* ytr — window class IDs (0..C-1)
* purity_tr — per-window purity for weighting and analysis
3. Val/Test Strict vs Mixed
prepare_eval_with_minfrac() builds eval windows for:
* Strict: windows with purity ≥ VALTEST_STRICT_MINFRAC (default 0.98)
* Mixed: windows with purity ≥ VALTEST_MIXED_MINFRAC (default 0.70)
Splits:
* val_strict / val_mixed from df_val
* test_strict / test_mixed from df_test
Each split has:
* start_indices, labels, purity, and the corresponding (T,F) tensors.

L1 SLA & Embeddings
The L2 classifier depends on L1 for embeddings:
1. Find L1 model & config
  find_l1_model_and_config() locates:
  * L1 .keras model (.../SmartNetGuard_DeepConv1D_AE.keras or finetuned/...)
  * preprocessing_config.json with:
    * feature_names (BASE7)
    * standardizer.mean / standardizer.scale
    * z_clip
    * window_size
2. SLA Check
  assert_sla_and_prepare_l1_preproc() verifies:
  * window_size in L1 config == WINDOW_SIZE in L2
  * All L1 features are present in the L2 dataset
  * Extracts:
    * l1_feats, mean, scale, z_clip bounds
3. Standardization for L1
   standardize_for_l1() re-applies L1’s standardization and clipping to the L1 base features from L2 data:
    * Input: df_train/df_val/df_test
    * Output: 2D arrays ready as L1 input.
4. Embeddings
   load_l1_and_make_head() builds an embedding head:
    * Prefer layer: "Bottleneck_dense"
    * Fallback: "En4"
    compute_embeddings_for_windows() then computes an embedding per window on top of standardized L1 inputs.

L2 Model Architecture
build_dual_input_model() builds a dual-input Keras model:
* Input 1 — Time-Series Path ((T, F)):
    * Conv1D(64, kernel=5) → BN → ReLU → MaxPool1D(2)
    * Conv1D(128, kernel=5) → BN → ReLU → MaxPool1D(2)
    * Dilated Conv1D(128, dilation rates 1, 2, 4) with BN + ReLU
    * GlobalAveragePooling1D → vector
* Input 2 — L1 Embedding Path ((emb_dim,)):
    * LayerNormalization
    * Dense(64, activation="relu")
* Head:
    * Concatenate([ts_vector, emb_vector])
    * Dropout
    * Dense(128, activation="relu")
    * Dropout
    * Dense(num_classes) — logits, no softmax (useful for calibration).
Regularization:
* L2 weight decay on dense/conv layers: L2_WEIGHT
* Dropout rate: DROPOUT_HEAD

Loss, Optimization & Augmentations
Focal Loss
SparseFocalLoss:
* Works with sparse labels (0..C-1)
* Parameters:
  * gamma = FOCAL_GAMMA (default 1.0)
  * alpha = FOCAL_ALPHA — class weights, aligned with CANONICAL_CLASS_ORDER
* Applied on logits (from_logits=True).
Optimizer & LR Schedule
* If available: AdamW with CosineDecayRestarts schedule.
* Fallback: standard Adam with fixed learning_rate = 5e-4.
Window Augmentations
Augmentations are applied only to time-series input ((T,F)); embeddings remain unchanged:
* Jitter: additive Gaussian noise (JITTER_STDDEV)
* Time Mask: randomly zero-out a continuous fragment in time (TIME_MASK_PROB, TIME_MASK_MAX_FRAC)
* Feature Mask: randomly drop a fraction of features across the entire window (FEATURE_MASK_PROB, FEATURE_MASK_FRAC)
All augmentations are implemented in a tf.function for efficiency.


Training Pipeline
main() orchestrates the following steps:
1.Set seeds & devices (set_global_seed, detect_devices)
2.Create run directory under output_dir:
  * e.g. run_YYYYMMDD_HHMMSS_win112_seed42
3.Load trainval and test Parquet; canonicalize label strings.
4.Compute derived features; determine FEAT_NAMES.
5. Split TRAIN/VAL per class
6. Clip & standardize (L2 side):
  * Robust quantile clips per feature: [0.5%, 99.5%]
  * Standardization (mean/std) based on TRAIN only
  * Save clip_stats.json, scaler_stats.json
7. Sample TRAIN windows (class-conditional) → Xtr_ts, ytr, purity_tr.
8. L1 SLA + embeddings:
  * Load L1, verify config, standardize base features for L1.
  * Compute embeddings Etr for training windows.
9. Build tf.data train dataset:
  * Apply augmentations to time-series, compute sample-weights based on:
    * class imbalance
    * window purity
10. Build L2 model and compile with SparseFocalLoss.
11. Prepare VAL/TEST windows (strict & mixed).
12. Train:
  * Monitor val_loss on strict validation set.
  * Callbacks: CSVLogger, EarlyStopping, ModelCheckpoint.
  * Save:
    * history.json
    * curve_loss.png
    * epoch_log.csv
    * SmartNetGuard_L2_DualInput_best.keras
13. Evaluate & calibrate:
  * Evaluate val_strict (no calibration) → baseline.
  * Fit temperature T on val_mixed logits w.r.t CALIBRATION_OBJECTIVE (ece or nll).
  * Save calibration.json.
  * Evaluate val_mixed, test_strict, test_mixed using calibrated temperature.
14. Save final artifacts:
  * SmartNetGuard_L2_DualInput_last.keras
  * metrics_*.json (classification reports + confusion matrices)
  * confusion_*.png, f1_by_class_*.png, hist_confidence_*.png
  * summary.json with full launch configuration.


Metrics & Reports
For each split (val/test, strict/mixed), the script computes:
* Accuracy
* Balanced accuracy
* ROC AUC macro (ovr, multi-class) if possible
* Per-class metrics (precision, recall, F1) via classification_report
* Confusion matrix:
  * Stored in JSON
  * Normalized confusion plotted as confusion_<name>.png
* Per-class F1 bar chart (f1_by_class_<name>.png)
* Max confidence histogram (hist_confidence_<name>.png)
Temperature scaling is chosen on val_mixed to minimize:
* ECE (CALIBRATION_OBJECTIVE="ece") or
* NLL ("nll")
Calibrated probabilities are then used on test splits.

Outputs Overview
For each run, the output directory contains:
<output_dir>/
  run_YYYYMMDD_HHMMSS_win112_seed42/
    SmartNetGuard_L2_DualInput_best.keras
    SmartNetGuard_L2_DualInput_last.keras
    history.json
    curve_loss.png
    epoch_log.csv
    clip_stats.json
    scaler_stats.json
    calibration.json
    metrics_val_strict.json
    metrics_val_mixed.json
    metrics_test_strict.json
    metrics_test_mixed.json
    confusion_val_strict.png
    confusion_val_mixed.png
    confusion_test_strict.png
    confusion_test_mixed.png
    f1_by_class_val_strict.png
    f1_by_class_val_mixed.png
    f1_by_class_test_strict.png
    f1_by_class_test_mixed.png
    hist_confidence_val_strict.png
    hist_confidence_val_mixed.png
    hist_confidence_test_strict.png
    hist_confidence_test_mixed.png
    summary.json

summary.json acts as a launch passport and includes:
* paths used
* class order
* feature names
* train sampling configuration
* augmentation hyperparameters
* optimizer & focal loss parameters
* L2 regularization and dropout settings.

How to Run
1. Install Dependencies
  pip install numpy pandas scikit-learn matplotlib tensorflow keras
 If you want to use AdamW with CosineDecayRestarts, ensure Keras / TF versions support it.
2. Adjust Defaults or Use CLI
Option A — using defaults in __main__:
In interactive/IPython:
if __name__ == "__main__":
    IN_IPY = "ipykernel" in sys.modules
    if IN_IPY:
        main(DEFAULT_TRAINVAL_PATH, DEFAULT_TEST_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_L1_RUN)
Option B — CLI:
python Smart_Net_Guard_L2.py \
  --trainval "C:\...\L2_trainval_ready_no_bruteforce.parquet" \
  --test     "C:\...\L2_test_ready_no_bruteforce.parquet" \
  --outdir   "C:\...\L2_experiments\l2_dual_input_L2only" \
  --l1_run   "C:\...\L1_ready_for_working\run_04_09_2025_08-23-48" \
  --window   112 \
  --batch_size 256 \
  --epochs   40 \
  --seed     42

Safety & Publishing Notes
* The script does not contain credentials, secrets, or private tokens.
* Paths in the repository should use either placeholders (...) or generic local directories.
* All domain-specific logic is data-agnostic (no sensitive IPs, no payload content).
* It is safe to publish this module and its README in a public GitHub repository as part of a portfolio project.

License
This module is distributed under the MIT License as part of the SmartNetGuard project.
Maintainer

SmartNetGuard — Network Threat Detection Pipeline (L1 + L2)
Author: Artiom Maliovanii
