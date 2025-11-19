# SmartNetGuard — Level 2 (L2) Classifier  
**Dual-Input Conv1D Window Classifier with L1 Embeddings & Temperature Calibration**

---

## Overview

This module implements the **Level 2 (L2) classifier** in the SmartNetGuard pipeline.

L2 is a **dual-input window classifier** that combines:

- **Time-series of engineered L2 features** `WINDOW_SIZE × F` — processed by Conv1D blocks.  
- **Embedding from L1 AutoEncoder** (bottleneck / En4) — computed for the same window.

The output is a **multi-class prediction** over canonical attack classes:

- `volumetric_flood`
- `http_flood`
- `bot`
- `portscan`

The script performs full **offline training**, **evaluation**, and **temperature calibration** of L2, using a trained L1 run for **embeddings** and **SLA verification**.

---

## Data & Inputs

The script expects two **Parquet** files for L2 and one **L1 run** path:

- **`trainval`** — Parquet with row-wise time series for training + validation, including:
  - L1 base features (**BASE7**)
  - additional fields (e.g. bwd features, extra engineered columns)
  - a `label` column with string attack labels

- **`test`** — Parquet with the same structure (rows by time, `label` present).

- **`l1_run`** — either:
  - path to an L1 **run directory** (`run_*`), containing:
    - `SmartNetGuard_DeepConv1D_AE.keras` or  
      `finetuned/SmartNetGuard_DeepConv1D_AE_finetuned.keras`
    - `preprocessing_config.json`
  - or a **direct path** to the L1 `.keras` model  
    (the config is taken from the parent run directory).

**Default paths** in the script (you must replace `...` with real paths):

    DEFAULT_TRAINVAL_PATH = r"...\for_l2\L2_trainval_ready_no_bruteforce.parquet"
    DEFAULT_TEST_PATH     = r"...\for_l2\L2_test_ready_no_bruteforce.parquet"
    DEFAULT_OUTPUT_DIR    = r"...\L2_experiments\l2_dual_input_L2only"
    DEFAULT_L1_RUN        = r"...\L1_ready_for_working\run_04_09_2025_08-23-48\..."

---

## Role of L2 in SmartNetGuard

- L2 works **after** L1:
  - L1 detects anomalies and provides **window embeddings**.
  - L2 classifies anomalous (or all) windows into **attack types**.

- This file is a **“L2-only clean”** implementation:
  - No Zero-Day gateway
  - No OOD module
  - No event stitching
  - No KPI orchestration

It’s a **clean training/evaluation engine** for the L2 classifier itself.

---

## Canonical Classes & Label Mapping

The classifier uses a fixed **canonical class order**:

    CANONICAL_CLASS_ORDER = ["volumetric_flood", "http_flood", "bot", "portscan"]

String labels in the Parquet files are normalized via:

- `canonicalize_label()`
- `CANON_LABEL_ALIASES`

Examples:

- `ddos_volumetric_flood` → `volumetric_flood`
- `dos_http_flood`        → `http_flood`
- `network_portscan`      → `portscan`
- `auth_bruteforce`       → `bruteforce` (filtered out in this L2-only version)

Only rows whose labels are in **`CANONICAL_CLASS_ORDER`** are kept for training and testing.

---

## Features & Feature Engineering

### Base Features (visible even in encrypted traffic)

The L2 pipeline is built on top of the same **BASE7** features as L1:

    BASE_FEATURE_NAMES = [
        "flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts",
        "fwd_pkt_len_max", "fwd_pkt_len_mean",
        "flow_iat_mean", "flow_pkts_per_sec"
    ]

Optional backward (**bwd**) candidates:

    BWD_BASE_CANDIDATES = [
        "tot_bwd_pkts", "totlen_bwd_pkts",
        "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_iat_mean"
    ]

### Derived Features for L2

`compute_derived_features()` adds additional engineered features, for example:

- `fwd_bytes_per_sec`  = `totlen_fwd_pkts / (flow_duration + eps)`
- `fwd_bytes_per_pkt`  = `totlen_fwd_pkts / (tot_fwd_pkts + eps)`
- `pktlen_max_over_mean` = `fwd_pkt_len_max / (fwd_pkt_len_mean + eps)`
- `consistency` = `flow_pkts_per_sec * flow_iat_mean`  
  (sanity check ≈ 1 for stable flows)
- `fwd_pkts_per_sec`   = `tot_fwd_pkts / (flow_duration + eps)`
- `payload_sparsity`
- `log_tot_fwd_pkts`, `log_totlen_fwd_pkts`

If bwd fields are present, symmetric backward features and ratios are computed:

- `bwd_bytes_per_sec`, `bwd_bytes_per_pkt`, `bwd_pkts_per_sec`
- `ratio_bytes_per_sec_bwd_fwd`
- `ratio_pkts_per_sec_bwd_fwd`
- `ratio_bytes_per_pkt_bwd_fwd`

`get_all_feature_names()` returns the final **L2 feature list** used for training, always starting with core forward features and appending bwd/ratio features if available.

---

## Windows & Cleanliness

Data is provided **row-wise by time**.  
The script builds sliding windows of length:

    WINDOW_SIZE = 112

Window statistics are computed via `rolling_label_stats()`:

For each time window, it returns:

- `dom`   — dominant class ID within the window
- `purity` — fraction of rows belonging to the dominant class `[0..1]`

Purity is used for:

- **Train sampling** (pure vs mixed windows)
- **Val/Test** strict vs mixed splits

---

## Train / Val / Test Splits

### 1. Train / Val by Class

`split_train_val_per_class()` splits the **trainval Parquet** into:

- `df_train` — training rows  
- `df_val`   — validation rows  

The split is performed **separately for each class**, using:

    TRAIN_FRAC_PER_CLASS = 0.80

(80% train, 20% val inside each class).

---

### 2. Train Window Sampling

`sample_train_windows_random_starts()`:

1. Computes `dominant_class` and `purity` for sliding windows over `df_train`.
2. For each canonical class, it:
   - Samples a **target number of windows**:

        TRAIN_TARGET_PER_CLASS_MAP = {
            "volumetric_flood": 7000,
            "http_flood":       9000,
            "bot":             10000,
            "portscan":        10000
        }

   - Combines:
     - **pure windows** (`purity == 1.0`)
     - **mixed windows** (`purity >= TRAIN_MIX_MINFRAC`, default `0.75`,  
       with **higher thresholds** for some classes)

   - Uses per-class overrides:
     - `PURE_FRACTION_MAP`
     - `MIX_MINFRAC_MAP` (especially for `bot` and `http_flood`)

Output:

- `Xtr_ts` — time-series windows for training, shape `(B, WINDOW_SIZE, F)`  
- `ytr` — window class IDs `0..C-1`  
- `purity_tr` — per-window purity for weighting and analysis  

---

### 3. Val / Test — Strict vs Mixed

`prepare_eval_with_minfrac()` builds evaluation windows for:

- **Strict**:
  - windows with purity ≥ `VALTEST_STRICT_MINFRAC` (default `0.98`)

- **Mixed**:
  - windows with purity ≥ `VALTEST_MIXED_MINFRAC` (default `0.70`)

Resulting splits:

- `val_strict`, `val_mixed` from `df_val`  
- `test_strict`, `test_mixed` from `df_test`

Each split has:

- `start_indices`  
- `labels` (class IDs)  
- `purity`  
- and the corresponding `(T,F)` tensors for model inputs.

---

## L1 SLA & Embeddings

The L2 classifier **depends on L1** for embeddings.

### 1. Find L1 Model & Config

`find_l1_model_and_config()` locates:

- the L1 `.keras` model:
  - `SmartNetGuard_DeepConv1D_AE.keras`  
    or `finetuned/SmartNetGuard_DeepConv1D_AE_finetuned.keras`
- `preprocessing_config.json` with:
  - `feature_names` (BASE7)
  - `standardizer.mean` / `standardizer.scale`
  - `z_clip`
  - `window_size`

### 2. SLA Check

`assert_sla_and_prepare_l1_preproc()` verifies:

- `window_size` in L1 config == `WINDOW_SIZE` in L2
- All L1 features are present in the L2 dataset

It extracts:

- `l1_feats`
- `mean`
- `scale`
- `z_clip` bounds (lo/hi)

### 3. Standardization for L1

`standardize_for_l1()` re-applies L1 standardization and clipping to the **L1 base features** taken from L2 data.

Input:

- `df_train`, `df_val`, `df_test`

Output:

- 2D arrays ready as L1 input: `(N_rows, len(l1_feats))`

### 4. Embeddings

`load_l1_and_make_head()` builds an embedding head:

- Prefer layer: `"Bottleneck_dense"` (Dense bottleneck)  
- Fallback: `"En4"` (GAP on En4_conv)

`compute_embeddings_for_windows()` then computes an **embedding per window** using standardized L1 inputs.

---

## L2 Model Architecture

`build_dual_input_model()` builds a **dual-input Keras model**:

### Input 1 — Time-Series Path `(T, F)`

- Conv1D(64, kernel=5) → BatchNorm → ReLU → MaxPool1D(2)  
- Conv1D(128, kernel=5) → BatchNorm → ReLU → MaxPool1D(2)  
- Dilated Conv1D(128, kernel=3) with dilation rates `1, 2, 4`  
  (each with BatchNorm + ReLU)  
- GlobalAveragePooling1D → time-series feature vector

### Input 2 — L1 Embedding Path `(emb_dim,)`

- LayerNormalization  
- Dense(64, activation="relu")

### Head (Fusion + MLP)

- Concatenate `[ts_vector, emb_vector]`
- Dropout
- Dense(128, activation="relu")
- Dropout
- Dense(`num_classes`) → **logits** (no softmax)

Regularization:

- L2 weight decay on Conv/Dense layers: `L2_WEIGHT`  
- Dropout rate in head: `DROPOUT_HEAD`

---

## Loss, Optimization & Augmentations

### Focal Loss

`SparseFocalLoss`:

- Works with **sparse labels** `0..C-1`
- Parameters:
  - `gamma = FOCAL_GAMMA` (default `1.0`)
  - `alpha = FOCAL_ALPHA` — vector of class weights aligned with `CANONICAL_CLASS_ORDER`
- Operates on **logits** (`from_logits=True`)

Purpose: focus learning on **hard** examples and handle **class imbalance**.

---

### Optimizer & LR Schedule

- If available:
  - `AdamW` with `CosineDecayRestarts` learning rate schedule
- Fallback:
  - standard `Adam` with fixed `learning_rate = 5e-4`

---

### Window Augmentations

Augmentations are applied **only to the time-series input** `(T,F)`; L1 embeddings remain unchanged.

Implemented in `_augment()` (wrapped in `tf.function`):

- **Jitter**:
  - Additive Gaussian noise with std `JITTER_STDDEV`

- **Time Mask**:
  - With probability `TIME_MASK_PROB`
  - Random continuous block in time axis zeroed out
  - Length up to `TIME_MASK_MAX_FRAC` of the window

- **Feature Mask**:
  - With probability `FEATURE_MASK_PROB`
  - Random fraction `FEATURE_MASK_FRAC` of features masked across the entire window

These augmentations improve robustness and generalization of L2.

---

## Training Pipeline

The `main()` function orchestrates the entire pipeline:

1. **Set seeds & devices**
   - `set_global_seed(seed)`
   - `detect_devices()` (GPU / CPU, soft memory growth)

2. **Create run directory** under `output_dir`
   - Example:  
     `run_YYYYMMDD_HHMMSS_win112_seed42`

3. **Load** `trainval` and `test` Parquet
   - Canonicalize `label` strings to canonical class names.

4. **Compute derived features & feature list**
   - `compute_derived_features()`
   - `get_all_feature_names()` → `FEAT_NAMES`

5. **Split TRAIN / VAL per class**
   - `split_train_val_per_class()`  
   - Balanced splitting per canonical class.

6. **Clip & standardize (L2 side)**
   - Robust quantile clips per feature: `[0.5%, 99.5%]`
   - Standardization `(x - mean) / std` based on TRAIN only.
   - Saves:
     - `clip_stats.json`
     - `scaler_stats.json`

7. **Sample TRAIN windows**
   - `sample_train_windows_random_starts()` →  
     `Xtr_ts`, `ytr`, `purity_tr` (class-conditional and purity-aware).

8. **L1 SLA + embeddings**
   - `find_l1_model_and_config()`
   - `assert_sla_and_prepare_l1_preproc()`
   - `standardize_for_l1()` for train/val/test
   - Build L1 embedding head:
     - `load_l1_and_make_head()`
   - Compute window embeddings:
     - `Etr` for training windows

9. **Build tf.data TRAIN dataset**
   - Combine `(Xtr_ts, Etr)` with labels `ytr` and `sample_weights`  
   - Sample weights depend on:
     - class imbalance
     - window purity
   - Apply `_augment()` to time-series windows
   - Batch + prefetch

10. **Build & compile L2 model**
    - `build_dual_input_model()`
    - Loss: `SparseFocalLoss`
    - Optimizer: `AdamW + CosineDecayRestarts` or `Adam`

11. **Prepare VAL/TEST windows (strict & mixed)**
    - `prepare_eval_with_minfrac()` for:
      - `val_strict`, `val_mixed`
      - `test_strict`, `test_mixed`
    - Compute embeddings for each split using L1 head.

12. **Training**
    - Train with training dataset `ds_train`
    - Validation data: **strict** validation (`val_strict`)  
      (embeddings + time-series windows)
    - Callbacks:
      - `CSVLogger` → `epoch_log.csv`
      - `EarlyStopping` (on `val_loss`)
      - `ModelCheckpoint` (saves best model)
    - Saves:
      - `history.json` (loss/metrics history)
      - `curve_loss.png` (training/validation curves)
      - `SmartNetGuard_L2_DualInput_best.keras` (best by `val_loss`)

13. **Evaluation & Calibration**
    - `eval_and_dump_basic()` for each split:
      - Computes accuracy, balanced accuracy, ROC AUC macro
      - Builds classification report & confusion matrix
      - Plots:
        - Confusion matrix (normalized, `.png`)
        - F1 per class bar chart
        - Confidence histogram (max softmax)
      - Saves JSON & plots.

    - Workflow:
      - Evaluate `val_strict` **without calibration** → baseline.
      - On `val_mixed`:
        - Fit temperature `T` over a grid `TS_GRID` using:
          - `CALIBRATION_OBJECTIVE = "ece"` or `"nll"`
        - Save `calibration.json`
        - Re-evaluate `val_mixed` with calibrated probabilities.
      - For `test_strict` and `test_mixed`:
        - Use stored temperature from `calibration.json`.

14. **Save final artifacts**
    - `SmartNetGuard_L2_DualInput_last.keras`
    - `metrics_val_strict.json`
    - `metrics_val_mixed.json`
    - `metrics_test_strict.json`
    - `metrics_test_mixed.json`
    - `confusion_*.png`, `f1_by_class_*.png`, `hist_confidence_*.png`
    - `summary.json` — launch configuration & metadata.

---

## Metrics & Reports

For each split (`val/test`, `strict/mixed`), the script computes:

- **Accuracy**
- **Balanced Accuracy**
- **ROC AUC (macro, OVR)** — if numerically possible
- **Per-class metrics**:
  - precision
  - recall
  - F1-score (via `classification_report`)

**Confusion matrices**:

- Saved to `metrics_*.json` and as plots:
  - `confusion_val_strict.png`
  - `confusion_val_mixed.png`
  - `confusion_test_strict.png`
  - `confusion_test_mixed.png`

**Per-class F1 bar charts**:

- `f1_by_class_val_strict.png`
- `f1_by_class_val_mixed.png`
- `f1_by_class_test_strict.png`
- `f1_by_class_test_mixed.png`

**Max confidence histograms** (max softmax per window):

- `hist_confidence_val_strict.png`
- `hist_confidence_val_mixed.png`
- `hist_confidence_test_strict.png`
- `hist_confidence_test_mixed.png`

**Temperature scaling**:

- Chosen on `val_mixed` to minimize:
  - ECE (`CALIBRATION_OBJECTIVE = "ece"`) or
  - NLL (`CALIBRATION_OBJECTIVE = "nll"`)
- Calibrated probabilities are then used for **test** splits.

---

## Outputs Overview

For each run, the output directory looks like:

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

`summary.json` serves as a **“launch passport”** and includes:

- paths used (`trainval`, `test`, `outdir`, `l1_run`)
- class order (`CANONICAL_CLASS_ORDER`)
- feature names (`FEAT_NAMES`)
- train sampling configuration
- augmentation hyperparameters
- optimizer and focal loss parameters
- L2 regularization and dropout settings

---

## How to Run

### 1. Install Dependencies

Example (minimal set):

    pip install numpy pandas scikit-learn matplotlib tensorflow keras

If you want to use **AdamW + CosineDecayRestarts**, ensure your Keras / TF versions support these APIs.

---

### 2. Run with Defaults (from Python / IPython)

The script has:

    if __name__ == "__main__":
        IN_IPY = "ipykernel" in sys.modules
        if IN_IPY:
            main(DEFAULT_TRAINVAL_PATH, DEFAULT_TEST_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_L1_RUN)
        else:
            ...

In a notebook or IPython environment, just call:

    main(
        trainval_path=DEFAULT_TRAINVAL_PATH,
        test_path=DEFAULT_TEST_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        l1_run=DEFAULT_L1_RUN
    )

(after replacing the default paths with real ones).

---

### 3. Run via CLI

Example:

    python Smart_Net_Guard_L2.py ^
      --trainval "C:\...\L2_trainval_ready_no_bruteforce.parquet" ^
      --test     "C:\...\L2_test_ready_no_bruteforce.parquet" ^
      --outdir   "C:\...\L2_experiments\l2_dual_input_L2only" ^
      --l1_run   "C:\...\L1_ready_for_working\run_04_09_2025_08-23-48" ^
      --window   112 ^
      --batch_size 256 ^
      --epochs   40 ^
      --seed     42

(adapt paths and parameters as needed).

---

## Safety & Publishing Notes

- The script does **not** contain any credentials, secrets, or tokens.
- Repository paths should use either placeholder prefixes (`...`) or generic local paths.
- All domain-specific logic is **data-agnostic**:
  - No user IPs
  - No payload contents
  - Only abstract flow statistics

It is safe to publish this module and its README in a **public GitHub repository** as part of a portfolio project.

---

## License

This module is distributed under the **MIT License** as part of the **SmartNetGuard** project.

---

## Maintainer

**SmartNetGuard — Network Threat Detection Pipeline (L1 + L2)**  
Author: **Artiom Maliovanii**
