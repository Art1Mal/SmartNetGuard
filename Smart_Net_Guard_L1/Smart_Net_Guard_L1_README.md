# SmartNetGuard — Level 1 (L1) Detector  
Deep Conv1D Autoencoder with Fine-Tuning & Event-Level Metrics

## Overview

This module implements the **Level 1 (L1) anomaly detector** of the SmartNetGuard pipeline.
It trains a **deep 1D convolutional autoencoder** on sliding windows of network features, evaluates reconstruction errors, computes AR-based metrics, performs **fine-tuning on “hard normals”**, and clusters latent embeddings (En4) via **HDBSCAN / DBSCAN** for anomaly structure analysis.
L1 is responsible for:
- Learning the normal (BENIGN) traffic behaviour on **time windows**  
- Detecting anomalous windows via **reconstruction error (MSE)**  
- Providing:
  - AR metrics (Attack/Normal reconstruction ratio)
  - **event-level detection metrics** under a budget of flagged windows
  - En4 embeddings for downstream analysis and L2 integration
  - A standardized preprocessing config for L2
This module is designed as an **offline trainer & evaluator** for the L1 AutoEncoder.
---
## Data & Inputs
The script expects three input files in **Parquet** format:
- `TRAIN_PATH` — features for training (no labels), shape `(N_train, F)`
- `TEST_PATH` — features for testing (no labels), shape `(N_test, F)`
- `LABEL_PATH` — binary labels for test rows, shape `(N_test,)`, values `{0=normal, 1=attack}`
These files should already be cleaned and harmonized by the **preprocessing module** (BASE7 / unified features).
In the code:

```python
TRAIN_PATH = r"...\Project Smart Net Guard\L1_dataset's\X_train_ready_full_cleaned_float32.parquet"
TEST_PATH  = r"...\Project Smart Net Guard\L1_dataset's\X_test_ready_renamed_cleaned_float32.parquet"
LABEL_PATH = r"...\Project Smart Net Guard\L1_dataset's\y_test_binary_cleaned.parquet"

SAVE_DIR   = r"...\Project Smart Net Guard\L1_ready_for_working"
Replace the ... placeholders with your actual local paths before running.

Core Concepts
Windows
* Raw rows are standardized and then converted to sliding time windows:
  * Shape: (T, F) where:
    * T = WINDOW_SIZE (e.g. 112 rows)
    * F = num_features
* L1 always works at the window level, not single rows.
Reconstruction Error
* Autoencoder is trained to reconstruct the same window (x → x)
* Reconstruction error per window:
  MSE_window_mean = mean_t,f ((x - x_hat)^2)
* Errors are used for:
  * Attack / normal separation (AR-value)
  * Ranking windows for event-level metrics & alarms
  * Picking “hard normals” for fine-tune
AR-Value
* AR (Attack/Normal ratio) is defined as:
  AR = mean(MSE | attack) / mean(MSE | normal)
* Higher AR ⇒ better separation of attacks and normal traffic at window level
The script also computes:
* AR(any-in-window) — label 1 if any attack row appears inside the window
* AR_trimmed — AR after trimming extreme 1% error values (robust to outliers)

Event-Level Metrics
Instead of only counting windows, the script groups anomalies into events using connected True-runs in a binary mask:
* For a given budget b (e.g. 4% of all windows):
  * Take the top-K windows by reconstruction error (K = ceil(b * N)).
  * Build a binary prediction mask on windows.
  * Aggregate to events and compute:
    * event_recall — fraction of GT events hit by at least one predicted window
    * event_precision — fraction of predicted event-clusters that overlap at least one GT event
    * Time-To-Detect (TTD): median and 95th percentile detection delay from start of each event
Budget list is controlled via:
  EVENT_BUDGETS_TO_REPORT = [0.04]  # e.g. 4% top windows

Model Architecture
The autoencoder is a deep Conv1D encoder–decoder with:
* 4 Conv1D encoder blocks:
  256 → 128 → 64 → 32 filters with Norm + LeakyReLU + Dropout
* GlobalAveragePooling on the last conv (En4) to obtain a window embedding
* Bottleneck:
  * Dense(8) + normalization + LeakyReLU
  * Squeeze-and-Excitation on the bottleneck vector
* Decoder:
  * RepeatVector(T)
  * Conv1D stack 32 → 64 → 128 → 256 with Norm + LeakyReLU + Dropout
  * Skip-connection from En4_conv via Conv1D(32, 1) + Add
  * Final Conv1D(F, 1) for reconstruction
Training details:
* Optimizer:
  * Tries AdamW with weight decay (if available), otherwise falls back to Adam
  * Gradient clipping via clipnorm=GRAD_CLIP_NORM
* Loss:
  * Huber loss with delta = HUBER_DELTA (robust to outliers)
* Normalization:
  * LayerNorm (default) or BatchNorm controlled by USE_LAYER_NORM****

Training Pipeline
1. Load data from TRAIN_PATH, TEST_PATH, LABEL_PATH
2. Standardize features with StandardScaler (fit on train, apply to test)
3. Apply Z-clip to ±8 to limit extreme z-scores
4. Create a new run_YYYY_MM_DD_HH-MM-SS directory in SAVE_DIR
  * Save std_mean.npy and std_scale.npy
5. Build and compile the Conv1D autoencoder
6. Split train into:
  * train_for_fit (95%)
  * train_holdout_norm (5%) — used as validation stream
7. Build TF datasets:
  * Random windows from train_for_fit for training
  * Random windows from holdout for validation
8. Train with:
  * EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, EarlyStopping, ReduceLROnPlateau
9. Save:
  * .keras model
  * hyperparams.json
  * train_history.json
  * train_curve.png (loss/val_loss)

Evaluation Pipeline
1. Build sliding windows over test data:
  * stride = 1, no shuffling
2. Split sequence into:
  * validation part (prefix) — first VALIDATION_FRACTION of windows
  * final part (suffix) — remaining windows
3. Assign labels:
  * y_window_ends — label based on last row in window
  * y_any_in_window — 1 if any attack row inside window
4. Compute reconstruction errors for both splits via:
  * compute_reconstruction_errors_streaming(...)
5. Compute metrics:
  * AR (window-end)
  * AR(any-in-window)
  * Trimmed AR (1% tails)
6. Compute event-level metrics for configured budgets on:
  * Validation split
  * Final split
7. Save:
  * ar_values.json
  * diagnostics.json
  * event_metrics_val.json
  * event_metrics_final.json

Fine-Tuning on “Hard Normals”
If FINETUNE_ENABLED = True, the script performs an additional training stage:
1. Identify windows that are:
  * Labelled as normal (any-in-window = 0)
  * Far from attacks via mask dilation (dilate_attack_mask)
2. From those “safe normals”:
  * Sort by reconstruction error (descending)
  * Take the top FINETUNE_TOP_PCT_NORMALS (capped by FINETUNE_MAX_WINDOWS)
3. Train the autoencoder a few extra epochs on this subset with:
  * Reduced learning rate = LEARNING_RATE * FINETUNE_LR_FACTOR
4.Recompute reconstruction errors and AR metrics after fine-tune
5. Save under run_dir/finetuned/:
  * SmartNetGuard_DeepConv1D_AE_finetuned.keras
  * finetune_info.json
  * compare_before_after.json (AR before vs after)
  * Extra event-level metrics: *_after_ft.json
This helps the model better reconstruct borderline normal windows without overfitting to attacks.

Clustering of En4 Embeddings
The script also explores the structure of anomalies in latent space:
1. Extract embeddings from layer "En4" for a subsample of final windows
2. Reduce dimensionality with IncrementalPCA (e.g. 16 components)
3. Cluster using:
  * HDBSCAN (preferred, if installed), or
  * DBSCAN as a fallback
4. Compute:
  * Cluster sizes
  * Approximate attack rate per cluster (if labels available)
5. Save:
  * hdbscan_idx_subset.npy — indices of sampled windows
  * hdbscan_labels_subset.npy — cluster labels
  * hdbscan_cluster_stats.json — summary
  * hdbscan_pca_subset.png — 2D PCA visualization colored by cluster
This can be used for anomaly family discovery and potential Zero-Day analysis.

Outputs Overview
For each run, the script creates a directory:
  <L1_ready_for_working>/
    run_DD_MM_YYYY_HH-MM-SS/
        SmartNetGuard_DeepConv1D_AE.keras
        std_mean.npy
        std_scale.npy
        preprocessing_config.json
        hyperparams.json
        train_history.json
        train_curve.png
        ar_values.json
        diagnostics.json
        event_metrics_val.json
        event_metrics_final.json
        hist_reconstruction_error_val.png
        hist_reconstruction_error_final.png
        hist_feature_*.png
        per_feature_errors_val.json
        per_feature_errors_final.json
        hdbscan_idx_subset.npy
        hdbscan_labels_subset.npy
        hdbscan_cluster_stats.json
        hdbscan_pca_subset.png
        summary.txt
        finetuned/
            SmartNetGuard_DeepConv1D_AE_finetuned.keras
            finetune_info.json
            compare_before_after.json
            event_metrics_val_after_ft.json
            event_metrics_final_after_ft.json
preprocessing_config.json is especially important for the L2 module, as it contains:
* Feature names
* Standardization parameters (mean, scale)
* Z-clip range
* Window size
* Error metric name

How to Run
1. Install dependencies (example):
pip install numpy pandas scikit-learn matplotlib tensorflow keras hdbscan
2. Adjust paths in the script:
TRAIN_PATH = r"C:\Users\...\X_train_ready_full_cleaned_float32.parquet"
TEST_PATH  = r"C:\Users\...\X_test_ready_renamed_cleaned_float32.parquet"
LABEL_PATH = r"C:\Users\...\y_test_binary_cleaned.parquet"
SAVE_DIR   = r"C:\Users\...\L1_ready_for_working"
3. Run the script:
python Smart_Net_Guard_L1.py
  Note: Training is GPU-accelerated if a compatible GPU is available.
  The script also enables memory growth for TensorFlow GPUs to avoid pre-allocating all VRAM.

Safety Notes
* The script does not contain any secrets, credentials or private tokens.
* All paths in the repo use either placeholders (...) or generic local directories.
* It is safe to publish this file and its README on GitHub as part of a public portfolio project.

License
This module is distributed under the MIT License as part of the SmartNetGuard project.

Maintainer
SmartNetGuard — Network Threat Detection Pipeline (L1 + L2)
Author: Artiom Maliovanii
