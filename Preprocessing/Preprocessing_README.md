# SmartNetGuard — Preprocessing Module  
Unified Dataset Preparation for L1 (AutoEncoder) and L2 (Classifier)

## Overview
The preprocessing module standardizes all raw network-flow datasets used in the **SmartNetGuard** pipeline.  
It unifies features across heterogeneous datasets (CICIDS2017, CIC-IDS-2018, UNSW, Bot-IoT, IoT-23, etc.), removes technical identifiers, normalizes label values, converts raw features to a unified schema, and produces clean `.parquet` files ready for training:

- **L1 AutoEncoder** — anomaly detection on BENIGN-only traffic  
- **L2 Classifier** — supervised multi-class attack classification  

This module ensures **dataset compatibility**, **feature consistency**, and **clean numerical input** for all stages of SmartNetGuard.

---

## Key Responsibilities

### ✔️ 1. Recursive dataset traversal
The module scans any directory tree and automatically finds **all `.parquet` files**:

raw_datasets/
├── CICIDS2017/
├── UNSW-NB15/
├── Bot-IoT/
└── ... (any other datasets)

### ✔️ 2. Feature standardization & renaming
Different datasets use different column names.  
Example:  
`"Flow Duration"`, `"flowDur"`, `"dur"` → **flow_duration**

All aliases are merged into a canonical name using `FEATURE_ALIAS_GROUPS`.

### ✔️ 3. Removal of technical identifiers
The following fields are removed automatically:
- Flow ID  
- IP addresses  
- Ports  
- Timestamps  
- Session identifiers  

These features **cannot** be used for ML security models due to:
- leakage of ground-truth attack patterns  
- instability between datasets  
- zero generalization to encrypted traffic  

### ✔️ 4. Label normalization
Labels are normalized into canonical classes:

BENIGN
ATTACK
Bot
DoS
PortScan
Web Attack


This enables consistent multi-class training for L2.

### ✔️ 5. Data cleaning
- Convert all columns → `float32`  
- Strings → numeric (via `pd.to_numeric`)  
- Non-numeric columns removed  
- Replace `inf` / `-inf` → NaN  
- Drop rows containing NaN  

### ✔️ 6. BASE7 feature extraction (optional)
SmartNetGuard’s BASE7 set includes stable metadata features visible even under TLS/HTTPS encryption:

flow_duration
tot_fwd_pkts
totlen_fwd_pkts
fwd_pkt_len_max
fwd_pkt_len_mean
flow_iat_mean
flow_pkts_per_sec

Used for both L1 and L2.

### ✔️ 7. BENIGN-only filtering (L1 mode)
If `mode="l1"`:
- Only BENIGN traffic is kept  
- Label column is removed  

This prepares clean normal traffic for AutoEncoder anomaly detection.

### ✔️ 8. Output structure (per-file subfolders)
Each input file is saved into:

cleaned_datasets/
<filename_without_ext>/
<filename>_cleaned_float32.parquet

---

## Usage

### 🔧 Basic example (L1 mode – AutoEncoder)
```python
from Smart_Net_Guard_Preprocessing import walk_and_preprocess

walk_and_preprocess(
    in_root="raw_parquet_datasets",
    out_root="cleaned_for_L1",
    mode="l1",
    keep_only_base7=True,
    filter_benign=True
)
```
This will:
* scan raw_parquet_datasets/
* preprocess all files
* keep only BENIGN rows
* select only BASE7 features
* save the results into cleaned_for_L1/

🔧 Example (L2 mode – Supervised classifier)
walk_and_preprocess(
    in_root="raw_parquet_datasets",
    out_root="cleaned_for_L2",
    mode="l2",
    keep_only_base7=True,
    filter_benign=False,
    l2_keep_labels=["BENIGN", "Bot", "DoS", "PortScan", "Web Attack"]
)
This:
* keeps multiple classes
* normalizes all label variants
* removes technical IDs
* produces supervised L2-ready datasets
---
Functions Overview
harmonize_feature_names(df)
Maps feature aliases to canonical names.
drop_technical_columns(df)
Removes unsafe identifiers (IP, ports, timestamps).
cast_numeric_columns(df)
Converts all numeric columns to float32; removes non-numeric ones.
clean_nans_and_infs(df)
Replaces inf → NaN and removes rows with NaN.
normalize_labels(df)
Standardizes label spelling and class names.
filter_benign_only(df)
Selects only BENIGN rows and removes label column.
select_base7_features(df)
Keeps only BASE7 features.
preprocess_single_file(path, out_root, mode, ...)
Processes one .parquet and returns its output path.
walk_and_preprocess(in_root, out_root, ...)
Recursively processes an entire directory tree.
---
Output Files Example
```
cleaned_for_L1/
    Thursday-WorkingHours/
        Thursday-WorkingHours_cleaned_float32.parquet
    Friday-Afternoon/
        Friday-Afternoon_cleaned_float32.parquet
```
All files contain:
* only float32 numeric features
* unified feature names
* no IP/port/timestamp leakage
* no NaNs / no Infs
* optionally BASE7 only
---
Safety Notes
This script contains no credentials, no API keys, no private device paths, and no sensitive information.
It is 100% safe for public GitHub publication.
---
License
This module is released under the MIT License as part of the SmartNetGuard project.
---
Maintainer
SmartNetGuard Project (L1 + L2 NDR Pipeline)
Author: Artiom Maliovanii
