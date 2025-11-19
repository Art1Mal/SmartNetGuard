#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNetGuard – Unified dataset preprocessing (L1 & L2)
-------------------------------------------------------

This module implements the unified preprocessing used in the project.
SmartNetGuard for dataset preparation:

    • L1 – AutoEncoder anomaly detector (trained on BENIGN traffic),
    • L2 – classifier of known attacks (multi-class).

Main tasks:
    - recursive traversal of a directory with .parquet files (different datasets),
    - bringing column names to a single scheme (feature aliases),
    - removal of technical identifiers (IP, ports, timestamp, ID),
    - normalization of attack labels (label aliases),
    - data cleaning (Inf → NaN, removing rows with NaN),
    - converting all numeric columns to float32,
    - optional selection of basic 7 features (BASE7 features),
    - saving cleaned .parquet

Operating modes:
    - mode="l1" — preparing the dataset for AutoEncoder:
        * filtering only BENIGN (if there is a label column),
        * deleting the label column,
        * then only numerical features remain (BASE7 or full set).

    - mode="l2" — preparing the dataset for the classifier:
        * normalization of string labels to canonical classes,

Each source parquet file is saved in a separate subfolder by its name, in train files and stored.
(averages by features) in JSON format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration: technical columns and feature/label aliases
# ---------------------------------------------------------------------------

# Technical columns that are never fed into the model
TECH_COLUMNS: List[str] = [
    "Flow ID", "Flow_ID", "flow_id",
    "Src IP", "src_ip", "Dst IP", "dst_ip",
    "Src Port", "src_port", "Dst Port", "dst_port",
    "Timestamp", "timestamp", "Time", "time",
]

# The 7 basic signs that are "visible" even when encrypted (the canonical set)
BASE7_FEATURES: List[str] = [
    "flow_duration",
    "tot_fwd_pkts",
    "totlen_fwd_pkts",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "flow_iat_mean",
    "flow_pkts_per_sec",
]

# Aliases for merging different datasets with a single feature name
FEATURE_ALIAS_GROUPS: Dict[str, List[str]] = {
    "flow_duration": [
        "flow_duration",
        "Flow Duration",
        "Flow_Duration",
        "flowDur",
        "dur",
    ],
    "tot_fwd_pkts": [
        "tot_fwd_pkts",
        "Tot Fwd Pkts",
        "Tot_Fwd_Pkts",
        "total_fwd_packets",
    ],
    "totlen_fwd_pkts": [
        "totlen_fwd_pkts",
        "TotLen Fwd Pkts",
        "TotLen_Fwd_Pkts",
        "total_fwd_bytes",
        "total_fwd_payload_bytes",
    ],
    "fwd_pkt_len_max": [
        "fwd_pkt_len_max",
        "Fwd Pkt Len Max",
        "Fwd_Pkt_Len_Max",
        "fwd_packet_length_max",
    ],
    "fwd_pkt_len_mean": [
        "fwd_pkt_len_mean",
        "Fwd Pkt Len Mean",
        "Fwd_Pkt_Len_Mean",
        "fwd_packet_length_mean",
    ],
    "flow_iat_mean": [
        "flow_iat_mean",
        "Flow IAT Mean",
        "Flow_IAT_Mean",
        "iat_mean",
    ],
    "flow_pkts_per_sec": [
        "flow_pkts_per_sec",
        "Flow Pkts/s",
        "Flow Packets/s",
        "flow_packets_per_s",
        "flow_pkts_per_s",
    ],
}

# Aliases for casting string labels to canonical classes
LABEL_ALIAS_GROUPS: Dict[str, List[str]] = {
    "BENIGN": ["BENIGN", "Benign", "benign", "NORMAL", "Normal", "normal"],
    "ATTACK": ["Attack", "ATTACK"],
    "Bot": ["Bot", "bot", "Botnet"],
    "DoS": ["DoS", "DDoS", "DDOS"],
    "PortScan": ["PortScan", "Port Scan", "portscan"],
    "Web Attack": ["Web Attack", "WebAttack"],
}


# ---------------------------------------------------------------------------
# Auxiliary functions
# ---------------------------------------------------------------------------

def harmonize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converting column names to canonical names according to FEATURE_ALIAS_GROUPS.
    """
    rename_map: Dict[str, str] = {}
    existing = set(df.columns)

    for canonical, aliases in FEATURE_ALIAS_GROUPS.items():
        for alias in aliases:
            if alias in existing:
                rename_map[alias] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def drop_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removing technical columns (IP addresses, ports, timestamps, ID).
    """
    cols_to_drop = [c for c in TECH_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all remaining columns to float32 numeric type.

    - Numeric types → float32.
    - Strings/objects → pd.to_numeric(..., errors="coerce") → float32.
    - Columns that contain only NaN after the conversion are removed.
    """
    d = df.copy()

    for col in list(d.columns):
        s = d[col]

        if pd.api.types.is_numeric_dtype(s):
            d[col] = s.astype("float32")
            continue

        # We try to carefully convert it to a numerical form
        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().any():
            d[col] = converted.astype("float32")
        else:
            # Completely non-numeric column - delete
            d = d.drop(columns=[col])

    return d


def clean_nans_and_infs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +/-Inf with NaN and remove rows with NaN.
    """
    d = df.copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(axis=0, inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


def normalize_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
    keep_labels: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Normalization of label values by LABEL_ALIAS_GROUPS and (optional)
    filtering by list of acceptable classes.
    """
    if label_col not in df.columns:
        return df

    # raw_value_lower -> canonical_label
    alias_to_canonical: Dict[str, str] = {}
    for canonical, aliases in LABEL_ALIAS_GROUPS.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical

    labels = df[label_col].astype(str).str.strip()
    canonical = labels.map(lambda x: alias_to_canonical.get(x.lower(), x))

    df = df.copy()
    df[label_col] = canonical

    if keep_labels is not None:
        keep_set = set(keep_labels)
        df = df[df[label_col].isin(keep_set)]

    df.reset_index(drop=True, inplace=True)
    return df


def filter_benign_only(
    df: pd.DataFrame,
    label_col: str = "Label",
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """
    Keeps only BENIGN rows and removes the label column.
    If there is no label column, returns the dataframe unchanged.
    """
    if label_col not in df.columns:
        return df

    df = normalize_labels(df, label_col=label_col)
    df = df[df[label_col] == benign_label].copy()
    df = df.drop(columns=[label_col])
    df.reset_index(drop=True, inplace=True)
    return df


def select_base7_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leaves only the intersection of BASE7_FEATURES with the actual available columns.
    """
    available = [c for c in BASE7_FEATURES if c in df.columns]
    if not available:
        return df
    return df[available].copy()


# ---------------------------------------------------------------------------
# Preprocessing a single file
# ---------------------------------------------------------------------------

def preprocess_single_file(
    parquet_path: Path,
    out_root: Path,
    mode: str = "l1",
    keep_only_base7: bool = True,
    filter_benign: bool = True,
    label_col: str = "Label",
    l2_keep_labels: Optional[Iterable[str]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Full cycle of preprocessing of one .parquet file.

    Returns:
        (path_to_cleaned_parquet, path_to_scaler_json_or_None)

    StandardScaler is saved only for files that contain the word "train" in their name (case insensitive).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    subdir = out_root / parquet_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] {parquet_path}")

    # 1) Loading
    df = pd.read_parquet(parquet_path)

    # 2) Renaming and deleting technical columns
    df = harmonize_feature_names(df)
    df = drop_technical_columns(df)

    # 3) Processing of tags depending on the mode
    if mode.lower() == "l1":
        if filter_benign:
            df = filter_benign_only(df, label_col=label_col)
        # For L1, labels are no longer needed - only features remain
    elif mode.lower() == "l2":
        df = normalize_labels(df, label_col=label_col, keep_labels=l2_keep_labels)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'l1' or 'l2'.")

    # 4) Cast to float32 + NaN/Inf clearing
    df = cast_numeric_columns(df)
    df = clean_nans_and_infs(df)

    # 5) Selection of basic features (optional)
    if keep_only_base7:
        df = select_base7_features(df)

    if df.empty:
        print(f"[WARN] Resulting dataframe is empty after cleaning: {parquet_path.name}")

    # 6) Preserving cleaned parquet
    clean_path = subdir / f"{parquet_path.stem}_cleaned_float32.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"  saved cleaned parquet: {clean_path}")

    return clean_path


# ---------------------------------------------------------------------------
# Recursive directory processing
# ---------------------------------------------------------------------------

def walk_and_preprocess(
    in_root: str | Path,
    out_root: str | Path,
    mode: str = "l1",
    keep_only_base7: bool = True,
    filter_benign: bool = True,
    label_col: str = "Label",
    l2_keep_labels: Optional[Iterable[str]] = None,
) -> None:
    """
    Recursively traverses the `in_root` directory and processes all .parquet files.

    For each file, its own subfolder is created in `out_root` with the cleaned parquet
    """
    in_root_path = Path(in_root)
    out_root_path = Path(out_root)
    if not in_root_path.exists() or not in_root_path.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {in_root_path}")

    parquet_files = sorted(in_root_path.rglob("*.parquet"))
    if not parquet_files:
        print(f"[WARN] No parquet files found under: {in_root_path}")
        return

    print(f"[walk] Found {len(parquet_files)} parquet files under {in_root_path}")

    for p in parquet_files:
        try:
            preprocess_single_file(
                parquet_path=p,
                out_root=out_root_path,
                mode=mode,
                keep_only_base7=keep_only_base7,
                filter_benign=filter_benign,
                label_col=label_col,
                l2_keep_labels=l2_keep_labels,
            )
        except Exception as exc:
            print(f"[ERROR] Failed on {p}: {exc}")


if __name__ == "__main__":
    # Example of usage (paths must be replaced with real ones):
    RAW_ROOT = r"/path/to/raw_parquet_datasets"
    OUT_ROOT = r"/path/to/cleaned_datasets"

    # Example: Preparing datasets for L1 (BENIGN-only, BASE7)
    # walk_and_preprocess(
    #     in_root=RAW_ROOT,
    #     out_root=OUT_ROOT,
    #     mode="l1",
    #     keep_only_base7=True,
    #     filter_benign=True,
    # )

    # Example: Preparing datasets for L2 (multi-class, BASE7 only)
    # walk_and_preprocess(
    #     in_root=RAW_ROOT,
    #     out_root=OUT_ROOT,
    #     mode="l2",
    #     keep_only_base7=True,
    #     filter_benign=False,
    #     l2_keep_labels=["BENIGN", "Bot", "DoS", "PortScan", "Web Attack"],
    # )

    pass
