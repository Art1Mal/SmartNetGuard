#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNetGuard - unified runner (switch)

High-level pipeline
-------------------
This script is the main "switch" / entry point for the SmartNetGuard production pipeline.

The pipeline is:

    Source (pcap or live sniffer)
        -> FeatureTransformer (packet → per-flow features, IAT in microseconds, TLS info)
        -> L1 AutoEncoder (emb_head + recon_head)
        -> pre-L2 filter (optional, based on L1 reconstruction MSE/MAE)
        -> L2 classifier (Dual-Input: windowed features + L1 embedding + logit calibration)
        -> ZDGate (Zero-Day gate based on uncertainty and margins)
        -> EventStitcher (group windows into higher-level events)
        -> Context/policy router (HTTP / volumetric / bot / portscan semantics)
        -> OOD-lite on events (simple score combining MSE, pmax, policy conflicts)
        -> Final event label + flags (zd/ctx_mismatch/ood)
        -> Optional: live dumps, logits dumps, confident-window dumps, email alerts.

Key assumptions and agreements
------------------------------
- The FeatureTransformer already emits:
    * IAT in microseconds (`--iat_unit us`),
    * time-binned flow statistics (`time_index`, `flow_pkts_per_sec`, etc.),
    * TLS-related columns such as `is_tls`, `tls_alpn`, `tls_sni`, `tls_ja3` when available.
  Post-rescale of IAT is disabled here – we trust the transformer configuration.

- L1 AutoEncoder:
    * works on a standardised Z-space defined by `preprocessing_config.json`,
    * returns embeddings from bottleneck layer and reconstruction for MSE/MAE,
    * is used as an anomaly detector (high MSE → more suspicious window).

- L2 classifier:
    * uses robust clipped + standardised features (`scaler_stats.json` + `clip_stats.json`),
    * may be calibrated using `calibration.json` (temperature + per-class bias),
    * acts as the primary class predictor for `volumetric_flood`, `http_flood`, `bot`, `portscan`.

- Policy router:
    * looks at per-event context (HTTP/TLS, bot ports, portscan patterns, PPS),
    * can confirm, flip, or downgrade labels (or mark as Zero-Day / unknown),
    * uses presets `--policy_preset default|strict` to change the strength of HTTP confirmation,
    * symmetrically checks HTTP↔VOL (volumetric flood) to avoid systematic mislabelling.

- ZDGate:
    * operates on window-level probabilities (optionally with separate policy temperature),
    * flags uncertain windows via thresholds on `pmax`, entropy, and margin.

- OOD-lite:
    * operates at event-level using a simple linear score:
          score = α * (MSE_norm) + β * (1 - pmax) + γ * ctx_mismatch
    * high score → event is treated as unknown / Zero-Day-like.

- Outputs:
    * window-level: console prints + optional CSV/Parquet logs (live / logits / confident),
    * event-level: console prints, CSV/Parquet dumps, optional streaming JSON and email alerts.

The goal of this file is to keep all orchestration logic in one place, with strict separation between:
    - feature engineering,
    - model loading and preparation,
    - window generation,
    - post-processing (policy / OOD / stitching),
    - IO (sniffer/pcap, email alerts, logging).

Docstrings and comments in this file are intentionally verbose so that:
    - a new engineer can treat it as documentation of the production pipeline;
    - a future maintainer can reason about each decision (e.g., thresholds, masks, policy flow);
    - behaviour of legacy code remains clear even years later.
"""

from __future__ import annotations

import os, sys, time, argparse, socket, gc, json, importlib, types, re, ipaddress, smtplib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models as KM
from email.message import EmailMessage

# ==== internal modules====
from the_flow_transformer import FeatureTransformer  # transformer: IAT to μs and TLS fields (if available)

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
# ----------------------------------------------------------------------
# DICTIONARY OF FIELD ALIASES
# ----------------------------------------------------------------------
ALIAS2CANON = {
    "flow duration": "flow_duration",
    "tot fwd pkts": "tot_fwd_pkts",
    "total fwd packets": "tot_fwd_pkts",
    "totlen fwd pkts": "totlen_fwd_pkts",
    "total length of fwd packets": "totlen_fwd_pkts",
    "fwd pkt len max": "fwd_pkt_len_max",
    "fwd packet length max": "fwd_pkt_len_max",
    "fwd pkt len mean": "fwd_pkt_len_mean",
    "fwd packet length mean": "fwd_pkt_len_mean",
    "flow iat mean": "flow_iat_mean",
    "flow iat mean (ms)": "flow_iat_mean",
    "flow iat mean (us)": "flow_iat_mean",
    "flow iat mean (µs)": "flow_iat_mean",
    "flow packets per sec": "flow_pkts_per_sec",
    "flow pkts per sec": "flow_pkts_per_sec",
    "tot bwd pkts": "tot_bwd_pkts",
    "total bwd packets": "tot_bwd_pkts",
    "totlen bwd pkts": "totlen_bwd_pkts",
    "total length of bwd packets": "totlen_bwd_pkts",
    "bwd pkt len max": "bwd_pkt_len_max",
    "bwd packet length max": "bwd_pkt_len_max",
    "bwd pkt len mean": "bwd_pkt_len_mean",
    "bwd packet length mean": "bwd_pkt_len_mean",
    "bwd iat mean": "bwd_iat_mean",
}

# -------------------------------------------------------------
# slug
# -------------------------------------------------------------
def _slug(s: str) -> str:
    """
    Normalise a raw column name to a simple lowercase "slug".

    This helps to unify feature names coming from different tools/datasets.

    Steps:
        - strip whitespace;
        - lowercase;
        - drop parentheses and their content ("(ms)", "(us)", etc.);
        - replace '-' and '/' with spaces;
        - remove all non-alphanumeric/underscore/space characters;
        - collapse consecutive spaces;
        - replace spaces with '_'.

    Parameters
    ----------
    s : str
        Original column name as present in the DataFrame.

    Returns
    -------
    str
        Canonicalised slug used for alias lookup and canonical names
    """
    s = s.strip().lower()
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")

# ---------------------------------------------------------------------
# normalize_feature_names
# ---------------------------------------------------------------------
def normalize_feature_names(df: pd.DataFrame, debug=False) -> pd.DataFrame:
    """
    Rename DataFrame columns to a canonical form using the ALIAS2CANON mapping.

    Columns are first passed through `_slug`, then, if there is a known alias,
    they're mapped to a canonical feature name (e.g. 'tot fwd pkts' → 'tot_fwd_pkts').

    This function does NOT drop any columns; it only renames them.

    Parameters
    ----------
    df : pandas.DataFrame
        Input feature frame from the transformer (or external CSV/parquet).
    debug : bool, optional
        If True, logs the rename mapping for diagnostics.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with renamed columns (if needed).
    """
    ren = {}
    for c in df.columns:
        base = _slug(c)
        canon = ALIAS2CANON.get(base, base)
        if canon != c:
            ren[c] = canon
    if ren:
        if debug:
            _log("INFO", f"[FEAT-NORM] rename -> {ren}")
        df = df.rename(columns=ren)
    return df

# ==== constants/classes ====
CANONICAL_CLASS_ORDER = ["volumetric_flood", "http_flood", "bot", "portscan"]
WINDOW_SIZE = 112
NUMERIC_EPS = 1e-6
_ALPN_HTTP_REGEX = r"(?:^|,|\s)(?:h2|http/1\.1)(?:$|,|\s)"

# ---------- logging utilities ----------
def _supports_reconfigure():
    """
    Check whether the current `sys.stdout` object exposes a `reconfigure()` method.

    This helper is used to detect if we can safely adjust stdout encoding at runtime.
    On some older Python runtimes (or restricted environments) `reconfigure()` is not
    available, in which case we fall back to environment variables instead.

    Returns
    -------
    bool
        True if `sys.stdout.reconfigure(...)` can be called, False otherwise.
    """
    return hasattr(sys.stdout, "reconfigure")

def _setup_utf8():
    """
    Configure stdout/stderr to use UTF-8 encoding where possible.

    This function:
        - uses `sys.stdout.reconfigure` / `sys.stderr.reconfigure` if available,
        - falls back to setting `PYTHONIOENCODING` to 'utf-8-sig' otherwise.

    It is called once at startup to avoid encoding issues when printing Unicode
    (domain names, arrows, non-ASCII labels) on Windows and mixed environments.
    """
    try:
        if _supports_reconfigure():
            sys.stdout.reconfigure(encoding="utf-8-sig", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8-sig", errors="replace")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8-sig")
    except Exception:
        pass

def _log(level: str, msg: str):
    """
    Lightweight logging helper for the entire pipeline.

    The format is:

        [<LEVEL>] <message>

    where LEVEL is a short marker such as 'INFO', 'WARN', 'ERROR', 'INIT', etc.

    Parameters
    ----------
    level : str
        Textual log level marker.
    msg : str
        Human-readable log message that will be printed to stdout.
    """
    print(f"[{level}] {msg}")

def _dbg(enabled: bool, msg: str):
    """
    Conditional debug logger.

    This is a thin wrapper around `_log` that only prints a message when the
    `enabled` flag is True. It is used for verbose diagnostic output that should
    not be visible in normal production runs.

    Parameters
    ----------
    enabled : bool
        If False, the message is suppressed.
    msg : str
        Debug message to log when enabled.
    """
    if enabled:
        _log("DEBUG", msg)

# ---------- whitelist loader / helper ----------
def load_ip_whitelist(path: str):
    """
    Load a list of IP networks / addresses from a text file.

    Each non-empty, non-comment line is interpreted as an IP subnet or address
    (e.g. '10.0.0.0/8', '192.168.0.1'). The resulting objects are `ip_network`
    instances with `strict=False`.

    Parameters
    ----------
    path : str
        Path to a text file with IP/CIDR entries. If empty/None, returns [].

    Returns
    -------
    list
        List of ipaddress.IPv4Network/IPv6Network objects.
    """
    ipnets = []
    if not path:
        return ipnets
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            for ln in fh:
                s = ln.split("#", 1)[0].strip()
                if not s:
                    continue
                try:
                    ipnets.append(ipaddress.ip_network(s, strict=False))
                except Exception:
                    _log("WARN", f"Invalid whitelist entry (ignored): {s}")
    except Exception as e:
        _log("WARN", f"Failed to read whitelist {path}: {e}")
    return ipnets

def _ip_in_whitelist(ip_str: str, ip_whitelist: list) -> bool:
    """
    Check if an IP string belongs to any of the networks in a whitelist.

    Parameters
    ----------
    ip_str : str
        IP address (IPv4 or IPv6) as text.
    ip_whitelist : list
        List of ipaddress.IPv4Network/IPv6Network objects.

    Returns
    -------
    bool
        True if ip_str is contained in any of the provided networks.
    """
    if not ip_whitelist:
        return False
    try:
        ip = ipaddress.ip_address(ip_str)
        for item in ip_whitelist:
            if isinstance(item, (ipaddress.IPv4Network, ipaddress.IPv6Network)) and ip in item:
                return True
    except Exception:
        pass
    return False

# ---------- derivative features ----------
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute auxiliary, higher-level features on top of the base transformer output.

    The transformer produces low-level flow statistics such as:
        - flow_duration (µs),
        - tot_fwd_pkts / totlen_fwd_pkts,
        - flow_pkts_per_sec,
        - flow_iat_mean, etc.

    This function enriches them with additional derived metrics:
        - fwd_bytes_per_sec, fwd_bytes_per_pkt,
        - pktlen_max_over_mean,
        - consistency = flow_pkts_per_sec * flow_iat_mean (sanity check for units),
        - fwd_pkts_per_sec, payload_sparsity,
        - log_tot_fwd_pkts, log_totlen_fwd_pkts,
        - (optional) backward metrics and ratios (if BWD fields are present).

    All computations are done on a copy of the input DataFrame to avoid side-effects.

    Parameters
    ----------
    df : pandas.DataFrame
        Base features from the transformer.

    Returns
    -------
    pandas.DataFrame
        Copy of df with additional derived columns.
    """
    d = df.copy()
    dur_s = (d["flow_duration"] / 1e6) + NUMERIC_EPS  # flow_duration in µs → seconds for "per_sec"
    d["fwd_bytes_per_sec"] = d["totlen_fwd_pkts"] / dur_s
    d["fwd_bytes_per_pkt"] = d["totlen_fwd_pkts"] / (d["tot_fwd_pkts"] + NUMERIC_EPS)
    d["pktlen_max_over_mean"] = d["fwd_pkt_len_max"] / (d["fwd_pkt_len_mean"] + NUMERIC_EPS)
    # use the copied/derived dataframe 'd' to compute derived features consistently
    d["consistency"] = d["flow_pkts_per_sec"] * d["flow_iat_mean"]
    d["fwd_pkts_per_sec"]  = d["tot_fwd_pkts"]   / dur_s
    d["payload_sparsity"] = 1.0 - (d["fwd_bytes_per_pkt"] / (d["fwd_pkt_len_max"] + NUMERIC_EPS)).clip(0, 1)
    d["log_tot_fwd_pkts"] = np.log1p(d["tot_fwd_pkts"].clip(lower=0))
    d["log_totlen_fwd_pkts"] = np.log1p(d["totlen_fwd_pkts"].clip(lower=0))
    if all(c in d.columns for c in
           ["tot_bwd_pkts", "totlen_bwd_pkts", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_iat_mean"]):
        d["bwd_bytes_per_sec"] = d["totlen_bwd_pkts"] / dur_s
        d["bwd_bytes_per_pkt"] = d["totlen_bwd_pkts"] / (d["tot_bwd_pkts"] + NUMERIC_EPS)
        d["bwd_pkts_per_sec"]  = d["tot_bwd_pkts"]   / dur_s
        d["ratio_bytes_per_sec_bwd_fwd"] = d["bwd_bytes_per_sec"] / (d["fwd_bytes_per_sec"] + NUMERIC_EPS)
        d["ratio_pkts_per_sec_bwd_fwd"] = d["bwd_pkts_per_sec"] / (d["fwd_pkts_per_sec"] + NUMERIC_EPS)
        d["ratio_bytes_per_pkt_bwd_fwd"] = d["bwd_bytes_per_pkt"] / (d["fwd_bytes_per_pkt"] + NUMERIC_EPS)
    return d

# ---------- Keras/TensorFlow compatibility ----------
def _patch_keras_functional_alias():
    """
    Install a compatibility shim for the Keras `Functional` class import path.

    Different Keras / tf.keras versions serialize models with slightly different
    internal module paths, e.g.:

        - keras.src.models.functional.Functional
        - keras.src.engine.functional.Functional
        - keras.engine.functional.Functional

    When loading legacy models, those paths may not exist in the current
    environment. This helper:

        - tries to import `keras.src.models.functional`,
        - if missing, imports `Functional` from a known available location,
        - synthesizes a module `keras.src.models.functional` and registers it
          in `sys.modules`.

    This makes `load_model` robust across Keras versions without changing
    the saved model files.def _install_inputlayer_shim
    """
    try:
        import keras
        try:
            importlib.import_module("keras.src.models.functional")
            return
        except Exception:
            pass
        try:
            from keras.src.engine.functional import Functional as _F
        except Exception:
            from keras.engine.functional import Functional as _F
        mod = types.ModuleType("keras.src.models.functional")
        mod.Functional = _F
        sys.modules["keras.src.models.functional"] = mod
    except Exception:
        pass

def _install_inputlayer_shim():
    """
    Patch Keras InputLayer constructors to accept 'batch_shape' in legacy models.

    Some saved models use `batch_shape` instead of `batch_input_shape` in
    layer configs. This shim translates the argument before calling the
    original constructor.

    The patch is applied both to standalone `keras.layers.InputLayer` and
    to `tf.keras.layers.InputLayer` where available.
    """
    def _patch(cls):
        try:
            if getattr(cls, "__sng_patched__", False):
                return
            _orig = cls.__init__
            def _new(self, *args, **kwargs):
                if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                    bs = kwargs.pop("batch_shape")
                    try:
                        kwargs["batch_input_shape"] = tuple(bs)
                    except Exception:
                        pass
                return _orig(self, *args, **kwargs)
            cls.__init__ = _new
            cls.__sng_patched__ = True
        except Exception:
            pass
    try:
        import keras as k; _patch(k.layers.InputLayer)
    except Exception:
        pass
    try:
        _patch(tf.keras.layers.InputLayer)
    except Exception:
        pass

def _make_custom_objects():
    """
    Build a `custom_objects` dictionary for robust model deserialization.

    Some saved models reference non-standard types (e.g. mixed-precision
    policies or `KerasTensor`), which need to be provided to `load_model`
    via the `custom_objects` parameter.

    This helper currently:
        - exposes `DTypePolicy` mapped to the mixed-precision `Policy` class,
        - provides a dummy fallback `KerasTensor` type for legacy graphs.

    Returns
    -------
    dict
        Mapping of custom symbol names to their corresponding classes/objects
        to be passed into `keras.models.load_model(..., custom_objects=...)`.
    """
    try:
        from tensorflow.keras.mixed_precision import Policy as TFPolicy
        custom = {"DTypePolicy": TFPolicy}
    except Exception:
        custom = {}
    class _KT: pass
    """Fallback dummy for older Keras `KerasTensor` references."""
    custom.setdefault("KerasTensor", _KT)
    return custom

def _load_keras_any(model_path: Path):
    """
    Load a Keras/TensorFlow model from the given path with multiple fallbacks.

    Strategy:
        1. Install Keras compatibility shims.
        2. If path is a directory, treat it as a SavedModel folder.
        3. Otherwise, try `keras.models.load_model` with custom_objects.
        4. If that fails, fall back to `tf.keras.models.load_model`.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to .keras/.h5 file or SavedModel folder.

    Returns
    -------
    tf.keras.Model
        Loaded model instance (uncompiled).
    """
    _patch_keras_functional_alias()
    _install_inputlayer_shim()
    custom_objects = _make_custom_objects()
    if model_path.is_dir():
        return tf.keras.models.load_model(str(model_path), compile=False)
    try:
        import keras as k
        return k.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
    except Exception:
        pass
    return tf.keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)

# ---------- L1 loading and preprocessing ----------
def load_l1_heads_and_sla(l1_input: Path, window: int):
    """
    Load L1 AutoEncoder model, build embedding/reconstruction heads, and read SLA config.

    The function supports two layout patterns:

        1) Direct model file:
            - l1_input is a .keras/.h5 file;
            - `preprocessing_config.json` is searched next to the model or in its parent.

        2) Run folder:
            - l1_input is a directory;
            - model candidates:
                * finetuned/SmartNetGuard_DeepConv1D_AE_finetuned.keras
                * SmartNetGuard_DeepConv1D_AE.keras
                * l1_converted.h5
            - `preprocessing_config.json` is expected inside the run folder.

    The config is expected to contain:
        * window_size
        * feature_names
        * standardizer.mean, standardizer.scale
        * z_clip

    Returns two small models:
        - emb_head  : input → bottleneck embedding
        - recon_head: input → reconstruction output

    Parameters
    ----------
    l1_input : pathlib.Path
        Path to L1 run directory or directly to a model file.
    window : int
        Expected window length (must match config['window_size']).

    Returns
    -------
    (emb_head, recon_head, l1_feats, mean, scale, zlo, zhi)
        emb_head : tf.keras.Model
            Model returning only the embedding (bottleneck) layer.
        recon_head : tf.keras.Model
            Model returning reconstruction output.
        l1_feats : list[str]
            Feature names in the order used by L1.
        mean : np.ndarray
            Standardizer means.
        scale : np.ndarray
            Standardizer scales (stds).
        zlo, zhi : float
            Clipping bounds in Z-space.
    """
    _patch_keras_functional_alias()
    model_path = None
    cfg_path = None
    if l1_input.is_file() and l1_input.suffix.lower() in (".keras", ".h5"):
        model_path = l1_input
        base_dir = model_path.parent if model_path.parent.name.lower() != "finetuned" else model_path.parent.parent
        cand_cfgs = [base_dir / "preprocessing_config.json", model_path.parent / "preprocessing_config.json"]
        cfg_path = next((p for p in cand_cfgs if p.exists()), None)
    elif l1_input.is_dir():
        cands = [
            l1_input / "finetuned" / "SmartNetGuard_DeepConv1D_AE_finetuned.keras",
            l1_input / "SmartNetGuard_DeepConv1D_AE.keras",
            l1_input / "l1_converted.h5",
        ]
        model_path = next((p for p in cands if p.exists()), None)
        cfg_path = l1_input / "preprocessing_config.json"

    _log("DEBUG", f"L1 model_path={model_path}")
    _log("DEBUG", f"L1 cfg_path={cfg_path} exists={cfg_path.exists() if cfg_path else None}")
    assert model_path and cfg_path and cfg_path.exists(), "L1 .keras/.h5 or preprocessing_config.json not found"

    cfg = json.load(open(cfg_path, "r", encoding="utf-8-sig"))
    assert int(cfg["window_size"]) == int(window), f"WINDOW_SIZE for L1 does not match: cfg={cfg['window_size']} vs code={window}"

    l1_feats = cfg["feature_names"]
    mean = np.array(cfg["standardizer"]["mean"], np.float32)
    scale = np.array(cfg["standardizer"]["scale"], np.float32)
    zlo, zhi = [float(x) for x in cfg["z_clip"]]

    l1 = _load_keras_any(model_path)
    try:
        emb_layer = l1.get_layer("Bottleneck_dense")
    except Exception:
        emb_layer = l1.get_layer("En4")
    emb_head = KM.Model(l1.input, emb_layer.output)
    try:
        recon_out = l1.get_layer("Reconstruction").output
    except Exception:
        recon_out = l1.outputs[0]
    recon_head = KM.Model(l1.input, recon_out)
    return emb_head, recon_head, l1_feats, mean, scale, zlo, zhi

def standardize_for_l1(df_base: pd.DataFrame, l1_feats, mean, scale, zlo, zhi, clip_map=None):
    """
    Apply L1 standardisation (Z-space) and optional post-clipping per feature.

    Steps:
        - select columns in `l1_feats` order;
        - cast to float32 ndarray;
        - subtract `mean`, divide by `scale` (with numerical guard);
        - global clipping in [zlo, zhi];
        - optional per-feature clipping via `clip_map`.

    Parameters
    ----------
    df_base : pandas.DataFrame
        Source DataFrame containing all raw/derived features.
    l1_feats : list[str]
        Names of features used by L1 in a fixed order.
    mean : np.ndarray
        Per-feature means from L1 training.
    scale : np.ndarray
        Per-feature scales (std).
    zlo, zhi : float
        Global clipping bounds in Z-space.
    clip_map : dict or None
        Optional dict {feature_name: [low, high]} for additional clipping.

    Returns
    -------
    np.ndarray
        Standardised features with shape (N, F) in Z-space.
    """
    X = df_base[l1_feats].to_numpy(np.float32)
    X = (X - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)
    np.clip(X, zlo, zhi, out=X)
    if clip_map:
        for fname, bounds in clip_map.items():
            if fname in l1_feats and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                j = l1_feats.index(fname)
                lo, hi = float(bounds[0]), float(bounds[1])
                X[:, j] = np.clip(X[:, j], lo, hi, out=X[:, j])
    return X

# ---------- L2 loadout and stats ----------
def load_l2_and_stats(l2_run: Path, use_biases: bool):
    """
    Load L2 classifier model and all associated scaler/clip/calibration stats.

    Layout:
        - l2_run is typically a 'run_...' directory.
        - supported model forms:
            * SavedModel folder (contains 'saved_model.pb');
            * .keras files:
                - SmartNetGuard_L2_DualInput_best.keras
                - SmartNetGuard_L2_DualInput_last.keras

        - scaler_stats.json:
            * "feature_names": ordered feature list for L2,
            * "mean": per-feature mean,
            * "std": per-feature std.

        - clip_stats.json:
            * "per_feature" object with "low"/"high" maps for clipping.

        - calibration.json (optional):
            * "temperature": global temperature scaling,
            * "biases": per-class logit offsets (same order as CANONICAL_CLASS_ORDER).

    Parameters
    ----------
    l2_run : pathlib.Path
        Path to L2 run directory or SavedModel folder.
    use_biases : bool
        Whether to apply biases from calibration.json (if present).

    Returns
    -------
    (l2, feat_names, mu, sd, lo, hi, T, biases)
        l2 : tf.keras.Model
            Dual-input classifier model.
        feat_names : list[str]
            Names of L2 features.
        mu, sd : dict
            Per-feature mean/std.
        lo, hi : dict
            Per-feature clipping ranges.
        T : float or None
            Temperature from calibration.json (if any).
        biases : np.ndarray or None
            Bias vector for logits (if any and use_biases=True).:
    """
    if l2_run.is_dir() and (l2_run / "saved_model.pb").exists():
        l2 = _load_keras_any(l2_run)
    else:
        model_path = None
        for name in ["SmartNetGuard_L2_DualInput_best.keras", "SmartNetGuard_L2_DualInput_last.keras"]:
            p = l2_run / name
            if p.exists():
                model_path = p
                break
        assert model_path, "No L2 .keras found in run folder (and no SavedModel found)"
        l2 = _load_keras_any(model_path)

    scaler_stats = json.load(open(l2_run / "scaler_stats.json", "r", encoding="utf-8-sig"))
    feat_names = scaler_stats["feature_names"]; mu = scaler_stats["mean"]; sd = scaler_stats["std"]
    clip_stats = json.load(open(l2_run / "clip_stats.json", "r", encoding="utf-8-sig"))["per_feature"]
    lo = clip_stats["low"]; hi = clip_stats["high"]

    T = None; biases = None
    cal = l2_run / "calibration.json"
    if cal.exists():
        obj = json.load(open(cal, "r", encoding="utf-8-sig"))
        T = float(obj.get("temperature", 1.0))
        if use_biases:
            biases = obj.get("biases", None)
            if biases is not None:
                biases = np.array(biases, dtype=np.float32)
    return l2, feat_names, mu, sd, lo, hi, T, biases

def clip_std_l2(df: pd.DataFrame, feat_names, lo, hi, mu, sd):
    """
    Apply robust clipping and standardisation for L2 features.

    For each feature:
        - clip to [lo[c], hi[c]],
        - subtract mu[c],
        - divide by sd[c] (or 1.0 if std is 0).

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe with raw (or derived) feature values.
    feat_names : list[str]
        Ordered list of features to process.
    lo, hi : dict
        Per-feature clipping bounds.
    mu, sd : dict
        Per-feature mean / std.

    Returns
    -------
    pandas.DataFrame
        New dataframe containing standardised L2 features.
    """
    d = df.copy()
    for c in feat_names:
        d[c] = d[c].clip(lower=lo[c], upper=hi[c])
        s = sd[c] if sd[c] != 0 else 1.0
        d[c] = (d[c] - mu[c]) / s
    return d

# ---------- inference utilities ----------
def _entropy(p):
    """
    Compute per-row Shannon entropy of probability vectors.

    Parameters
    ----------
    p : np.ndarray
        Array of shape (N, C) with class probabilities.

    Returns
    -------
    np.ndarray
        Entropy for each row (N,).
    """
    ps = np.clip(p, 1e-9, 1.0)
    return -np.sum(ps * np.log(ps), axis=1)

def _pred_with_T_bias(logits, T=None, biases=None):
    """
    Convenience wrapper: apply optional temperature and biases, then softmax.

    This helper is kept for backwards compatibility / possible future use.
    In the current pipeline, calibration is done manually in `run()`.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits array of shape (N, C).
    T : float or None
        Temperature scaling parameter.
    biases : np.ndarray or None
        Per-class bias vector (logit offsets).

    Returns
    -------
    (y, probs)
        y : np.ndarray
            Predicted class indices.
        probs : np.ndarray
            Softmax probabilities after calibration.
    """
    z = logits if T is None else (logits / float(T))
    if biases is not None:
        # subtract the background offset (as in the battle path)
        z = z - biases.reshape(1, -1)
    probs = tf.nn.softmax(z, axis=1).numpy()
    y = np.argmax(probs, axis=1)
    return y, probs

def select_by_recon(mse: np.ndarray, budget: float | None = None, thresh: float | None = None):
    """
    Select windows to pass to L2 based on reconstruction error.

    Modes:
        - `thresh` is set: keep all windows with mse >= thresh.
        - `budget` in (0,1): keep the top-k windows by MSE that correspond
          to a fraction `budget` of the total windows.
        - None: keep all windows.

    Parameters
    ----------
    mse : np.ndarray
        Reconstruction MSE for each window (N,).
    budget : float or None
        Fraction of windows to keep (0-1), highest MSE first.
    thresh : float or None
        Absolute MSE threshold.

    Returns
    -------
    np.ndarray
        Boolean mask of length N indicating which windows are kept.
    """
    n = len(mse)
    if thresh is not None:
        return mse >= float(thresh)
    if budget is not None and 0.0 < budget < 1.0 and n > 0:
        k = max(1, int(round(n * budget)))
        idx = np.argpartition(mse, n - k)[n - k:]
        m = np.zeros(n, dtype=bool); m[idx] = True; return m
    return np.ones(n, dtype=bool)

def zd_gate_mask(probs: np.ndarray, pmax_thr=0.0, entropy_thr=None, margin_thr=None):
    """
    Compute Zero-Day gate mask based on probability distribution properties.

    The gate considers:
        - low maximum probability (pmax),
        - high entropy (uncertainty),
        - low margin between top1 and top2 classes.

    Parameters
    ----------
    probs : np.ndarray
        Probabilities of shape (N, C).
    pmax_thr : float
        Threshold on pmax; below this, window is considered suspicious.
    entropy_thr : float or None
        Threshold on entropy; above this, window is suspicious.
    margin_thr : float or None
        Threshold on (p_top1 - p_top2); below this, window is suspicious.

    Returns
    -------
    (mask, pmax, ent, marg)
        mask : np.ndarray[bool]
            True for windows considered Zero-Day candidates.
        pmax : np.ndarray
            Max probability per row.
        ent : np.ndarray
            Entropy per row.
        marg : np.ndarray
            Margin per row.
    """
    pmax = probs.max(axis=1)
    sortp = np.sort(probs, axis=1)
    marg = sortp[:, -1] - sortp[:, -2]
    ent = _entropy(probs)
    conds = []
    if pmax_thr and pmax_thr > 0: conds.append(pmax < float(pmax_thr))
    if entropy_thr is not None: conds.append(ent > float(entropy_thr))
    if margin_thr is not None: conds.append(marg < float(margin_thr))
    if not conds:
        return np.zeros(len(pmax), dtype=bool), pmax, ent, marg
    mask = np.zeros(len(pmax), dtype=bool)
    for c in conds: mask |= c
    return mask, pmax, ent, marg

# ---------- window generation ----------
def last_windows_per_flow(df_all_feats: pd.DataFrame, feat_cols, window):
    """
    Build a single last window per flow, based on `time_index` ordering.

    For each (src, sport, dst, dport, proto) group:
        - sort by time_index,
        - if the group is shorter than `window`, skip it,
        - otherwise, take the last `window` rows as a single window.

    Parameters
    ----------
    df_all_feats : pandas.DataFrame
        DataFrame with both features and meta columns, including 'time_index'.
    feat_cols : list[str]
        Columns to be used as numeric features.
    window : int
        Length of each window.

    Returns
    -------
    (keys, mats, starts)
        keys : list[tuple]
            Flow keys (src, sport, dst, dport, proto) per window.
        mats : np.ndarray
            Window feature tensor of shape (N, window, F).
        starts : np.ndarray[int64]
            time_index of the first element in each window (here the last window's start).
    """
    keys, mats, starts = [], [], []
    if df_all_feats.empty:
        return keys, np.zeros((0, window, len(feat_cols)), np.float32), np.zeros((0,), np.int64)
    for key, sub in df_all_feats.groupby(["src", "sport", "dst", "dport", "proto"], sort=False):
        sub = sub.sort_values("time_index")
        if len(sub) < window:
            continue
        X = sub[feat_cols].to_numpy(np.float32)
        Xw = X[-window:, :]
        t0 = int(sub.iloc[-window]["time_index"])
        keys.append(key); mats.append(Xw); starts.append(t0)
    if not mats:
        return [], np.zeros((0, window, len(feat_cols)), np.float32), np.zeros((0,), np.int64)
    return keys, np.stack(mats, axis=0), np.array(starts, dtype=np.int64)

def sliding_windows_per_flow(df_all_feats: pd.DataFrame, feat_cols, window):
    """
    Build all sliding windows for each flow, step = 1.

    For each (src, sport, dst, dport, proto) group:
        - sort by time_index,
        - if len(group) < window: skip,
        - otherwise, create windows [0:window], [1:window+1], ..., [n-window:n].

    Parameters
    ----------
    df_all_feats : pandas.DataFrame
        DataFrame with both features and meta columns, including 'time_index'.
    feat_cols : list[str]
        Columns to be used as numeric features.
    window : int
        Length of each window.

    Returns
    -------
    (keys, mats, starts)
        keys : list[tuple]
            Flow keys repeated for each window.
        mats : np.ndarray
            Windows tensor of shape (N, window, F).
        starts : np.ndarray[int64]
            time_index of the first element in each window.
    """
    keys, mats, starts = [], [], []
    if df_all_feats.empty:
        return keys, np.zeros((0, window, len(feat_cols)), np.float32), np.zeros((0,), np.int64)
    for key, sub in df_all_feats.groupby(["src", "sport", "dst", "dport", "proto"], sort=False):
        sub = sub.sort_values("time_index")
        X = sub[feat_cols].to_numpy(np.float32)
        if len(X) < window:
            continue
        idxs = np.arange(0, len(X) - window + 1, dtype=np.int64)
        for i in idxs:
            mats.append(X[i:i + window, :])
            starts.append(int(sub.iloc[i]["time_index"]))
            keys.append(key)
    if not mats:
        return [], np.zeros((0, window, len(feat_cols)), np.float32), np.zeros((0,), np.int64)
    return keys, np.stack(mats, axis=0), np.array(starts, dtype=np.int64)

# ---------- class confirmation policy ----------
def _confirm_http(http_ratio: float, pps_mean: float, mean_consistency: float | None,
                  args, has_tls_signal: bool = False) -> bool:
    """
    Decide whether an event is consistent with HTTP flood semantics.

    Heuristics:
        - http_ratio: fraction of packets aligned with HTTP/TLS ports and metadata.
        - pps_mean: packets per second for the event.
        - mean_consistency: sanity measure (pkts/sec * IAT) range check.
        - has_tls_signal: whether TLS ALPN/SNI suggests HTTP-over-TLS.

    The thresholds depend on `policy_preset`:
        * default:
            - http_event_min_ctx
            - http_min_pps
        * strict:
            - http_event_min_ctx_strict
            - http_min_pps_strict
            - plus additional TLS/consistency requirements
              (at least 2 out of {ctx, TLS, consistency} must be positive).

    Returns
    -------
    bool
        True if HTTP flood is confirmed by context, False otherwise.
    """
    need_ctx = float(args.http_event_min_ctx)
    need_pps = float(args.http_min_pps)
    if args.policy_preset == "strict":
        need_ctx = float(args.http_event_min_ctx_strict)
        need_pps = float(args.http_min_pps_strict)

    base_ok = (http_ratio >= need_ctx) and (pps_mean >= need_pps)
    if not base_ok:
        return False

    if args.policy_preset == "strict":
        signals = 0
        if http_ratio >= need_ctx:
            signals += 1
        if has_tls_signal:
            signals += 1
        if mean_consistency is not None and (
                float(args.consistency_min_strict) <= mean_consistency <= float(args.consistency_max_strict)):
            signals += 1
        return signals >= 2
    else:
        return True

def _confirm_vol(http_ratio: float, pps_mean: float, args) -> bool:
    """
    Decide whether an event behaves like a non-HTTP volumetric flood.

    Criteria:
        - high packet rate (`pps_mean >= vol_event_min_pps`),
        - relatively low HTTP context (`http_ratio <= vol_event_max_http_ctx`).

    Parameters
    ----------
    http_ratio : float
        Share of HTTP/TLS-context packets for the event.
    pps_mean : float
        Mean packets-per-second for the event.
    args : argparse.Namespace
        Parsed CLI arguments with volumetric thresholds.

    Returns
    -------
    bool
        True if volumetric flood semantics are confirmed.
    """
    return (pps_mean >= float(args.vol_event_min_pps)) and (http_ratio <= float(args.vol_event_max_http_ctx))

def _confirm_bot(bot_ratio: float, args) -> bool:
    """
    Decide whether an event is consistent with a 'bot' context.

    The bot context is derived from known service ports that are often abused
    by malware (RDP/SMB/SSH/etc.), and `bot_ratio` is the fraction of
    packets targeting such ports.

    Parameters
    ----------
    bot_ratio : float
        Fraction of packets whose destination port is in the configured
        `bot_service_ports` list.
    args : argparse.Namespace
        CLI arguments with `bot_event_min_ctx` threshold.

    Returns
    -------
    bool
        True if the proportion of bot-context packets is high enough.
    """
    return bot_ratio >= float(args.bot_event_min_ctx)

def _confirm_ps(uniq_dport: int, uniq_dst: int, args) -> bool:
    """
    Decide whether an event exhibits portscan-like behaviour.

    Two canonical scenarios are considered:
        - Horizontal scan: many distinct destination ports on one/few hosts.
        - Vertical scan  : many distinct target hosts.

    Parameters
    ----------
    uniq_dport : int
        Number of distinct destination ports observed for the event.
    uniq_dst : int
        Number of distinct destination IPs observed for the event.
    args : argparse.Namespace
        CLI arguments with `ps_event_min_uniq_dports` and
        `ps_event_min_uniq_targets` thresholds.

    Returns
    -------
    bool
        True if either diversity threshold is satisfied.
    """
    return (uniq_dport >= int(args.ps_event_min_uniq_dports)) or (uniq_dst >= int(args.ps_event_min_uniq_targets))

def _route_policy(pred_label: str,
                  http_ratio: float,
                  pps_mean: float,
                  uniq_dport: int,
                  uniq_dst: int,
                  bot_ratio: float,
                  args,
                  mean_consistency: float | None = None,
                  has_tls_signal: bool = False) -> tuple[str, int, int, str]:
    """
    Central policy router that reconciles ML predictions with protocol semantics.

    Inputs:
        - pred_label: L2 predicted class for the event.
        - http_ratio, pps_mean: HTTP and volumetric context.
        - uniq_dport, uniq_dst: portscan context (diversity of ports/targets).
        - bot_ratio: bot-specific port context.
        - mean_consistency: sanity measure (optional).
        - has_tls_signal: TLS-based HTTP context flag.

    Behaviour:
        - For HTTP/VOL:
            * confirm HTTP or VOL if their respective conditions pass;
            * if HTTP is predicted but VOL confirmed → relabel to volumetric_flood;
            * if VOL is predicted but HTTP confirmed → relabel to http_flood;
            * otherwise apply fallback (mark/zd/relabel_to_unknown).

        - For BOT:
            * confirm if bot_ratio is high enough, otherwise fallback.

        - For PORTSCAN:
            * confirm if portscan diversity is high enough, otherwise fallback.

    Returns
    -------
    (final_label, ctx_mismatch, force_zd, reason)
        final_label : str
            Effective label after policy (could be unchanged or relabelled).
        ctx_mismatch : int
            1 if semantics disagreed with ML prediction, 0 otherwise.
        force_zd : int
            1 if event should be forced into Zero-Day bucket (per fallback).
        reason : str
            Textual tag for tracing decisions ('confirm:http', 'fallback->vol', ...).
    """
    ctx_mismatch = 0
    force_zd = 0
    final_label = pred_label
    reason = "ok"

    if pred_label == "http_flood":
        ok_http = _confirm_http(http_ratio, pps_mean, mean_consistency, args, has_tls_signal=has_tls_signal)
        if ok_http:
            return final_label, ctx_mismatch, force_zd, "confirm:http"
        ctx_mismatch = 1
        if _confirm_vol(http_ratio, pps_mean, args):
            return "volumetric_flood", ctx_mismatch, force_zd, "fallback->vol"
        fa = args.fallback_action
        if fa == "zd":
            force_zd = 1; reason = "fallback->zd"
        elif fa == "relabel_to_unknown":
            final_label = args.unknown_label; reason = "fallback->unknown"
        else:
            reason = "fallback->mark"
        return final_label, ctx_mismatch, force_zd, reason

    if pred_label == "volumetric_flood":
        ok_vol = _confirm_vol(http_ratio, pps_mean, args)
        if ok_vol:
            return final_label, ctx_mismatch, force_zd, "confirm:vol"
        ctx_mismatch = 1
        if _confirm_http(http_ratio, pps_mean, mean_consistency, args, has_tls_signal=has_tls_signal):
            return "http_flood", ctx_mismatch, force_zd, "fallback->http"
        fa = args.fallback_action
        if fa == "zd":
            force_zd = 1; reason = "fallback->zd"
        elif fa == "relabel_to_unknown":
            final_label = args.unknown_label; reason = "fallback->unknown"
        else:
            reason = "fallback->mark"
        return final_label, ctx_mismatch, force_zd, reason

    if pred_label == "bot":
        ok_bot = _confirm_bot(bot_ratio, args)
        if ok_bot:
            return final_label, ctx_mismatch, force_zd, "confirm:bot"
        ctx_mismatch = 1
        fa = args.fallback_action
        if fa == "zd":
            force_zd = 1; reason = "fallback->zd"
        elif fa == "relabel_to_unknown":
            final_label = args.unknown_label; reason = "fallback->unknown"
        else:
            reason = "fallback->mark"
        return final_label, ctx_mismatch, force_zd, reason

    if pred_label == "portscan":
        ok_ps = _confirm_ps(uniq_dport, uniq_dst, args)
        if ok_ps:
            return final_label, ctx_mismatch, force_zd, "confirm:ps"
        ctx_mismatch = 1
        fa = args.fallback_action
        if fa == "zd":
            force_zd = 1; reason = "fallback->zd"
        elif fa == "relabel_to_unknown":
            final_label = args.unknown_label; reason = "fallback->unknown"
        else:
            reason = "fallback->mark"
        return final_label, ctx_mismatch, force_zd, reason

    return final_label, ctx_mismatch, force_zd, "noop"

# ---------- splicing of events ----------
class EventStitcher:
    """
    Accumulate window-level detections into higher-level events per flow.

    A "window" is identified by:
        - flow key: (src, sport, dst, dport, proto),
        - predicted label,
        - ZD flag,
        - start_idx (time_index aligned),
        - timestamp and scores.

    The stitcher maintains an `active` dict keyed by (flow, label, zd),
    and grows contiguous sequences of windows as long as the gap between
    window starts does not exceed `max_gap+1`.

    When a gap is too large or flow/label/zd changes, the previous event
    is closed and yielded (if length >= min_len).

    This converts noisy per-window detections into more stable "events"
    suitable for policy and alerting.
    """
    def __init__(self, max_gap=2, min_len=2):
        """
        Initialize a new EventStitcher instance.

        Parameters
        ----------
        max_gap : int, optional
            Maximum allowed difference between consecutive window
            `start_idx` values to treat them as contiguous.
        min_len : int, optional
            Minimum number of windows an event must contain before it can
            be emitted (short bursts can be discarded as noise).
        """
        self.max_gap = int(max_gap)
        self.min_len = int(min_len)
        self.active = {}

    def update(self, items):
        """
        Update active events with a list of new window items.

        Each item is expected to have keys:
            - flow: flow tuple
            - label: predicted class
            - zd: bool/int
            - start_idx: window starting index (time_index)
            - ts: timestamp
            - pmax: confidence
            - ctx: dict with extra signals (e.g., recon_mse)

        Parameters
        ----------
        items : list[dict]
            Window-level results for the current batch.

        Returns
        -------
        list[dict]
            List of finished events that reached min_len and were closed
            due to gaps or flow/label/zd changes.
        """
        finished = []
        for it in items:
            k = (it["flow"], it["label"], bool(it["zd"]))
            st = self.active.get(k)
            if st is None:
                to_close = [kk for kk in list(self.active.keys()) if kk[0] == it["flow"] and kk != k]
                for kk in to_close:
                    finished.append(self._finish(kk))

                ctx = it.get("ctx") or {}
                has_mse = isinstance(ctx, dict) and ("recon_mse" in ctx)

                self.active[k] = {
                    "start_idx": it["start_idx"],
                    "end_idx": it["start_idx"],
                    "count": 1,
                    "first_ts": it["ts"],
                    "last_ts": it["ts"],
                    "pmax_max": float(it["pmax"]),
                    "pmax_sum": float(it["pmax"]),
                    "ctx": it.get("ctx", None),
                    "recon_mse_sum": float(ctx.get("recon_mse", 0.0)),
                    "recon_mse_cnt": 1 if has_mse else 0,
                }
            else:
                gap_ok = (it["start_idx"] - st["end_idx"]) <= (self.max_gap + 1)
                if gap_ok:
                    st["end_idx"] = it["start_idx"]
                    st["count"] += 1
                    st["last_ts"] = it["ts"]
                    st["pmax_max"] = max(st["pmax_max"], float(it["pmax"]))
                    st["pmax_sum"] += float(it["pmax"])
                    if it.get("ctx") and "recon_mse" in it["ctx"]:
                        st["recon_mse_sum"] += float(it["ctx"]["recon_mse"])
                        st["recon_mse_cnt"] += 1
                else:
                    # gap greater than max_gap: close the old event and start a new one
                    finished.append(self._finish(k))

                    ctx = it.get("ctx") or {}
                    has_mse = isinstance(ctx, dict) and ("recon_mse" in ctx)

                    self.active[k] = {
                        "start_idx": it["start_idx"],
                        "end_idx": it["start_idx"],
                        "count": 1,
                        "first_ts": it["ts"],
                        "last_ts": it["ts"],
                        "pmax_max": float(it["pmax"]),
                        "pmax_sum": float(it["pmax"]),
                        "ctx": it.get("ctx", None),
                        "recon_mse_sum": float(ctx.get("recon_mse", 0.0)),
                        "recon_mse_cnt": 1 if has_mse else 0,
                    }

        return [e for e in finished if e["count"] >= self.min_len]

    def flush(self):
        """
          Close and emit all currently active events.

        This should be called at the end of a PCAP or sniffer run to make
        sure no partial event is left hanging in memory.

        Returns
        -------
        list[dict]
            List of all finished events that meet the `min_len` requirement.
        """
        out = [self._finish(k) for k in list(self.active.keys())]
        self.active.clear()
        return [e for e in out if e["count"] >= self.min_len]

    def _finish(self, k):
        """
        Internal helper: finalize and remove an event from `active`.

        The event metrics are aggregated as follows:
            - `count` : number of windows,
            - `start_idx` / `end_idx` : index range,
            - `first_ts` / `last_ts` : wall-clock span,
            - `pmax_max` / `pmax_mean` : confidence summary,
            - `recon_mse_mean` : average reconstruction error if available.

        Parameters
        ----------
        k : tuple
            Key of the event in `active`, of the form
            (flow_key, label, zd_flag).

        Returns
        -------
        dict
            Completed event record suitable for further policy/OOD processing.
        """
        st = self.active.pop(k)
        flow, label, zd = k
        recon_mse_mean = float(st["recon_mse_sum"] / max(1, st["recon_mse_cnt"])) if "recon_mse_sum" in st else 0.0
        return {
            "flow": flow,
            "label": label,
            "zd": int(zd),
            "count": int(st["count"]),
            "start_idx": int(st["start_idx"]),
            "end_idx": int(st["end_idx"]),
            "first_ts": float(st["first_ts"]),
            "last_ts": float(st["last_ts"]),
            "pmax_max": float(st["pmax_max"]),
            "pmax_mean": float(st["pmax_sum"] / max(1, st["count"])),
            "recon_mse_mean": recon_mse_mean,
        }

def _save_events(ev_list, args):
    """
    Persist a list of event dictionaries to disk, as CSV or Parquet.

    The output filename is generated as:

        events_<unix_timestamp>.csv  or  events_<unix_timestamp>.parquet

    depending on `args.dump`. Parquet requires `pyarrow`; if it is not
    available, the function falls back to CSV and logs a warning.

    Parameters
    ----------
    ev_list : list[dict]
        Events produced by the EventStitcher and enriched by policy/OOD.
    args : argparse.Namespace
        Parsed CLI arguments; uses `args.out` (output directory) and
        `args.dump` (format).
    """
    if not ev_list:
        return
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(ev_list)
    base = f"events_{int(time.time())}"
    out_path = Path(args.out) / (base + (".parquet" if args.dump == "parquet" else ".csv"))
    if args.dump == "parquet":
        try:
            import pyarrow  # noqa
            df.to_parquet(out_path, index=False)
        except Exception:
            _log("WARN", "pyarrow is unavailable, saving as CSV.")
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    _log("SAVE", f"events -> {out_path}")


def _send_email_alert(event: dict, args) -> bool:
    """
    Send a concise email alert for a single high-level event.

    The email is designed to be human-readable and suitable for on-call
    notifications. It contains:

        - Subject: event label and UTC timestamp of `first_ts`,
        - Body: flow tuple, label/original label, Zero-Day flag, time span,
          basic metrics (pmax, recon MSE, context ratios),
        - Optional JSON attachment with the full event payload.

    SMTP configuration is taken from CLI arguments:

        - `email_smtp_host`, `email_smtp_port`,
        - `email_user`, `email_password_env`, `email_use_ssl`,
        - `email_from`, `email_to`.

    Parameters
    ----------
    event : dict
        Final event record after policy and OOD processing.
    args : argparse.Namespace
        CLI arguments that hold SMTP and addressing configuration.

    Returns
    -------
    bool
        True if the message was sent successfully, False on any error.
    """
    if args.email_alerts != "on":
        return False
    try:
        # collecting addresses
        to_addrs = args.email_to if isinstance(args.email_to, (list, tuple)) else [args.email_to]
        if not to_addrs:
            _log("WARN", "email_alerts are enabled, but --email_to is not specified.")
            return False

        # From
        frm = args.email_from or (args.email_user if args.email_user else "smartnetguard@localhost")

        # Subject / body
        subj = f"[SmartNetGuard] {event.get('label','unknown')} @ {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(event.get('first_ts', time.time())))}"
        body_lines = [
            f"Label: {event.get('label')}",
            f"Original label: {event.get('label_orig', '')}",
            f"ZD flag: {event.get('zd')}",
            f"Flow: {event.get('flow')}",
            f"Time span: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(event.get('first_ts', time.time())))} - {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(event.get('last_ts', time.time())))}",
            f"Windows: {event.get('count')}",
            f"pmax_mean: {event.get('pmax_mean'):.4f}",
            f"recon_mse_mean: {event.get('recon_mse_mean', 0):.6g}",
            f"ctx_http_ratio: {event.get('ctx_http_ratio', 0):.3f}",
            f"ctx_bot_ratio: {event.get('ctx_bot_ratio', 0):.3f}",
            f"pps_mean: {event.get('pps_mean', 0):.2f}",
            f"ps_uniq_dport: {event.get('ps_uniq_dport', 0)}",
            f"ps_uniq_dst: {event.get('ps_uniq_dst', 0)}",
            f"policy_reason: {event.get('policy_reason','')}",
            "",
            "Details in the attached CSV/logs on the collector side (if necessary)."
        ]
        body = "\n".join(body_lines)

        msg = EmailMessage()
        msg["Subject"] = subj
        msg["From"] = frm
        msg["To"] = ", ".join(to_addrs)
        msg.set_content(body)

        # Additionally, you can attach a JSON event
        try:
            json_blob = json.dumps(event, ensure_ascii=False, indent=2)
            msg.add_attachment(json_blob.encode("utf-8"), maintype="application", subtype="json", filename="event.json")
        except Exception:
            pass

        # connection to SMTP
        smtp_host = args.email_smtp_host
        smtp_port = int(args.email_smtp_port)
        password = os.getenv(args.email_password_env, None)

        if args.email_use_ssl:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
            server.ehlo()
            server.starttls()
            server.ehlo()

        if args.email_user and password:
            server.login(args.email_user, password)

        server.send_message(msg)
        server.quit()
        _log("ALERT", f"Sent email for event {event.get('flow')}, label={event.get('label')}")
        return True
    except Exception as ex:
        _log("WARN", f"Failed to send email alert: {ex}")
        return False

def _ensure_logits_meta_once(args, class_order, T, biases):
    """
    Persist a small JSON metadata file describing logit calibration.

    This writes `<out>/logits_meta.json` once per run, with the fields:

        {
            "class_order": [...],
            "temperature": <float or null>,
            "biases": [list or null]
        }

    This is useful for later offline analysis of logits CSV dumps, ensuring
    that consumers know the class order and calibration used during inference.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with `args.out`.
    class_order : list[str]
        Canonical class order used by the model.
    T : float or None
        Temperature scaling parameter (if used).
    biases : np.ndarray or None
        Bias vector applied to logits (if used).
    """
    try:
        os.makedirs(args.out, exist_ok=True)
        meta_path = Path(args.out) / "logits_meta.json"
        if not meta_path.exists():
            obj = {
                "class_order": list(class_order),
                "temperature": None if T is None else float(T),
                "biases": None if biases is None else [float(x) for x in np.ravel(biases).tolist()],
            }
            with open(meta_path, "w", encoding="utf-8-sig") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            _log("SAVE", f"logits meta -> {meta_path}")
    except Exception as e:
        _log("WARN", f"Cannot save logits_meta.json: {e}")

def _save_logits_rows(rows, args):
    """
    Persist a lightweight CSV dump of raw and calibrated logits.

    Each row corresponds to a single window with:
        - flow identifiers,
        - raw logits and calibrated logits per class,
        - probabilities (`prob_*` and `prob_policy_*`),
        - summary scores such as `p_max`, `pmax_policy`, margins, etc.

    The file name is:

        <dump_logits_prefix><unix_timestamp>.csv

    Parameters
    ----------
    rows : list[dict]
        Logit records for a given microbatch.
    args : argparse.Namespace
        Parsed CLI arguments with `args.out` and `args.dump_logits_prefix`.
    """
    if not rows:
        return
    try:
        os.makedirs(args.out, exist_ok=True)
        ts = int(time.time())
        base = f"{args.dump_logits_prefix}{ts}"
        out_path = Path(args.out) / f"{base}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        _log("SAVE", f"logits -> {out_path}")
    except Exception as e:
        _log("WARN", f"Cannot save logits csv: {e}")

# ---------- context functions ----------
def _ctx_http_mask(df_slice: pd.DataFrame, http_ports: set, ip_whitelist: list = None) -> np.ndarray:
    """
    Build a boolean mask marking rows that are considered HTTP/TLS context.

    A row is treated as HTTP-related if:
        - protocol is TCP, AND
        - source or destination port is in `http_ports`, OR
        - TLS metadata indicates HTTP-over-TLS:
            * ALPN includes 'h2' or 'http/1.1',
            * SNI is non-empty.

    Additionally, if `ip_whitelist` is provided, rows where either endpoint
    is inside a trusted subnet are suppressed from port-based HTTP detection.
    This avoids flagging internal monitoring/admin traffic as HTTP attacks.

    Parameters
    ----------
    df_slice : pandas.DataFrame
        Slice of rows for a single flow / window.
    http_ports : set[int]
        Set of ports considered HTTP service ports (80, 443, etc.).
    ip_whitelist : list or None
        List of networks from `load_ip_whitelist` to de-emphasize.

    Returns
    -------
    np.ndarray
        Boolean array of length len(df_slice) marking HTTP-related rows.
    """
    if ip_whitelist is None:
        ip_whitelist = []

    # Strong typing and safe data sanitization
    proto = pd.to_numeric(df_slice.get("proto", 0), errors="coerce").fillna(0).astype(int)
    sport = pd.to_numeric(df_slice.get("sport", 0), errors="coerce").fillna(0).astype(int)
    dport = pd.to_numeric(df_slice.get("dport", 0), errors="coerce").fillna(0).astype(int)

    by_port = ((proto == 6) & (dport.isin(http_ports) | sport.isin(http_ports)))

    # Optional whitelist (CIDR/addresses) to suppress HTTP spoofing from trusted IPs
    if ip_whitelist:
        try:
            srcs = df_slice.get("src", "").astype(str)
            dsts = df_slice.get("dst", "").astype(str)
            wl_mask = srcs.apply(lambda x: _ip_in_whitelist(x, ip_whitelist)) | dsts.apply(
                lambda x: _ip_in_whitelist(x, ip_whitelist))
            if wl_mask.any():
                _log("POLICY", f"Whitelist suppressed {wl_mask.sum()} rows in a window (of {len(wl_mask)})")
            by_port = by_port & (~wl_mask)
        except Exception:
            pass

    # TLS Signals (ALPN/SNI) - Safely Handling NaNs
    is_tls = df_slice.get("is_tls", 0)
    if not isinstance(is_tls, pd.Series):
        is_tls = pd.Series([is_tls] * len(df_slice), index=df_slice.index)
    is_tls = pd.to_numeric(is_tls, errors="coerce").fillna(0).astype(bool)

    tls_alpn = df_slice.get("tls_alpn", None)
    if isinstance(tls_alpn, pd.Series):
        has_alpn_http = tls_alpn.fillna("").astype(str).str.contains(_ALPN_HTTP_REGEX, regex=True, na=False)
    else:
        has_alpn_http = pd.Series([False] * len(df_slice), index=df_slice.index)

    tls_sni = df_slice.get("tls_sni", None)
    if isinstance(tls_sni, pd.Series):
        has_sni = tls_sni.fillna("").astype(str) != ""
    else:
        has_sni = pd.Series([False] * len(df_slice), index=df_slice.index)

    by_tls = is_tls & (has_alpn_http | has_sni)

    return (by_port | by_tls).to_numpy()


def _ctx_bot_mask(df_slice: pd.DataFrame, bot_ports: set) -> np.ndarray:
    """
    Build a boolean mask for rows that contribute to 'bot' context.

    A row is treated as bot-related if:
        - protocol is TCP,
        - destination port is in `bot_ports` (e.g. SMB, RDP, SSH).

    Parameters
    ----------
    df_slice : pandas.DataFrame
        Slice of rows for a single flow / window.
    bot_ports : set[int]
        Set of ports associated with typical malware C2 / lateral movement.

    Returns
    -------
    np.ndarray
        Boolean array, True where bot context is detected.
    """
    return ((df_slice["proto"] == 6) & (df_slice["dport"].isin(bot_ports))).to_numpy()

def _pps_mean(df_slice: pd.DataFrame) -> float:
    """
    Compute the average packets-per-second (pps) for a slice.

    The transformer is expected to provide `flow_pkts_per_sec`, which is
    averaged over all rows in the slice.

    Parameters
    ----------
    df_slice : pandas.DataFrame
        Window or event slice with a `flow_pkts_per_sec` column.

    Returns
    -------
    float
        Mean packets-per-second over the slice, or 0.0 if empty..
    """
    v = df_slice["flow_pkts_per_sec"].to_numpy(dtype=np.float32)
    return float(np.mean(v)) if len(v) else 0.0

def _uniq_counts_for_ps(df_slice: pd.DataFrame):
    """
    Compute diversity metrics used for portscan detection.

    Parameters
    ----------
    df_slice : pandas.DataFrame
        Slice of flow rows with at least `dport` and `dst` columns.

    Returns
    -------
    tuple
        (uniq_dport, uniq_dst) where:
            uniq_dport : int
                Number of distinct destination ports.
            uniq_dst : int
                Number of distinct destination IPs.
    """
    return int(df_slice["dport"].nunique()), int(df_slice["dst"].nunique())

# ========================= Main runner =========================
def run(args):
    """
    Main orchestrator for SmartNetGuard inference.

    This function:

        1) Sets up UTF-8 logging and TensorFlow threading/GPU behaviour.
        2) Loads L1 AutoEncoder heads and L2 classifier (+ calibration).
        3) Validates L2 feature set vs model input; can auto-align (drop features)
           if `--l2_feat_autofix` is given.
        4) Instantiates FeatureTransformer and EventStitcher.
        5) Defines inner utilities:
            - _emit: JSON streaming helper for GUI integrations.
            - finalize_and_dump: flush stitched events buffer, apply policy/OOD,
              print and dump events, send email alerts if enabled.
            - process_df: core logic that runs on a single DataFrame snapshot
              (from transformer), derives features, prepares L1/L2 inputs,
              performs L1+L2 inference, applies pre-L2 filter, ZDGate,
              OOD-lite (per-window pre-score), event stitching, context-aware policy,
              and dumps multiple views (windows, live logs, confident windows).
        6) Depending on `--source`:
            - 'sniffer': attach to a live interface using Scapy, accumulate packets
              into transformer and periodically call process_df + finalize_and_dump.
            - 'pcap': iterate packets from .pcap/.pcapng (dpkt), feed transformer,
              periodically flush to process_df, then finalize_and_dump at the end.

    The logic is intentionally monolithic so that:
        - the full end-to-end behaviour is visible in one place,
        - debugging sessions can follow the data from packet → event.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (see `main()` for details).
    """
    _setup_utf8()

    def _emit(kind: str, payload: dict):
        """
        Stream a JSON object to stdout to be consumed by a GUI or external tool.

        The format is:
            {"kind": kind, ...payload_fields...}

        Controlled by `--stream_json`. If not enabled, this is a no-op.
        """
        if not getattr(args, "stream_json", False):
            return
        try:
            import sys, json as _json
            sys.stdout.write(_json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(args.tf_intra))
        tf.config.threading.set_inter_op_parallelism_threads(int(args.tf_inter))
    except Exception:
        pass
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    _dbg(True, f"CUDA devices={len(gpus)} (mem_growth on)")

    if args.dump == "parquet":
        try:
            import pyarrow  # noqa
        except Exception:
            _log("WARN", "Parquet needs 'pyarrow'. Switching to CSV.")
            args.dump = "csv"

    # 1) models and stats
    emb_head, recon_head, l1_feats, l1_mean, l1_scale, zlo, zhi = load_l1_heads_and_sla(Path(args.l1_run), WINDOW_SIZE)
    l2, l2_feats, mu, sd, lo, hi, T_loaded, biases_loaded = load_l2_and_stats(
        Path(args.l2_run),
        use_biases=args.use_calib_biases
    )

    # --- Hard check: L2 must NOT contain BWD/ratio features
    _bwd_block = {
        "tot_bwd_pkts", "totlen_bwd_pkts", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_iat_mean",
        "bwd_pkts_per_sec", "bwd_bytes_per_sec",
        "ratio_bytes_per_sec_bwd_fwd", "ratio_pkts_per_sec_bwd_fwd", "ratio_bytes_per_pkt_bwd_fwd"
    }
    _bad = sorted(list(set(l2_feats) & _bwd_block))
    if _bad:
        raise RuntimeError(f"[L2-FEATS] scaler_stats contains BWD/ratio features (disabled for this model): {_bad}")
    _log("CHECK", f"L2 feature_names({len(l2_feats)}): {list(l2_feats)}")

    # Checking the completeness of scaler/clip stats to avoid KeyErrors later
    _missing_stats = [c for c in l2_feats if c not in mu or c not in sd or c not in lo or c not in hi]
    if _missing_stats:
        raise RuntimeError(f"[L2-FEATS] There are no limits/stats in scaler_stats/clip_stats for: {_missing_stats}")

    # --- overriding calibration via CLI ---
    T = T_loaded
    biases = biases_loaded

    if args.no_calib:
        T = None
        biases = None
        _log("CALIB", "Calibration disabled by --no_calib")
    else:
        if args.calib_T is not None:
            try:
                T = float(args.calib_T)
            except Exception:
                pass
        if args.calib_bias_json:
            try:
                with open(args.calib_bias_json, "r", encoding="utf-8-sig") as f:
                    bias_map = json.load(f)
                biases = np.array(
                    [float(bias_map.get(f"logit_{cls}", 0.0)) for cls in CANONICAL_CLASS_ORDER],
                    dtype=np.float32
                )
                _log("CALIB", f"Loaded bias from {args.calib_bias_json}")
            except Exception as e:
                _log("WARN", f"Cannot load calib_bias_json: {e}; continue without external bias")

    # Checking the consistency of the number of features
    # Let's compare the expected number of features by the model and those in scaler_stats
    try:
        expected_n = int(l2.inputs[0].shape[-1]) if hasattr(l2, "inputs") and l2.inputs[0].shape[
            -1] is not None else len(l2_feats)
    except Exception:
        expected_n = len(l2_feats)
    orig_n = len(l2_feats)
    dropped = []
    if args.l2_feat_autofix and orig_n != expected_n:
        _log("INFO", f"[L2-FEATS] scaler_stats={orig_n} features, model expects {expected_n}. Aligning...")
        drop_priority = [
            "flow_bytes_total", "flow_pkts_total",
            "payload_sparsity", "consistency",
            "ratio_bytes_per_pkt_bwd_fwd", "ratio_bytes_per_sec_bwd_fwd", "ratio_pkts_per_sec_bwd_fwd",
            "bwd_iat_mean", "bwd_pkt_len_mean", "bwd_pkt_len_max", "totlen_bwd_pkts", "tot_bwd_pkts",
            "log_totlen_fwd_pkts", "log_tot_fwd_pkts",
            "fwd_pkt_len_max", "fwd_pkt_len_mean",
        ]
        l2_feats = list(l2_feats)
        for cand in drop_priority:
            if len(l2_feats) <= expected_n:
                break
            if cand in l2_feats:
                l2_feats.remove(cand);
                dropped.append(cand)
        while len(l2_feats) > expected_n:
            dropped.append(l2_feats.pop())
        _log("INFO", f"[L2-FEATS] dropped={dropped} | final_count={len(l2_feats)}")
    elif orig_n != expected_n:
        # The default is to stop immediately to avoid masking problems.
        raise RuntimeError(f"[L2-FEATS] scaler_stats={orig_n} vs model={expected_n}. "
                           f"Cannot continue. If this is expected, explicitly specify --l2_feat_autofix.")

    _log("INIT",
         f"L1 feats={len(l1_feats)} | L2 feats={len(l2_feats)} | Tcal={T} | biases={'on' if biases is not None else 'off'}")
    _dbg(True, f"L2 feat list: {list(l2_feats)}")
    _log("INIT", f"IAT unit expected by training: {args.iat_unit} (post-rescale is disabled)")

    _ensure_logits_meta_once(args, CANONICAL_CLASS_ORDER, T, biases)

    # 2) transformer + stitcher + counters
    try:
        transformer = FeatureTransformer(time_bin_sec=args.bin, iat_unit=args.iat_unit)
    except TypeError:
        transformer = FeatureTransformer(time_bin_sec=args.bin)
    stitcher = EventStitcher(max_gap=args.stitch_max_gap, min_len=args.stitch_min_len)
    events_buffer = []
    stats = defaultdict(int)
    last_stats_print = time.time()
    stats_period = max(5.0, args.flush_sec if args.source == "sniffer" else 5.0)

    http_ports = set(int(p) for p in args.http_ports)
    bot_ports = set(int(p) for p in args.bot_service_ports)
    ip_whitelist = load_ip_whitelist(args.http_whitelist_file) if getattr(args, "http_whitelist_file", None) else []

    _log("CTXCONF",
         f"policy={args.policy_preset} | http_ports={sorted(http_ports)} | bot_ports={sorted(bot_ports)} "
         f"| http_min_ctx={args.http_event_min_ctx} (strict={args.http_event_min_ctx_strict}) "
         f"http_min_pps={args.http_min_pps} (strict={args.http_min_pps_strict}) "
         f"| vol_min_pps={args.vol_event_min_pps} vol_max_http_ctx={args.vol_event_max_http_ctx} "
         f"| ps_min(dports,targets)=({args.ps_event_min_uniq_dports},{args.ps_event_min_uniq_targets}) "
         f"| actions: http={args.http_event_action} vol={args.vol_event_action} bot={args.bot_event_action} ps={args.ps_event_action} "
         f"| fallback={args.fallback_action} | policy_debug={'ON' if args.policy_debug else 'OFF'} | whitelist_entries={len(ip_whitelist)}")

    recent_flow_groups: dict = {}

    def finalize_and_dump():
        """
        Flush all active events from the EventStitcher, apply policy and OOD-lite,
        then print and persist the resulting events.

        Processing steps:
            1) Call `stitcher.flush()` to obtain completed events.
            2) For each event:
                - re-construct a representative WINDOW_SIZE slice from
                  `recent_flow_groups`,
                - compute HTTP/bot/portscan context metrics,
                - run `_route_policy` to reconcile ML label with semantics,
                - compute event-level OOD-lite score,
                - optionally relabel to `unknown_label` or force ZD.
            3) Print `[EVENT]` lines to stdout for human inspection.
            4) Optionally emit 'event' JSON objects for UI (`--stream_json`).
            5) Optionally send email alerts (`--email_alerts on`).
            6) Append to `events_buffer` and persist to disk (CSV/Parquet)
               when `args.dump` is enabled.

        This function is typically called at the end of a run or when an
        explicit flush of in-flight events is required.
        """
        post = stitcher.flush()
        if not post:
            return
        post_events = []
        for e in post:
            key = e["flow"]
            sub = recent_flow_groups.get(key)
            if sub is None or len(sub) < WINDOW_SIZE:
                http_ratio = bot_ratio = 0.0; pps_m = 0.0; u_dport = u_dst = 0; mean_cons = 0.0; tls_present = False
            else:
                pos_map = {int(t): int(p) for p, t in enumerate(sub["time_index"].values.astype(np.int64))}
                pos = pos_map.get(int(e["start_idx"]), None)
                if pos is None or (pos + WINDOW_SIZE > len(sub)):
                    http_ratio = bot_ratio = 0.0; pps_m = 0.0; u_dport = u_dst = 0; mean_cons = 0.0; tls_present = False
                else:
                    slice_df = sub.iloc[pos:pos + WINDOW_SIZE]
                    http_ratio = float(np.mean(_ctx_http_mask(slice_df, http_ports, ip_whitelist)))
                    bot_ratio = float(np.mean(_ctx_bot_mask(slice_df, bot_ports)))
                    pps_m = _pps_mean(slice_df)
                    u_dport, u_dst = _uniq_counts_for_ps(slice_df)
                    if "consistency" in slice_df.columns:
                        mean_cons = float(slice_df["consistency"].mean())
                    else:
                        mean_cons = float((slice_df["flow_pkts_per_sec"] * slice_df["flow_iat_mean"]).mean())
                    tls_present = False
                    if "is_tls" in slice_df.columns:
                        is_tls_s = slice_df["is_tls"].fillna(0).astype(bool)
                        alpn_s = slice_df.get("tls_alpn", pd.Series([""] * len(slice_df))).fillna("").astype(str)
                        sni_s = slice_df.get("tls_sni", pd.Series([""] * len(slice_df))).fillna("").astype(str)
                        tls_present = bool((is_tls_s & (alpn_s.str.contains(_ALPN_HTTP_REGEX, regex=True, na=False) | (sni_s != ""))).any())

            final_label, ctx_mismatch, force_zd, reason = _route_policy(
                e["label"], http_ratio, pps_m, u_dport, u_dst, bot_ratio, args,
                mean_consistency=mean_cons, has_tls_signal=tls_present
            )

            pmax_ev = float(e["pmax_mean"])
            mse_norm_ev = float(e.get("recon_mse_mean", args.ood_mse_median)) / max(args.ood_mse_median, 1e-9)
            ood_score_ev = (args.ood_alpha * mse_norm_ev + args.ood_beta * (1.0 - pmax_ev) + args.ood_gamma * float(ctx_mismatch))

            if args.ood == "on" and (ood_score_ev > float(args.ood_thresh)):
                final_label = args.unknown_label
                if args.ood_action == "zd_unknown":
                    force_zd = 1
                reason = (reason + "|OOD") if reason else "OOD"

            if args.policy_debug:
                _log("POLICY",
                     f"{key} pred={e['label']} -> final={final_label} "
                     f"ctx_http={http_ratio:.3f} pps={pps_m:.1f} "
                     f"bot_ratio={bot_ratio:.3f} ps(dport,dst)=({u_dport},{u_dst}) "
                     f"cons={mean_cons:.3f} mismatch={ctx_mismatch} action={reason} tls_present={tls_present}")

            e_out = dict(e)
            e_out["label_orig"] = e["label"]
            e_out["label"] = final_label
            e_out["ctx_http_ratio"] = float(http_ratio)
            e_out["ctx_bot_ratio"] = float(bot_ratio)
            e_out["pps_mean"] = float(pps_m)
            e_out["ps_uniq_dport"] = int(u_dport)
            e_out["ps_uniq_dst"] = int(u_dst)
            e_out["mean_consistency"] = float(mean_cons)
            e_out["ctx_mismatch"] = int(ctx_mismatch)
            e_out["policy_reason"] = reason
            if args.ood == "on":
                e_out["ood_score"] = float(ood_score_ev)
                e_out["ood_thresh"] = float(args.ood_thresh)
            if force_zd:
                e_out["zd"] = 1
            post_events.append(e_out)



        for e in post_events:
            flag = " ZD!" if e.get("zd", 0) else " OK "
            print(f"[EVENT]{flag} {e['flow']} | label={e['label']:<16} "
                  f"windows={e['count']} pmax_max={e['pmax_max']:.3f} pmax_mean={e['pmax_mean']:.3f} "
                  f"span=[{e['start_idx']}..{e['end_idx']}] "
                  f"| ctx_http={e['ctx_http_ratio']:.2f} pps={e['pps_mean']:.1f} "
                  f"ps(dport,dst)=({e['ps_uniq_dport']},{e['ps_uniq_dst']}) "
                  f"cons={e['mean_consistency']:.2f} mismatch={e['ctx_mismatch']} reason={e['policy_reason']}")

            _emit("event",{
                "flow": list(e["flow"]) if isinstance(e["flow"], tuple) else e["flow"],
                "label": e["label"],
                "label_orig": e.get("label_orig"),
                "zd": int(e.get("zd", 0)),
                "count": int(e["count"]),
                "start_idx": int(e["start_idx"]),
                "end_idx": int(e["end_idx"]),
                "first_ts": float(e["first_ts"]),
                "last_ts": float(e["last_ts"]),
                "pmax_max": float(e["pmax_max"]),
                "pmax_mean": float(e["pmax_mean"]),
                "recon_mse_mean": float(e.get("recon_mse_mean", 0.0)),
                "ctx_http_ratio": float(e.get("ctx_http_ratio", 0.0)),
                "ctx_bot_ratio": float(e.get("ctx_bot_ratio", 0.0)),
                "pps_mean": float(e.get("pps_mean", 0.0)),
                "ps_uniq_dport": int(e.get("ps_uniq_dport", 0)),
                "ps_uniq_dst": int(e.get("ps_uniq_dst", 0)),
                "ctx_mismatch": int(e.get("ctx_mismatch", 0)),
                "policy_reason": e.get("policy_reason", ""),
                "ood_score": float(e.get("ood_score", 0.0)) if "ood_score" in e else None,
                "ood_thresh": float(e.get("ood_thresh", 0.0)) if "ood_thresh" in e else None,
            })

        # Send email for each event (if enabled)
        if args.email_alerts == "on":
            for e in post_events:
                try:
                    _send_email_alert(e, args)
                except Exception:
                    pass

        events_buffer.extend(post_events)
        if args.dump != "none" and events_buffer:
            _save_events(events_buffer, args)
            events_buffer.clear()

    # PROCESSING ONE SNAPSHOT
    def process_df(df_snap: pd.DataFrame):
        """
        Run the full L1 + L2 + policy pipeline on a single feature snapshot.

        A "snapshot" is a DataFrame returned by `FeatureTransformer.flush()`
        representing aggregated per-flow statistics in the last time bin.

        The function performs the following steps:

            1) Feature preparation:
                - normalize column names via `normalize_feature_names`,
                - compute derived features (`compute_derived_features`),
                - sanity-check units using `consistency` statistics,
                - log presence/absence of BWD-related features.

            2) L1 preprocessing:
                - standardize features into L1 Z-space (`standardize_for_l1`),
                - produce an array `X_base_std` aligned with `l1_feats`.

            3) L2 preprocessing:
                - apply robust clipping + standardisation (`clip_std_l2`),
                - attach meta columns for context (src/dst/ports, TLS, etc.),
                - validate that all required L2 features are present.

            4) Window generation:
                - build temporal windows per flow using either:
                    * `last_windows_per_flow` (one window per flow), or
                    * `sliding_windows_per_flow` (all windows),
                - maintain mappings from flow keys and `time_index` to support
                  reconstruction of base indices and context slices.

            5) L1 inference:
                - construct per-window L1 inputs from `X_base_std`,
                - run `emb_head` for embeddings and `recon_head` for
                  reconstructions,
                - compute per-window MSE / MAE and log them to `windows_mse.csv`.

            6) Pre-L2 filtering:
                - optionally filter windows based on reconstruction error
                  using `select_by_recon` (`--pre_l2_filter`),
                - optionally apply strict MSE threshold (`--recon_min_mse`).

            7) L2 inference and calibration:
                - run L2 model on (windowed features, embeddings),
                - obtain raw logits,
                - apply optional bias and temperature calibration,
                - compute class probabilities,
                - derive uncertainty metrics (pmax, entropy, margin),
                - optionally compute alternative "policy" softmax (`policy_T`).

            8) Zero-Day gating and OOD-lite:
                - compute ZD mask using `zd_gate_mask` over policy probabilities,
                - compute window-level OOD-lite pre-score combining
                  reconstruction error and uncertainty.

            9) Window-level logging and streaming:
                - print per-window summaries (`[OK]/[ZD!]` lines),
                - optionally emit 'window' JSON objects for GUI,
                - construct items for EventStitcher, including context
                  (HTTP, bot, portscan, TLS, recon errors).

           10) Event-level aggregation:
                - update EventStitcher with window items,
                - for finished events, recompute context metrics,
                - apply `_route_policy` and OOD-lite at event level,
                - print `[EVENT]` lines,
                - optionally emit 'event' JSON objects and email alerts,
                - push events into `events_buffer`.

           11) Periodic STATS logging:
                - log total windows, kept windows, Zero-Day windows,
                  and buffer size at regular intervals.

           12) Optional dumping:
                - 'live' dump: write per-window diagnostic CSV/Parquet with
                  logits, probabilities, recon errors, and averaged features.
                - 'confident' dump: write confident windows per class and
                  Zero-Day into dedicated subfolders for later offline analysis.

        Parameters
        ----------
        df_snap : pandas.DataFrame
            Snapshot of per-flow features from the transformer.
        """
        nonlocal last_stats_print, events_buffer, recent_flow_groups

        if df_snap.empty:
            return

        df_snap = normalize_feature_names(df_snap, debug=True)
        df_snap = df_snap.reset_index(drop=True)
        df_derived = compute_derived_features(df_snap)

        # sanity-check units: pkts/s * IAT(µs) should be in reasonable range
        try:
            _c = df_derived.loc[
                (df_derived["flow_pkts_per_sec"] > 0) & (df_derived["flow_iat_mean"] > 0), "consistency"]
            if len(_c):
                _med = float(_c.median())
                if not (1e3 <= _med <= 1e7):
                    _log("WARN",
                         f"[IAT/units?] consistency median = {_med:.2f} (expected 1e3..1e7). Check IAT/bin units.")
            else:
                _log("WARN", "[IAT/units?] check skipped: no valid rows.")
        except Exception:
            pass

        # --- BWD sanity check (after derive) ---
        _check_cols = [
            "tot_bwd_pkts", "totlen_bwd_pkts",
            "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_iat_mean",
            "bwd_pkts_per_sec", "bwd_bytes_per_sec",
            "ratio_pkts_per_sec_bwd_fwd", "ratio_bytes_per_sec_bwd_fwd", "ratio_bytes_per_pkt_bwd_fwd",
        ]
        _present = [c for c in _check_cols if c in df_derived.columns]
        if _present:
            share_bwd = float((df_derived.get("tot_bwd_pkts", pd.Series([0] * len(df_derived))) > 0).mean())
            _log("CHECK", f"BWD columns present: {_present} | share(tot_bwd_pkts>0)={share_bwd:.2%}")
        else:
            _log("CHECK", "BWD columns are ABSENT in df_derived (meaning the transformer didn't give them)")

        clip_map = None
        if getattr(args, "l1_clip_json", None):
            try:
                clip_map = json.load(open(args.l1_clip_json, "r", encoding="utf-8-sig"))
            except Exception as e:
                _log("WARN", f"Cannot read --l1_clip_json: {e}")
        X_base_std = standardize_for_l1(df_derived, l1_feats, l1_mean, l1_scale, zlo, zhi, clip_map=clip_map)
        df_l2_std = clip_std_l2(df_derived[l2_feats], l2_feats, lo, hi, mu, sd)
        _log("CHECK", f"L2 slice assembled: df_l2_std.shape={df_l2_std.shape} (cols={len(l2_feats)})")

        meta_cols = [
            "src", "sport", "dst", "dport", "proto", "time_index", "ts_edge", "flow_pkts_per_sec",
            "is_tls", "tls_alpn", "tls_sni", "tls_ja3", "consistency",
            # --- BWD / RATIO (if they are not present, pandas will simply ignore them) ---
            "tot_bwd_pkts", "totlen_bwd_pkts", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_iat_mean",
            "bwd_pkts_per_sec", "bwd_bytes_per_sec",
            "ratio_pkts_per_sec_bwd_fwd", "ratio_bytes_per_sec_bwd_fwd", "ratio_bytes_per_pkt_bwd_fwd",
        ]
        meta_add = [c for c in meta_cols if c in df_derived.columns and c not in df_l2_std.columns]
        if meta_add:
            df_l2_std = pd.concat([df_l2_std, df_derived[meta_add]], axis=1)

        missing_l2 = [c for c in l2_feats if c not in df_l2_std.columns]
        if missing_l2:
            raise ValueError(f"Missing L2 Features: {missing_l2}")
        feat_cols = list(l2_feats)

        if args.all_windows:
            keys, X_ts, starts = sliding_windows_per_flow(df_l2_std, feat_cols, WINDOW_SIZE)
        else:
            keys, X_ts, starts = last_windows_per_flow(df_l2_std, feat_cols, WINDOW_SIZE)
        if len(keys) == 0:
            return

        # Additional window shape check for L2
        if X_ts.ndim != 3 or X_ts.shape[1] != WINDOW_SIZE or X_ts.shape[2] != len(l2_feats):
            raise RuntimeError(f"[L2-SHAPE] Were expecting (*,{WINDOW_SIZE},{len(l2_feats)}), received {X_ts.shape}")
        _log("CHECK", f"X_ts.shape={X_ts.shape} (ok)")

        stats["windows_total"] += len(keys)

        base_index_map = {}
        start_pos_map = {}
        flow_groups = {}
        for key, sub in df_derived.groupby(["src", "sport", "dst", "dport", "proto"], sort=False):
            sub = sub.sort_values("time_index")
            idxs = sub.index.values
            tids = sub["time_index"].values.astype(np.int64)
            base_index_map[key] = idxs
            start_pos_map[key] = {int(t): int(p) for p, t in enumerate(tids)}
            flow_groups[key] = sub

        recent_flow_groups = flow_groups

        base_windows = []
        ctx_slices = []
        for i, key in enumerate(keys):
            idxs = base_index_map.get(key)
            pos_map = start_pos_map.get(key, {})
            s = int(starts[i])
            pos = pos_map.get(s, None)
            if (idxs is None) or (pos is None) or (pos + WINDOW_SIZE > len(idxs)):
                base_windows.append(None); ctx_slices.append(None); continue
            take = idxs[pos:pos + WINDOW_SIZE]
            base_windows.append(X_base_std[take, :])
            ctx_slices.append(flow_groups[key].iloc[pos:pos + WINDOW_SIZE])

        valid_idx = [i for i, x in enumerate(base_windows) if x is not None]
        if not valid_idx:
            return

        def _iter_chunks(n, m):
            """
            Small generator to split a list of length n into microbatches of size m.
            """
            i = 0
            while i < n:
                j = min(n, i + m)
                yield i, j
                i = j

        for i0, i1 in _iter_chunks(len(valid_idx), int(args.win_batch)):
            idx_slice = valid_idx[i0:i1]
            L1_win = np.stack([base_windows[i] for i in idx_slice], axis=0).astype(np.float32)
            X_ts_slice = X_ts[idx_slice]
            keys_slice = [keys[i] for i in idx_slice]
            starts_slice = starts[idx_slice]
            ctx_slice = [ctx_slices[i] for i in idx_slice]

            if not np.isfinite(L1_win).all():
                _log("WARN", "[SANITY] L1_win contains NaN/Inf - I'm replacing it with 0")
                L1_win = np.nan_to_num(L1_win, nan=0.0, posinf=0.0, neginf=0.0)

            E = emb_head.predict(L1_win, verbose=0, batch_size=int(args.l1_batch))
            R = recon_head.predict(L1_win, verbose=0, batch_size=int(args.l1_batch)).astype(np.float32)
            diff = (L1_win - R)
            recon_mse = np.mean(diff ** 2, axis=(1, 2)).astype(np.float32)
            recon_mae = np.mean(np.abs(diff), axis=(1, 2)).astype(np.float32)
            del R, diff, L1_win; gc.collect()

            if not hasattr(args, "_mse_log_init_done"):
                os.makedirs(args.out, exist_ok=True)
                args.mse_log_file = os.path.join(args.out, "windows_mse.csv")
                with open(args.mse_log_file, "w", encoding="utf-8", newline="") as _fh:
                    import csv as _csv
                    _w = _csv.writer(_fh)
                    _w.writerow([
                        "src", "sport", "dst", "dport", "proto",
                        "start_idx", "pmax_placeholder", "recon_mse", "recon_mae"
                    ])
                args._mse_log_init_done = True

            import csv as _csv
            with open(args.mse_log_file, "a", encoding="utf-8", newline="") as _fh:
                _w = _csv.writer(_fh)
                for _j, _key in enumerate(keys_slice):
                    _w.writerow([
                        _key[0], int(_key[1]), _key[2], int(_key[3]), int(_key[4]),
                        int(starts_slice[_j]),
                        "",  # pmax_placeholder
                        float(recon_mse[_j]),
                        float(recon_mae[_j]),
                    ])

            # --- pre-L2 base filter ---
            if args.pre_l2_filter == "on":
                keep = select_by_recon(
                    recon_mse,
                    budget=args.recon_budget if args.recon_thresh is None else None,
                    thresh=args.recon_thresh
                )
            else:
                keep = np.ones(len(keys_slice), dtype=bool)

            # --- ADD: Strict MSE screening (NEW) ---
            if getattr(args, "recon_min_mse", None) is not None:
                min_mse = float(args.recon_min_mse)
                keep = keep & (recon_mse >= min_mse)
                if keep.sum() == 0:
                    _log("INFO", f"[Pre-L2] After --recon_min_mse={min_mse} all windows are filtered (microbatch)")
                    continue

            if keep.sum() == 0:
                _log("INFO", "[Pre-L2] all windows are filtered (microbatch)")
                continue

            stats["windows_kept"] += int(keep.sum())

            X_ts_k = X_ts_slice[keep]
            E_k = E[keep]
            mse_k = recon_mse[keep]
            mae_k = recon_mae[keep]
            keys_k = [keys_slice[i] for i in np.where(keep)[0]]
            starts_k = starts_slice[keep]
            ctx_k = [ctx_slice[i] for i in np.where(keep)[0]]

            if not np.isfinite(X_ts_k).all():
                _log("WARN", "[SANITY] X_ts_k contains NaN/Inf - I'll replace it with 0")
                X_ts_k = np.nan_to_num(X_ts_k, nan=0.0, posinf=0.0, neginf=0.0)

            logits = l2.predict([X_ts_k, E_k], batch_size=int(args.l2_batch), verbose=0)  # [N, C]
            logits_raw = logits.copy()

            # primary calibration (bias/T) for the combat mark
            logits_cal = logits.copy()
            if biases is not None:
                try:
                    logits_cal = logits_cal - biases.reshape(1, -1)
                except Exception:
                    pass
            if T is not None and float(T) > 0:
                logits_cal = logits_cal / float(T)

            probs = tf.nn.softmax(logits_cal, axis=1).numpy()
            y = np.argmax(probs, axis=1)

            # --- policy-softmax (does NOT change y) ---
            if getattr(args, "policy_T", None) is not None and float(args.policy_T) > 0:
                pt = float(args.policy_T)
                try:
                    probs_policy = tf.nn.softmax(logits_raw / pt, axis=1).numpy()
                except Exception:
                    probs_policy = probs
            else:
                probs_policy = probs

            # Quick diagnostics of "sticking"
            pmax_med = float(np.median(probs_policy.max(axis=1))) if len(probs_policy) else 0.0
            if pmax_med > 0.995:
                _log("WARN",
                     f"[STICKY?] policy_pmax median ~ {pmax_med:.3f}. Check the calibration (T/bias) and thresholds.")

            # --- light dump of logits (if enabled) ---
            if args.dump_logits == 'on':
                log_rows = []
                now_ts = time.time()
                for i, key in enumerate(keys_k):
                    row = {
                        "ts": now_ts,
                        "src": key[0], "sport": key[1], "dst": key[2], "dport": key[3], "proto": key[4],
                        "start_idx": int(starts_k[i]),
                        "pred_id": int(y[i]),
                        "pred_label": CANONICAL_CLASS_ORDER[y[i]],
                        "p_max": float(np.max(probs[i])),
                        "pmax_policy": float(np.max(probs_policy[i])),
                    }
                    order = np.argsort(probs[i])
                    top1 = order[-1]; top2 = order[-2] if len(order) >= 2 else top1
                    row["top1_label"] = CANONICAL_CLASS_ORDER[int(top1)]
                    row["top2_label"] = CANONICAL_CLASS_ORDER[int(top2)]
                    row["margin_top1_top2"] = float(probs[i, top1] - probs[i, top2])
                    for ci, cname in enumerate(CANONICAL_CLASS_ORDER):
                        row[f"raw_logit_{cname}"] = float(logits_raw[i, ci])
                        row[f"logit_{cname}"] = float(logits_cal[i, ci])
                        row[f"prob_{cname}"] = float(probs[i, ci])
                        row[f"prob_policy_{cname}"] = float(probs_policy[i, ci])
                    log_rows.append(row)
                _save_logits_rows(log_rows, args)

            # --- ZDGate: by policy-probs (NEW) ---
            if args.zd_gate == "on":
                is_zd, pmax_policy_vals, ent_policy, marg_policy = zd_gate_mask(
                    probs_policy,
                    pmax_thr=args.zd_pmax,
                    entropy_thr=args.zd_entropy,
                    margin_thr=args.zd_margin
                )
            else:
                is_zd = np.zeros(len(y), dtype=bool)
                pmax_policy_vals = probs_policy.max(axis=1)
                sortp = np.sort(probs_policy, axis=1)
                marg_policy = sortp[:, -1] - sortp[:, -2]
                ent_policy = _entropy(probs_policy)

            # --- OOD-Lite pre-score for the main problems (left as is) ---
            if args.ood == "on":
                pmax = probs.max(axis=1)
                mse_norm = np.maximum(mse_k / max(args.ood_mse_median, 1e-9), 0.0)
                ood_score_pre = args.ood_alpha * mse_norm + args.ood_beta * (1.0 - pmax)
            else:
                ood_score_pre = np.zeros(len(y), dtype=float)

            stats["windows_zd"] += int(is_zd.sum())

            emit_items = []
            now_ts = time.time()
            # I'll leave the old names for pmax/ent/marg for printing, but they are considered higher for policy (I won't change the structure)
            pmax_main = probs.max(axis=1)
            sortp_main = np.sort(probs, axis=1)
            marg_main = sortp_main[:, -1] - sortp_main[:, -2]
            ent_main = _entropy(probs)

            for i, key in enumerate(keys_k):
                lab = CANONICAL_CLASS_ORDER[y[i]]
                ctx_df = ctx_k[i]
                http_mask = _ctx_http_mask(ctx_df, http_ports, ip_whitelist)
                bot_mask = _ctx_bot_mask(ctx_df, bot_ports)

                tls_present = False
                if "is_tls" in ctx_df.columns:
                    is_tls_s = ctx_df["is_tls"].fillna(0).astype(bool)
                    alpn_s = ctx_df.get("tls_alpn", pd.Series([""] * len(ctx_df))).fillna("").astype(str)
                    sni_s = ctx_df.get("tls_sni", pd.Series([""] * len(ctx_df))).fillna("").astype(str)
                    try:
                        tls_present = bool((is_tls_s & (alpn_s.str.contains(_ALPN_HTTP_REGEX, regex=True, na=False) | (sni_s != ""))).any())
                    except Exception:
                        tls_present = False

                http_ratio = float(np.mean(http_mask)) if len(http_mask) else 0.0
                bot_ratio = float(np.mean(bot_mask)) if len(bot_mask) else 0.0
                pps_m = _pps_mean(ctx_df)
                u_dport, u_dst = _uniq_counts_for_ps(ctx_df)
                emit_items.append({
                    "flow": key,
                    "label": lab,
                    "zd": bool(is_zd[i]),
                    "start_idx": int(starts_k[i]),
                    "ts": now_ts,
                    "pmax": float(pmax_main[i]),
                    "ctx": {
                        "http_ratio": http_ratio,
                        "pps_mean": pps_m,
                        "uniq_dport": u_dport,
                        "uniq_dst": u_dst,
                        "bot_ratio": bot_ratio,
                        "entropy": float(ent_main[i]),
                        "margin": float(marg_main[i]),
                        "recon_mse": float(mse_k[i]),
                        "recon_mae": float(mae_k[i]),
                        "tls_present": tls_present
                    }
                })

            finished = stitcher.update(emit_items)

            for i, key in enumerate(keys_k):
                lab = CANONICAL_CLASS_ORDER[y[i]]
                flag = "ZD!" if is_zd[i] else "OK "
                print(f"[{flag}] {key} | pred={lab:16s} pmax={pmax_main[i]:.3f} "
                      f"recon_mse={mse_k[i]:.4g} margin={marg_main[i]:.3f} H={ent_main[i]:.3f}")
                _emit("window",{
                    "ts": float(now_ts),
                    "flow": list(key),  # [src, sport, dst, dport, proto]
                    "pred": lab,
                    "pmax": float(pmax_main[i]),
                    "entropy": float(ent_main[i]),
                    "margin": float(marg_main[i]),
                    "start_idx": int(starts_k[i]),
                    "recon_mse": float(mse_k[i]),
                    "recon_mae": float(mae_k[i]),
                    "zd": bool(is_zd[i]),
                    "ctx": {
                        "http_ratio": float(np.mean(_ctx_http_mask(ctx_k[i], http_ports, ip_whitelist))),
                        "pps_mean": float(_pps_mean(ctx_k[i])),
                    }
                }
                )

            post_events = []
            for e in finished:
                key = e["flow"]
                sub = flow_groups.get(key)
                tls_present = False
                if sub is None or len(sub) < WINDOW_SIZE:
                    http_ratio = 0.0; bot_ratio = 0.0; pps_m = 0.0; u_dport = 0; u_dst = 0; mean_cons = 0.0
                else:
                    s = e["start_idx"]
                    pos_map = {int(t): int(p) for p, t in enumerate(sub["time_index"].values.astype(np.int64))}
                    pos = pos_map.get(int(s), None)
                    if pos is None or (pos + WINDOW_SIZE > len(sub)):
                        http_ratio = 0.0; bot_ratio = 0.0; pps_m = 0.0; u_dport = 0; u_dst = 0; mean_cons = 0.0
                    else:
                        slice_df = sub.iloc[pos:pos + WINDOW_SIZE]
                        http_ratio = float(np.mean(_ctx_http_mask(slice_df, http_ports, ip_whitelist)))
                        bot_ratio = float(np.mean(_ctx_bot_mask(slice_df, bot_ports)))
                        pps_m = _pps_mean(slice_df)
                        u_dport, u_dst = _uniq_counts_for_ps(slice_df)
                        if "consistency" in slice_df.columns:
                            mean_cons = float(slice_df["consistency"].mean())
                        else:
                            mean_cons = float((slice_df["flow_pkts_per_sec"] * slice_df["flow_iat_mean"]).mean())
                        if "is_tls" in slice_df.columns:
                            is_tls_s = slice_df["is_tls"].fillna(0).astype(bool)
                            alpn_s = slice_df.get("tls_alpn", pd.Series([""] * len(slice_df))).fillna("").astype(str)
                            sni_s = slice_df.get("tls_sni", pd.Series([""] * len(slice_df))).fillna("").astype(str)
                            tls_present = bool((is_tls_s & (alpn_s.str.contains(_ALPN_HTTP_REGEX, regex=True, na=False) | (sni_s != ""))).any())

                final_label, ctx_mismatch, force_zd, reason = _route_policy(
                    e["label"], http_ratio, pps_m, u_dport, u_dst, bot_ratio, args,
                    mean_consistency=mean_cons, has_tls_signal=tls_present
                )

                pmax_ev = float(e["pmax_mean"])
                mse_norm_ev = float(e.get("recon_mse_mean", args.ood_mse_median)) / max(args.ood_mse_median, 1e-9)
                ood_score_ev = (args.ood_alpha * mse_norm_ev + args.ood_beta * (1.0 - pmax_ev) + args.ood_gamma * float(ctx_mismatch))

                if args.ood == "on" and (ood_score_ev > float(args.ood_thresh)):
                    final_label = args.unknown_label
                    if args.ood_action == "zd_unknown":
                        force_zd = 1
                    reason = (reason + "|OOD") if reason else "OOD"

                if args.policy_debug:
                    _log("POLICY",
                         f"{key} pred={e['label']} -> final={final_label} "
                         f"ctx_http={http_ratio:.3f} pps={pps_m:.1f} "
                         f"bot_ratio={bot_ratio:.3f} ps(dport,dst)=({u_dport},{u_dst}) "
                         f"cons={mean_cons:.3f} mismatch={ctx_mismatch} action={reason} tls_present={tls_present}")

                e_out = dict(e)
                e_out["label_orig"] = e["label"]
                e_out["label"] = final_label
                e_out["ctx_http_ratio"] = float(http_ratio)
                e_out["ctx_bot_ratio"] = float(bot_ratio)
                e_out["pps_mean"] = float(pps_m)
                e_out["ps_uniq_dport"] = int(u_dport)
                e_out["ps_uniq_dst"] = int(u_dst)
                e_out["mean_consistency"] = float(mean_cons)
                e_out["ctx_mismatch"] = int(ctx_mismatch)
                e_out["policy_reason"] = reason
                if args.ood == "on":
                    e_out["ood_score"] = float(ood_score_ev)
                    e_out["ood_thresh"] = float(args.ood_thresh)
                if force_zd:
                    e_out["zd"] = 1
                post_events.append(e_out)

            for e in post_events:
                flag = " ZD!" if e.get("zd", 0) else " OK "
                print(f"[EVENT]{flag} {e['flow']} | label={e['label']:<16} "
                      f"windows={e['count']} pmax_max={e['pmax_max']:.3f} pmax_mean={e['pmax_mean']:.3f} "
                      f"span=[{e['start_idx']}..{e['end_idx']}] "
                      f"| ctx_http={e['ctx_http_ratio']:.2f} pps={e['pps_mean']:.1f} "
                      f"ps(dport,dst)=({e['ps_uniq_dport']},{e['ps_uniq_dst']}) "
                      f"cons={e['mean_consistency']:.2f} mismatch={e['ctx_mismatch']} reason={e['policy_reason']}")

                _emit("event", {
                    "flow": list(e["flow"]) if isinstance(e["flow"], tuple) else e["flow"],
                    "label": e["label"],
                    "label_orig": e.get("label_orig"),
                    "zd": int(e.get("zd", 0)),
                    "count": int(e["count"]),
                    "start_idx": int(e["start_idx"]),
                    "end_idx": int(e["end_idx"]),
                    "first_ts": float(e["first_ts"]),
                    "last_ts": float(e["last_ts"]),
                    "pmax_max": float(e["pmax_max"]),
                    "pmax_mean": float(e["pmax_mean"]),
                    "recon_mse_mean": float(e.get("recon_mse_mean", 0.0)),
                    "ctx_http_ratio": float(e.get("ctx_http_ratio", 0.0)),
                    "ctx_bot_ratio": float(e.get("ctx_bot_ratio", 0.0)),
                    "pps_mean": float(e.get("pps_mean", 0.0)),
                    "ps_uniq_dport": int(e.get("ps_uniq_dport", 0)),
                    "ps_uniq_dst": int(e.get("ps_uniq_dst", 0)),
                    "ctx_mismatch": int(e.get("ctx_mismatch", 0)),
                    "policy_reason": e.get("policy_reason", ""),
                    "ood_score": float(e.get("ood_score", 0.0)) if "ood_score" in e else None,
                    "ood_thresh": float(e.get("ood_thresh", 0.0)) if "ood_thresh" in e else None,
                })

            # Send email for each event (if enabled)
            if args.email_alerts == "on":
                for e in post_events:
                    try:
                        _send_email_alert(e, args)
                    except Exception:
                        pass

            events_buffer.extend(post_events)

            if time.time() - last_stats_print >= stats_period:
                last_stats_print = time.time()
                ratio_keep = stats["windows_kept"] / max(1, stats["windows_total"])
                ratio_zd = stats["windows_zd"] / max(1, stats["windows_kept"])
                _log("STATS", f"total={stats['windows_total']} kept={stats['windows_kept']} ({ratio_keep:.2%}) "
                              f"ZD={stats['windows_zd']} ({ratio_zd:.2%} of kept) events_buf={len(events_buffer)}")

            # ---------- LIVE dump of everything ----------
            if args.dump != "none":
                rows = []
                now_ts = time.time()
                for i, key in enumerate(keys_k):
                    row = {
                        "src": key[0], "sport": key[1], "dst": key[2], "dport": key[3], "proto": key[4],
                        "pred_id": int(y[i]),
                        "pred_label": CANONICAL_CLASS_ORDER[y[i]],
                        "p_max": float(np.max(probs[i])),
                        "pmax_policy": float(np.max(probs_policy[i])),
                        "entropy": float(ent_main[i]),
                        "margin": float(marg_main[i]),
                        "l1_recon_mse": float(mse_k[i]),
                        "l1_recon_mae": float(mae_k[i]),
                        "zd_flag": int(is_zd[i]),
                        "start_idx": int(starts_k[i]),
                        "ood_score_pre": float(ood_score_pre[i]),
                        "ood_thresh": float(args.ood_thresh),
                        "ts": now_ts
                    }
                    for ci, cname in enumerate(CANONICAL_CLASS_ORDER):
                        row[f"raw_logit_{cname}"] = float(logits_raw[i, ci])
                        row[f"logit_{cname}"] = float(logits_cal[i, ci])
                        row[f"prob_{cname}"] = float(probs[i, ci])
                        row[f"prob_policy_{cname}"] = float(probs_policy[i, ci])

                    ctx_df = ctx_k[i]
                    try:
                        raw_means = ctx_df[l2_feats].mean(numeric_only=True)
                    except Exception:
                        raw_means = pd.Series(dtype=float)

                    for fname in l2_feats:
                        if fname in raw_means.index:
                            val = float(raw_means[fname])
                            row[fname] = val
                            row[f"f_{fname}"] = val
                    rows.append(row)
                outdf = pd.DataFrame(rows)
                os.makedirs(args.out, exist_ok=True)
                base = f"live_{int(time.time())}"
                out_path = Path(args.out) / (base + (".parquet" if args.dump == "parquet" else ".csv"))
                if args.dump == "parquet":
                    try:
                        import pyarrow  # noqa
                        outdf.to_parquet(str(out_path), index=False)
                    except Exception:
                        _log("WARN", "pyarrow is unavailable, saving as CSV.")
                        out_path = out_path.with_suffix(".csv")
                        outdf.to_csv(str(out_path), index=False)
                else:
                    outdf.to_csv(str(out_path), index=False)
                _log("SAVE", f"live -> {out_path}")

            # ---------- NEW: Selective dump of confident windows by class + ZD ----------
            if args.confident_dump == "on":
                ts_now = int(time.time())
                base_dir = Path(args.out) / str(args.confident_subdir)
                # First, let's create the class and zd folders.
                for sub in list(CANONICAL_CLASS_ORDER) + ["zd"]:
                    os.makedirs(base_dir / sub, exist_ok=True)

                # Let's collect batch lines into folders
                buckets = defaultdict(list)

                for i, key in enumerate(keys_k):
                    cls_name = CANONICAL_CLASS_ORDER[y[i]]
                    # If ZD, we always write in zd
                    if bool(is_zd[i]):
                        target = "zd"
                    else:
                        # confidence threshold for policy_pmax (if policy_T is specified)
                        pmax_for_conf = float(np.max(probs_policy[i])) if getattr(args, "policy_T", None) else float(np.max(probs[i]))
                        if pmax_for_conf < float(args.confident_pmax):
                            continue
                        target = cls_name

                    row = {
                        "ts": ts_now,
                        "src": key[0], "sport": int(key[1]), "dst": key[2], "dport": int(key[3]), "proto": int(key[4]),
                        "start_idx": int(starts_k[i]),
                        "pred_label": cls_name,
                        "zd_flag": int(is_zd[i]),
                        "p_max": float(np.max(probs[i])),
                        "pmax_policy": float(np.max(probs_policy[i])),
                        "entropy": float(ent_main[i]),
                        "margin": float(marg_main[i]),
                        "l1_recon_mse": float(mse_k[i]),
                        "l1_recon_mae": float(mae_k[i]),
                    }
                    for ci, cname in enumerate(CANONICAL_CLASS_ORDER):
                        row[f"raw_logit_{cname}"] = float(logits_raw[i, ci])
                        row[f"logit_{cname}"] = float(logits_cal[i, ci])
                        row[f"prob_{cname}"] = float(probs[i, ci])
                        row[f"prob_policy_{cname}"] = float(probs_policy[i, ci])

                    # Average window features (as in live)
                    ctx_df = ctx_k[i]
                    try:
                        raw_means = ctx_df[l2_feats].mean(numeric_only=True)
                    except Exception:
                        raw_means = pd.Series(dtype=float)
                    for fname in l2_feats:
                        if fname in raw_means.index:
                            val = float(raw_means[fname])
                            row[f"f_{fname}"] = val

                    buckets[target].append(row)

                # Let's export a CSV file for each bucket.
                for sub, rows_sub in buckets.items():
                    if not rows_sub:
                        continue
                    out_path = base_dir / sub / f"win_{ts_now}.csv"
                    try:
                        pd.DataFrame(rows_sub).to_csv(out_path, index=False)
                        _log("SAVE", f"confident -> {out_path}")
                    except Exception as e:
                        _log("WARN", f"Cannot save confident dump: {e}")

    if args.source == "sniffer":
        try:
            from scapy.all import sniff, IP, TCP, UDP, IPv6, get_if_list  # Added get_if_list
        except Exception as e:
            _log("ERROR", f"pip install scapy ; and run with privileges (Admin/sudo). Error: {e}")
            sys.exit(1)

        # ----auto-select iface if 'auto' or empty is passed ----
        iface = args.iface
        if iface in (None, "", "auto"):
            try:
                all_ifaces = get_if_list()
                # throw out loopback if possible
                candidates = [
                    name for name in all_ifaces
                    if "loopback" not in name.lower()
                       and name.lower() not in ("lo",)
                ]
                if not candidates:
                    candidates = all_ifaces
                iface = candidates[0]
                _log("RUN", f"Auto-selected iface='{iface}' from: {all_ifaces}")
            except Exception as e:
                _log("WARN", f" We were able to auto-detect the iface, so we use the raw value.: {args.iface} ({e})")
                iface = args.iface

        last_flush = time.time()
        try:
            def on_pkt(pkt):
                nonlocal last_flush
                from scapy.all import IP, IPv6, TCP, UDP

                ip = None
                proto = None
                src = dst = None

                if pkt.haslayer(IP):
                    ip = pkt[IP]
                    proto = int(ip.proto)
                    src = ip.src
                    dst = ip.dst
                elif pkt.haslayer(IPv6):
                    ip = pkt[IPv6]
                    proto = int(ip.nh)
                    src = ip.src
                    dst = ip.dst
                else:
                    return

                sport = dport = 0
                if pkt.haslayer(TCP):
                    sport = int(pkt[TCP].sport)
                    dport = int(pkt[TCP].dport)
                elif pkt.haslayer(UDP):
                    sport = int(pkt[UDP].sport)
                    dport = int(pkt[UDP].dport)
                else:
                    return

                ts = float(pkt.time)
                try:
                    size = int(len(pkt))
                except Exception:
                    size = 0

                raw_bytes = None
                try:
                    if pkt.haslayer(TCP):
                        raw_bytes = bytes(pkt[TCP].payload)
                    elif pkt.haslayer(UDP):
                        raw_bytes = bytes(pkt[UDP].payload)
                except Exception:
                    raw_bytes = None

                transformer.ingest_packet(ts, src, sport, dst, dport, proto, size, raw_bytes=raw_bytes)

                now = time.time()
                if now - last_flush >= args.flush_sec:
                    df = transformer.flush()
                    if not df.empty:
                        process_df(df)
                    last_flush = now

            _log("RUN", f"Sniffer on iface={iface} bin={args.bin}s flush={args.flush_sec}s "
                        f"| preL2={args.pre_l2_filter} ZD={args.zd_gate} all_windows={args.all_windows}")
            sniff(iface=iface, prn=on_pkt, store=False)
        except KeyboardInterrupt:
            _log("SNF", "CTRL+C → flush events")
            finalize_and_dump()

    elif args.source == "pcap":
        try:
            import dpkt  # noqa: F401
        except Exception as e:
            _log("ERROR", f"pip install dpkt ; Error: {e}")
            sys.exit(1)

        import dpkt

        def inet_to_str(x: bytes, family=socket.AF_INET) -> str:
            return socket.inet_ntop(family, x)

        def _iter_packets(iterable):
            for rec in iterable:
                try:
                    if isinstance(rec, tuple):
                        if len(rec) >= 2:
                            yield float(rec[0]), rec[1]
                        else:
                            continue
                    else:
                        ts, buf = rec
                        yield float(ts), buf
                except Exception:
                    continue

        pkts_since = 0
        last_flush_ts = None
        read_pkts_total = 0
        first_ts = None

        try:
            with open(args.pcap, "rb") as f:
                try:
                    pcap = dpkt.pcap.Reader(f)
                    pk_iter = pcap
                    _log("PCAP", "Detected classic PCAP")
                except Exception:
                    f.seek(0)
                    try:
                        pcapng = dpkt.pcapng.Reader(f)
                        pk_iter = pcapng
                        _log("PCAP", "Detected PCAPNG container")
                    except Exception as e:
                        _log("ERROR", f"[PCAP] Unable to recognize format (pcap/pcapng). Error: {e}")
                        sys.exit(1)

                for ts, buf in _iter_packets(pk_iter):
                    if first_ts is None:
                        first_ts = float(ts)
                    if (args.pcap_max_sec is not None) and (float(ts) - first_ts > float(args.pcap_max_sec)):
                        _log("PCAP", f"Reached --pcap_max_sec={args.pcap_max_sec}s, stopping scan.")
                        break
                    if (args.pcap_max_pkts is not None) and (read_pkts_total >= int(args.pcap_max_pkts)):
                        _log("PCAP", f"Reached --pcap_max_pkts={args.pcap_max_pkts}, stopping scan.")
                        break

                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                    except Exception:
                        continue

                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        proto = int(ip.p)
                        try:
                            src = inet_to_str(ip.src, socket.AF_INET)
                            dst = inet_to_str(ip.dst, socket.AF_INET)
                        except Exception:
                            continue
                    elif hasattr(dpkt, "ip6") and isinstance(eth.data, dpkt.ip6.IP6):
                        ip = eth.data
                        proto = int(ip.nxt)
                        try:
                            src = inet_to_str(ip.src, socket.AF_INET6)
                            dst = inet_to_str(ip.dst, socket.AF_INET6)
                        except Exception:
                            continue
                    else:
                        continue

                    sport = dport = 0
                    is_tcp = (proto == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP)) if hasattr(dpkt,
                                                                                                              "ip") else isinstance(
                        ip.data, dpkt.tcp.TCP)
                    is_udp = (proto == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP)) if hasattr(dpkt,
                                                                                                              "ip") else isinstance(
                        ip.data, dpkt.udp.UDP)

                    if is_tcp:
                        sport = int(ip.data.sport);
                        dport = int(ip.data.dport)
                    elif is_udp:
                        sport = int(ip.data.sport);
                        dport = int(ip.data.dport)
                    else:
                        continue


                    size = int(len(ip))

                    raw_bytes = None
                    try:
                        if is_tcp:
                            raw_bytes = bytes(ip.data.data)
                        elif is_udp:
                            raw_bytes = bytes(ip.data.data)
                    except Exception:
                        raw_bytes = None

                    transformer.ingest_packet(float(ts), src, sport, dst, dport, proto, size, raw_bytes=raw_bytes)
                    read_pkts_total += 1
                    pkts_since += 1

                    need_flush = False
                    if pkts_since >= int(args.pcap_flush_pkts):
                        need_flush = True
                    if last_flush_ts is None or (float(ts) - last_flush_ts) >= float(args.pcap_flush_sec):
                        need_flush = True

                    if need_flush:
                        df = transformer.flush()
                        if not df.empty:
                            process_df(df)
                        pkts_since = 0
                        last_flush_ts = float(ts)

            df = transformer.flush()
            if not df.empty:
                process_df(df)

            finalize_and_dump()

        except FileNotFoundError:
            _log("ERROR", f"[PCAP] File not found: {args.pcap}")
            sys.exit(1)

    else:
        raise ValueError("unknown --source")

# -----------------------------------------------------------------------------
def main():
    """
     Parse CLI arguments, validate them, and start the SmartNetGuard run.

    Responsibilities:
        - define all CLI switches for:
            * data source (sniffer / pcap),
            * model paths (L1 / L2),
            * calibration, Zero-Day gate, OOD-lite,
            * policy thresholds (HTTP/VOL/BOT/PS),
            * output formats and dump options,
            * email alert configuration,
            * JSON streaming for UI integrations,
            * TensorFlow threading configuration.

        - perform minimal validation (e.g. `--pcap` required in pcap mode),
        - call `run(args)` which performs the full orchestration.

    This is the canonical entry point for the SmartNetGuard unified runner
    when invoked as a standalone script:
    """
    ap = argparse.ArgumentParser()
    # sources
    ap.add_argument("--source", choices=["sniffer", "pcap"], required=True)
    ap.add_argument(
        "--iface",
        default="auto",
        help="interface for sniffer: name/number or 'auto' (select automatically). "
             "On Windows you need Npcap and Admin rights."
    )
    ap.add_argument("--pcap", help="path to .pcap/.pcapng for pcap mode")
    ap.add_argument("--bin", type=float, default=0.1, help="time quantum (sec)")
    ap.add_argument("--flush_sec", type=float, default=1.5, help="How often should accumulated snapshots be processed (sniffer)?")
    ap.add_argument("--all_windows", action="store_true", help="if set, run all sliding windows (expensive)")

    # --- calibration (CLI) ---
    ap.add_argument("--calib_bias_json", type=str, default=None,
                    help="JSON with background logit offsets (logit_*: mean keys). If not specified, calibration.json from the L2 run is used (if available).")
    ap.add_argument("--calib_T", type=float, default=None,
                    help="Temperature scaling T (e.g. 3.0). If not specified, it is taken from calibration.json (if present).")
    ap.add_argument("--no_calib", action="store_true", help="Disable calibration (both bias and T) - for debugging.")

    # models
    ap.add_argument("--l1_run", required=True, help="run_... folder (or converted_l1_AE_ft) OR path to .keras/.h5; next to preprocessing_config.json")
    ap.add_argument("--l2_run", required=True, help="L2 run_... folder with the model and stats/calibration OR SavedModel folder")

    # calibration from calibration.json (compatibility)
    ap.add_argument("--use_calib_biases", action="store_true", help="If there are biases in calibration.json, apply them.")

    # pre-L2 filter
    ap.add_argument("--pre_l2_filter", choices=["off", "on"], default="on")
    ap.add_argument("--recon_thresh", type=float, default=None)
    ap.add_argument("--recon_budget", type=float, default=0.20)

    # ZDGate
    ap.add_argument("--zd_gate", choices=["off", "on"], default="on")
    ap.add_argument("--zd_pmax", type=float, default=0.55)
    ap.add_argument("--zd_entropy", type=float, default=1.0)
    ap.add_argument("--zd_margin", type=float, default=0.10)

    # --- OOD-Lite ---
    ap.add_argument("--ood", choices=["off", "on"], default="on", help="Enable simplified OOD-score and relay in unknown_dos by threshold")
    ap.add_argument("--ood_alpha", type=float, default=0.5, help="weight of L1 reconstruction in OOD-score")
    ap.add_argument("--ood_beta", type=float, default=0.4, help="uncertainty weight L2 (1 - pmax)")
    ap.add_argument("--ood_gamma", type=float, default=0.1, help="policy conflict weight (ctx_mismatch) at the EVENT level")
    ap.add_argument("--ood_thresh", type=float, default=1.35, help="threshold for OOD-score; above - considered unknown")
    ap.add_argument("--ood_mse_median", type=float, default=0.0035, help="median L1 MSE on normal traffic for normalization")
    ap.add_argument("--ood_action", choices=["unknown", "zd_unknown"], default="unknown", help="What to do with OOD: unknown_dos or unknown_dos+ZD")

    # IAT units
    ap.add_argument("--iat_unit", choices=["s", "ms", "us"], default="us")
    ap.add_argument("--post_iat_rescale", choices=["off", "on"], default="off", help="[IGNORED] The transformer is already giving out μs; post-rescaling is disabled.")

    # splicing of events
    ap.add_argument("--stitch_max_gap", type=int, default=2)
    ap.add_argument("--stitch_min_len", type=int, default=2)

    # dump
    ap.add_argument("--dump", choices=["none", "csv", "parquet"], default="none")
    ap.add_argument("--out", default="out_live")

    # batches
    ap.add_argument("--l1_batch", type=int, default=64, help="batch_size for L1")
    ap.add_argument("--l2_batch", type=int, default=64, help="batch_size for L2")
    ap.add_argument("--win_batch", type=int, default=2048, help="windows in the microbatch")
    ap.add_argument("--tf_intra", type=int, default=2, help="TF intra-op threads")
    ap.add_argument("--tf_inter", type=int, default=2, help="TF inter-op threads")

    # -------- Context: HTTP --------
    ap.add_argument("--http_ports", type=int, nargs="*", default=[80, 443, 8080, 8443])
    ap.add_argument("--http_event_min_ctx", type=float, default=0.50)
    ap.add_argument("--http_min_pps", type=float, default=200.0)
    ap.add_argument("--http_event_action", choices=["mark", "relabel_to_volumetric", "zd"], default="mark")

    # whitelist file (optional)
    ap.add_argument("--http_whitelist_file", default=None,
                    help="optional file with IP/CIDR (one per line). Whitelisted flows will be de-emphasized for port-based HTTP matches")

    # --- Politics/presets ---
    ap.add_argument("--policy_preset", choices=["default", "strict"], default="default")
    ap.add_argument("--http_event_min_ctx_strict", type=float, default=0.90)
    ap.add_argument("--http_min_pps_strict", type=float, default=600.0)
    ap.add_argument("--consistency_min_strict", type=float, default=1e4)
    ap.add_argument("--consistency_max_strict", type=float, default=1e8)

    # -------- Context: volumetric --------
    ap.add_argument("--vol_event_min_pps", type=float, default=300.0)
    ap.add_argument("--vol_event_max_http_ctx", type=float, default=0.30)
    ap.add_argument("--vol_event_action", choices=["mark"], default="mark")

    # -------- Easy logit dump --------
    ap.add_argument("--dump_logits", choices=["off", "on"], default="off")
    ap.add_argument("--dump_logits_prefix", default="logit_")

    # -------- Context: portscan --------
    ap.add_argument("--ps_event_min_uniq_dports", type=int, default=8)
    ap.add_argument("--ps_event_min_uniq_targets", type=int, default=5)
    ap.add_argument("--ps_event_action", choices=["mark"], default="mark")

    # -------- Context: bot --------
    ap.add_argument("--bot_service_ports", type=int, nargs="*", default=[445, 139, 135, 3389, 22])
    ap.add_argument("--bot_event_min_ctx", type=float, default=0.30)
    ap.add_argument("--bot_event_action", choices=["mark"], default="mark")

    # General fallback
    ap.add_argument("--fallback_action", choices=["mark", "zd", "relabel_to_unknown"], default="zd")
    ap.add_argument("--unknown_label", default="unknown_dos")

    # PCAP limiters
    ap.add_argument("--pcap_max_pkts", type=int, default=None, help="Maximum packets: read and stop")
    ap.add_argument("--pcap_max_sec", type=float, default=None, help="maximum seconds from the first packet and stop")

    # intermediate flush when reading PCAP
    ap.add_argument("--pcap_flush_pkts", type=int, default=200000, help="After how many packets should an intermediate flush() be performed (pcap)?")
    ap.add_argument("--pcap_flush_sec", type=float, default=5.0, help="or after how many seconds of trace should a flush() (pcap) be performed?")

    # L2 feature set auto-synchronization management
    ap.add_argument("--l2_feat_autofix", action="store_true", help="align the list of features to the number expected by the model")

    # ---- New: Policy Decision Tracing ----
    ap.add_argument("--policy_debug", action="store_true", help="Print detailed information about policy routing/fallbacks for each event")

    ap.add_argument("--l1_clip_json", type=str, default=None, help="Feature clip in L1 Z-space. Format: {flow_pkts_per_sec': [-3.5, 3.5], ...}")

    # ---------- MSE filter (L1) ----------
    ap.add_argument("--recon_min_mse", type=float, default=None,
                    help="If set, filter out windows with recon_mse < recon_min_mse (i.e. remove well-reconstructed windows).")

    # ---------- policy-specific temperature ----------
    ap.add_argument("--policy_T", type=float, default=None,
                    help="If specified, compute separate softmax(logits_raw/policy_T) for policy targets and pmax_policy.")

    # ---------- NEW: Selective dump of confident windows by class ----------
    ap.add_argument("--confident_dump", choices=["off","on"], default="off",
                    help="Save confident windows in subfolders by class + ZD.")
    ap.add_argument("--confident_pmax", type=float, default=0.80,
                    help="The confidence threshold pmax_policy (or p_max if policy_T is not set) for selective dump.")
    ap.add_argument("--confident_subdir", type=str, default="confident",
                    help="Subfolder inside --out for confident windows (will create class/ and zd/).")
    # -------- Email alerts ----------
    ap.add_argument("--email_alerts", choices=["off", "on"], default="off",
                    help="Enable sending email notifications about captured events.")
    ap.add_argument("--email_smtp_host", type=str, default="smtp.gmail.com", help="SMTP host")
    ap.add_argument("--email_smtp_port", type=int, default=587, help="SMTP port (STARTTLS)")
    ap.add_argument("--email_from", type=str, default=None, help="From — e.g. alerts@your.domain")
    ap.add_argument("--email_to", type=str, nargs="*", default=[], help="To (multiple addresses possible).")
    ap.add_argument("--email_user", type=str, default=None, help="SMTP username (if needed)")
    ap.add_argument("--email_password_env", type=str, default="EMAIL_PASSWORD",
                    help="Name of the env var where the password/token for SMTP is stored (recommended).")
    ap.add_argument("--email_use_ssl", action="store_true",
                    help="If you need to use SSL SMTP (smtps on 465). If enabled, the port is usually 465.")
    ap.add_argument("--stream_json", action="store_true", help="Stream JSON to stdout for GUI")

    args = ap.parse_args()

    if args.source == "pcap" and not args.pcap:
        ap.error("--pcap is required with --source pcap")

    run(args)

if __name__ == "__main__":
    main()
