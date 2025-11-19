# SmartNetGuard — Unified Runner (Pipeline Switch)
**End-to-End Orchestrator for L1 AutoEncoder, L2 Classifier, ZD-Gate, Policy Router & Event Engine**

---

## Overview

`smartnetguard_runner.py` is the **main orchestration entry point** for the entire SmartNetGuard production-grade inference pipeline.

This script glues together:

- **Packet → Feature Transformer (FlowTransformer)**
- **L1 AutoEncoder (embeddings + reconstruction error)**
- **L2 Dual-Input Classifier**
- **Zero-Day Gate (probability–entropy–margin checks)**
- **Event Stitcher (window → event aggregation)**
- **Context-aware Policy Router**
- **OOD-Lite detector**
- **Logging, Live Dumps, Confident Dumps**
- **Optional email alert system**
- **Sniffer / PCAP processing modes**

It is the **canonical entry point** for running SmartNetGuard on live traffic or PCAPs.

---

## Key Responsibilities

### 1. Packet → Feature Engineering
Uses:

- `FeatureTransformer`  
- Time-binned flows  
- Forward/Backward packet statistics  
- Timing features (µs-level IAT)  
- TLS ALPN/SNI/JA3 metadata  
- Per-flow IPL-based slicing  

Output: **per-flow per-bin DataFrame**

---

### 2. L1 AutoEncoder (Anomaly Detector)
The runner loads:

- embedding head  
- reconstruction head  
- preprocessing_config.json (means/scales/z-clips)

L1 responsibilities:

- Standardize BASE features → Z-space  
- Compute:
  - bottleneck embeddings  
  - reconstruction MSE and MAE  
- Pre-L2 filter using:
  - `--recon_thresh`
  - `--recon_budget`
  - `--recon_min_mse`

High reconstruction error → suspicious.

---

### 3. L2 Dual-Input Classifier

Input branches:

1. **Windowed feature sequence** (Conv1D)
2. **L1 embedding vector**

Outputs:

- logits (4 classes)
- calibrated logits (optional)
- probabilities
- uncertainties

Classes:

```
volumetric_flood
http_flood
bot
portscan
```

Supports:

- Temperature scaling  
- Bias correction  
- Policy-softmax (`--policy_T`)  
- Feature auto-fix (`--l2_feat_autofix`)  

---

### 4. Zero-Day Gate (ZDGate)

Flags suspicious windows via:

- low pmax
- high entropy
- low top1–top2 margin

Configurable via:

```
--zd_pmax
--zd_entropy
--zd_margin
```

ZD windows can trigger:

- event ZD flag  
- fallback to unknown class  
- additional OOD-lite checks  

---

### 5. Sliding Window System

Two modes:

- **last_windows_per_flow** (fast)
- **sliding_windows_per_flow** (full enumeration)

Window size: **112 samples**.

---

### 6. Event Stitcher

Groups windows into **stable high-level events**:

- merges consecutive windows  
- enforces max gap  
- enforces min length  
- outputs an event object:

```
flow key
label
zd flag
pmax stats
MSE stats
start/end idx
timestamps
```

---

### 7. Context-Aware Policy Router

For each event evaluates:

- HTTP semantics  
- volumetric semantics  
- bot-service-port semantics  
- portscan diversity  
- TLS metadata  
- consistency between pkt-rate and IAT  

May:

- confirm class  
- flip class (HTTP ↔ VOL)  
- downgrade to unknown  
- force Zero-Day  
- attach mismatch flags  

---

### 8. OOD-Lite (Event-Level Novelty)

Score:

```
score = α * MSE_norm + β * (1 - pmax) + γ * ctx_mismatch
```

If exceeds `--ood_thresh`:

- event becomes `unknown_dos`
- optionally flagged as ZD

---

### 9. Output System

Supports:

- Console logs
- Window JSON stream (`--stream_json`)
- Event JSON stream
- Live dumps (CSV/Parquet)
- Confident-window dumps (per class + ZD)
- Email alerts (SMTP)
- Logits CSV dump
- Event CSV/Parquet dumps

All dumps include:

- raw logits  
- calibrated logits  
- probabilities  
- reconstruction error  
- per-window context metrics  

---

## Pipeline Structure

```
Packets (PCAP / Sniffer)
      ↓
FeatureTransformer
      ↓
Derived Features + Normalization
      ↓
L1 AutoEncoder
   ├─ Embeddings
   └─ Reconstruction (MSE/MAE)
      ↓
Pre-L2 Filter
      ↓
Temporal Windows (window=112)
      ↓
L2 Classifier (Dual Input)
      ↓
Calibration (T/Bias)
      ↓
ZDGate & OOD-Lite
      ↓
Window-Level Items
      ↓
EventStitcher (aggregate windows → events)
      ↓
Policy Router (context-aware semantics)
      ↓
Final Events
      ↓
Logging / Dumps / Alerts / Stream
```

---

## Supported Modes

### Sniffer Mode

```
python smartnetguard_runner.py --source sniffer --iface auto ...
```

Automatically selects the best network interface.

### PCAP Mode

```
python smartnetguard_runner.py --source pcap --pcap traffic.pcap ...
```

Supports:

- .pcap  
- .pcapng  
- time limiting  
- packet count limiting  
- periodic transformer flush  

---

## Essential CLI Arguments

### Data Source
| Option | Description |
|--------|-------------|
| `--source sniffer/pcap` | Input mode |
| `--iface` | Sniffer interface |
| `--pcap` | PCAP file path |

### Models
| Option | Description |
|--------|-------------|
| `--l1_run` | Path to L1 model dir |
| `--l2_run` | Path to L2 model dir |

### ZD Gate
| Option | Purpose |
|--------|---------|
| `--zd_gate on/off` | Enable ZD gate |
| `--zd_pmax` | threshold for pmax |
| `--zd_entropy` | entropy threshold |
| `--zd_margin` | margin threshold |

### OOD-Lite
| Option | Purpose |
|--------|---------|
| `--ood on/off` | Enable OOD |
| `--ood_alpha` | weight for MSE |
| `--ood_beta` | weight for uncertainty |
| `--ood_gamma` | weight for mismatch |
| `--ood_thresh` | global threshold |

### Windowing
| Option | Description |
|--------|-------------|
| `--all_windows` | Use sliding windows |
| `--win_batch` | Microbatch window batch size |

### DUMPS
| Option | Description |
|--------|-------------|
| `--dump csv/parquet/none` | Save live dump |
| `--dump_logits on/off` | Save logits |
| `--confident_dump` | Save confident windows per class |
| `--out` | Output directory |

### Email Alerts
| Option | Description |
|--------|-------------|
| `--email_alerts on/off` | Enable alerts |
| `--email_to` | Recipients |
| `--email_user` | SMTP login |

---

## Output Files (Examples)

```
out_live/
    live_*.csv
    events_*.csv
    logits_*.csv
    windows_mse.csv
    confident/
        volumetric_flood/win_*.csv
        http_flood/win_*.csv
        bot/win_*.csv
        portscan/win_*.csv
        zd/win_*.csv
```

---

## Safety and Privacy

The pipeline does **not** store:

- packet payloads (except raw TLS handshake for ALPN/SNI)
- PII
- internal customer data
- secrets

Only statistical features are saved.

---

## Example Run (PCAP)

```
python smartnetguard_runner.py \
    --source pcap \
    --pcap Friday-WorkingHours.pcap \
    --l1_run L1_ready \
    --l2_run L2_ready \
    --dump csv \
    --zd_gate on \
    --ood on \
    --policy_debug
```

---

## Example Run (Live Sniffer)

```
python smartnetguard_runner.py \
    --source sniffer \
    --iface auto \
    --l1_run run_l1 \
    --l2_run run_l2 \
    --stream_json \
    --email_alerts on
```

---

## Author & License

- **SmartNetGuard — Network Threat Detection Pipeline**  
- Author: **Artiom Maliovanii**  
- License: **MIT**

