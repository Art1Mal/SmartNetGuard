# SmartNetGuard — Network Threat Detection Pipeline (L1 Anomaly Detection + L2 Classification + Policy Engine)

**SmartNetGuard** is a real-time, production-ready **Network Detection & Response (NDR)** pipeline built as a hybrid ML + policy engine system.  
It performs **behavioral anomaly detection**, **attack classification**, **Zero-Day detection**, **event stitching**, and **incident response hints** — all in a fully automated end-to-end pipeline with GUI support.

This project was originally designed as an **academic capstone**, but evolved into a **market-grade MVP** suitable for SOC workflows, red-team evaluations, and real-time PCAP/live network inference.

---

# 🚀 Project Highlights

### ✔ End-to-end ML pipeline
From raw packets → flows → windows → L1 → L2 → policy → events.

### ✔ Zero-Day awareness
Unknown anomalies handled via entropy, pmax, logit margin & OOD logic.

### ✔ Real-time visualization GUI
Qt + PyQtGraph equalizer-style MSE monitor, popups, email alerts.

### ✔ Full packet-to-feature engine
Fast FeatureTransformer with TLS SNI/ALPN/JA3 metadata extraction.

### ✔ Production-grade runner
Chunked PCAP processing, live sniffer mode, automatic binning & clipping.

### ✔ Safety-focused design
No payload stored. Only metadata, counts, lengths, timing, TLS fingerprints.


---

# 🧠 Architecture Overview

SmartNetGuard consists of **three logical layers**:

      ┌──────────────────────────┐
      │  L1 AutoEncoder (AE)     │
      │  - BASE7 forward stats   │
      │  - Behavioral anomaly    │
      └─────────────┬────────────┘
                    │ recon_mse
                    ▼
           ┌──────────────────────┐
           │  Policy Pre-Filter   │
           │  - Recon budget      │
           │  - Zero-Day gate     │
           │  - OOD detector      │
           └─────────────┬────────┘
                         │ filtered windows
                         ▼
            ┌─────────────────────────┐
            │     L2 Classifier       │
            │  (Conv1D time model)    │
            │  • volumetric_flood     │
            │  • http_flood           │
            │  • bot                  │
            │  • portscan             │
            └─────────────┬───────────┘
                          │ class + logits
                          ▼
                ┌───────────────────────┐
                │   Event Stitching     │
                │   Canonical labeling  │
                │   Zero-Day reasoning  │
                └─────────────┬─────────┘
                              │ events
                              ▼
                 ┌──────────────────────────┐
                 │       GUI / Alerts       │
                 │  Snapshots, PNG/TXT/JSON │
                 │  Email SOC notifications │
                 └──────────────────────────┘


---

# 🧩 Component Breakdown

## 1) **Flow Transformer (`the_flow_transformer.py`)**
Turns packets into **uniform flow-time windows**, producing exactly the feature set expected by L1/L2.

### Features produced:
- Forward statistics: **tot_fwd_pkts**, **fwd_pkt_len_mean**, …
- Backward statistics
- Timing: **flow_iat_mean**, **bwd_iat_mean**
- Rates: **flow_pkts_per_sec**
- TLS: **SNI**, **ALPN**, **JA3 raw**

---
## 📦 Dependencies

SmartNetGuard requires the following core libraries:

- **NumPy** — numerical operations for packet/flow statistics.
- **Pandas** — tabular data processing for flow windows.
- **SciPy** — additional numerical utilities.
- **TensorFlow / Keras** — L1 AutoEncoder & L2 Classifier inference.
- **PyQt6** — GUI framework for real-time visualization and controls.
- **PyQtGraph** — high-performance plotting (reconstruction error bar-graph).
- **Scapy** — optional packet parsing for LIVE sniffer mode.
- **Matplotlib** — used internally for snapshots and event artifacts.
- **psutil** — system information helper for runner utilities.

All dependencies are listed in `requirements.txt`.
---
### Why it is important:
It ensures **deterministic**, **dataset-agnostic**, **runner-compatible** feature formatting.

---

## 2) **L1 AutoEncoder**
Trained on *only normal* traffic, detects anomalies via reconstruction error (MSE).

✔ BASE7 features  
✔ Skip-connections  
✔ MultiHead Attention  
✔ Weighted IAT  
✔ Chunked training (1M windows per epoch)

Output:  
`recon_mse`

---

## 3) **Policy Layer**
Transforms raw ML outputs into decisions.

### Components:
- **Pre-L2 filter** (MSE threshold budget)
- **Zero-Day gate** (entropy, pmax, margin)
- **OOD module** (alpha, beta, gamma)
- **Bias calibration** (logit offsets per class)

Output:
Adjusted logits + filtered windows.

---

## 4) **L2 Classifier**
Conv1D model that classifies window sequences into:

- volumetric_flood  
- http_flood  
- bot  
- portscan  

Handles window temporal context (sequence length 50 by default).

---

## 5) **Event Processor**
- Streak tracking (8–10 windows)
- Refractory logic (15s)
- Event stitching by min_len + max_gap
- Creates artifacts:  
  - PNG (MSE snapshot)  
  - TXT (human-readable summary)  
  - JSON (for retraining)

Supports Zero-Day classification.

---

## 6) **GUI (`SmartNetGuard_GUI.py`)**
Live Qt interface with:

- Dynamic MSE equalizer
- Live PCAP or live interface ingestion
- Per-window logs + class predictions
- Blinking incident popups
- Email incident delivery
- Time-synced snapshots
- Adjustable thresholds

Ideal for demos, SOC operators, or academic presentations.

---

# 💾 Running SmartNetGuard

## 1) **Install dependencies**
```bash
pip install -r requirements.txt
```
---
## 2) Run the GUI
```
python SmartNetGuard_GUI.py
```
---
## 3) Run inference on a PCAP
```
python Smart_Net_Guard_Run.py \
    --source pcap \
    --pcap path/to/file.pcap \
    --l1_run L1_ready_for_working/run_xxx \
    --l2_run L2_ready_for_working/run_yyy \
    --bin 0.05 \
    --all_windows \
    --dump csv
```
---
## 4) Run in LIVE mode
```
python Smart_Net_Guard_Run.py \
    --source sniffer \
    --iface Ethernet \
    --l1_run L1_ready_for_working/run_xxx \
    --l2_run L2_ready_for_working/run_yyy
```
---
### 🧪 Datasets Used
SmartNetGuard has been trained/tested on:
  * CICIDS2017
  * UNSW-NB15
  * CSE-CIC-IDS2018
  * Bot-IoT
  * Custom enterprise PCAP traffic

Each dataset was unified via:
✔ column normalization
✔ BASE7 forward mapping
✔ time binning
✔ TLS metadata extraction

---
### 🛡 Zero-Day Detection Strategy
SmartNetGuard identifies unknown threats using:
  * pmax threshold (≤0.60 → anomaly)
  * entropy threshold (>1.20 → unstable prediction)
  * margin threshold (<0.09 → no dominant class)
  * OOD score (distance-based)
  * Stitch-time consistency
Zero-Day events are labeled unknown_anomaly and treated with stricter policy.

---
### 🔔 Output Artifacts
For every event, SmartNetGuard stores:
```
events/
   http_flood/
      event_20251010_141233.png
      event_20251010_141233.txt
      event_20251010_141233.json
   unknown_anomaly/
      ...
```
TXT includes SOC recommendations
JSON is used for future retraining

---
### 🛠 Development Status
| Component        | Status             | Notes                       |
| ---------------- | ------------------ | --------------------------- |
| L1 AutoEncoder   | ✔ stable           | no retraining required      |
| L2 Classifier    | ✔ stable           | tuned logits & bias         |
| Policy Engine    | ✔ stable           | Zero-Day tuned pmax/entropy |
| Flow Transformer | ✔ production-ready | TLS support                 |
| GUI              | ✔ demo-ready       | live sniffer, PCAP playback |

---
### 📄 License
This project is distributed under the MIT License.

---
###  Author
Artiom Maliovanii
SmartNetGuard · Network Threat Detection Pipeline
L1 AutoEncoder + L2 Classifier + Zero-Day Policy Engine

---
### How to Cite / Reference
If you use SmartNetGuard in academic work:
```
Maliovanii, A. (2025). SmartNetGuard: Network Threat Detection Pipeline (L1+L2+Policy).
GitHub Repository.
```
---
