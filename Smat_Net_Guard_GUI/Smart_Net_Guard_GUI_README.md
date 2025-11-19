# SmartNetGuard — Live Run GUI  
**Real-Time Visualization, Alerts & Event Dashboard (Qt + PyQtGraph)**

---

## Overview

`SmartNetGuard_GUI.py` provides a **real-time graphical interface** for the full SmartNetGuard inference pipeline.

It serves as a thin orchestration/observability layer on top of:

- the **SmartNetGuard runner** (`Smart_Net_Guard_Run_2.py`)
- L1 AutoEncoder (reconstruction & embeddings)
- L2 classifier (dual-input Conv1D + L1 embeddings)
- ZD-Gate (Zero-Day detection)
- Policy engine
- Event stitching system

The GUI allows:

- Live or offline PCAP-based inference  
- Real-time visualization of reconstruction error (MSE)  
- Per-window anomaly & classification monitoring  
- Zero-Day recognition  
- Event creation with PNG/TXT/JSON artifacts  
- Optional email alerts  
- Interactive interface for adjusting thresholds and policy settings  

---

## Key Features

### 🔥 1. Real-Time MSE Visualization
- PyQtGraph bar-chart updated **25 FPS**
- EMA smoothing (α adjustable in UI)
- Percentile-based normalization for stable scaling
- Color-coded bars:
  - **Green** — low anomaly
  - **Yellow** — suspicious
  - **Red** — high anomaly
- Smooth animation (interpolation between frames)

---

### 🔥 2. Window-Level Classification + Event Detection
Each window from the runner includes:

- predicted class  
- pmax  
- reconstruction MSE  

GUI uses this to:

- track **per-class streaks**  
- apply refractory logic  
- detect stable events (≥ N confident windows)  
- trigger alert actions  

---

### 🔥 3. Event Artifact Generation
Each detected event creates:

- **PNG snapshot** of the MSE plot  
- **TXT human-readable summary**  
- **JSON machine-readable record**  

Artifact contains:

| Field | Description |
|-------|-------------|
| raw_class | class label returned by the policy/runner |
| canonical_class | normalized class (e.g., unknown → unknown_anomaly) |
| is_zero_day | Zero-Day flag |
| pmax | maximum predicted probability |
| recon MSE | raw + normalized |
| severity | low / medium / high |
| response_hint | recommended incident response |
| raw_line | full runner log line |

---

### 🔥 4. Zero-Day / Unknown Detection
The GUI applies semantic normalization:

```
unknown, zeroday, unknown_anomaly, NoAnimals → unknown_anomaly
```

Unknown anomalies trigger:

- distinct visualization  
- dedicated event folder  
- email alert with Zero-Day subject prefix  
- stronger recommended actions  

---

### 🔥 5. Email Notifications (Optional)
The GUI can send:

- immediate alerts  
- Zero-Day notifications  
- with MSE info & response hints  

SMTP settings are configurable:

- email_to  
- email_from  
- app password  
- SMTP host  
- SMTP port  

Supports STARTTLS.

---

### 🔥 6. Live Sniffer Mode
The GUI can run SmartNetGuard on live traffic:

```
source = sniffer
iface  = Ethernet / Wi-Fi name
```

Supports:

- JSON streaming  
- continuous window ingestion  
- continual alerts  

---

### 🔥 7. Offline PCAP Mode
User selects one or more PCAP files:

GUI will:

- create separate output folders
- run inference sequentially per file
- generate:
  - window dumps  
  - event dumps  
  - logits  
  - MSE curves  
  - artifacts  

---

## How It Works Internally

### Packet Flow
```
PCAP / Sniffer
   ↓
SmartNetGuard Runner
   ↓  (JSON/STDOUT lines)
GUI Worker Thread
   ↓
Parser (_parse_line)
   ↓
Signals (Qt)
   ↓
Visualization + Event Logic
```

### Event Flow
```
Window-level predictions
   ↓
Streak counter
   ↓
Refractory logic
   ↓
Event created
   ↓
Popup dialog (blinking)
   ↓
Artifact folder
      PNG snapshot
      TXT summary
      JSON record
   ↓
Email alert (optional)
```

---

## UI Elements

### Inputs & Outputs
- Choose output directory  
- Select PCAP files  
- Enter live interface name  

### Flags & Parameters
- Pre-L2 filter  
- Event stitching on/off  
- Policy debug  
- JSON streaming  
- Stitch_min_len  
- Stitch_max_gap  
- confident_pmax  
- refractory_sec  
- max alerts per class  
- EMA α  
- warn / alarm thresholds  

### Email Configuration
- From / To  
- App password  
- SMTP host + port  

---

## Runner Argument Builder

GUI dynamically generates CLI args for the runner:

### For PCAP:
```
--source pcap
--pcap <file>
--pcap_flush_pkts 800000
--pcap_flush_sec 15
--bin 0.05
--all_windows
--l1_run <path>
--l2_run <path>
--policy_preset default
… etc …
```

### For Live:
```
--source sniffer
--iface Ethernet
--all_windows
--zd_gate on
--policy_debug
--stream_json
```

All calibration bias settings are created automatically into:

```
<OUT>/calib_bias_windows_run.json
```

---

## Popup Alerts

GUI shows:

- Blinking red popup
- Class + canonical class
- pmax
- MSE raw + normalized
- Timestamp
- Recommendations

For Zero-Day, title includes ZeroDay.

---

## Folder Structure Created by GUI

```
OUT/
 └── run_YYYYMMDD_HHMMSS/
      ├── windows_mse.csv
      ├── windows_probs.csv
      ├── logit_*.csv
      ├── events/
      │     ├── http_flood/
      │     │       event_*.png
      │     │       event_*.json
      │     │       event_*.txt
      │     ├── volumetric_flood/
      │     ├── bot/
      │     ├── portscan/
      │     └── unknown_anomaly/
      └── coords/
            window_*.csv
```

---

## Starting the GUI

```bash
python SmartNetGuard_GUI.py
```

On Windows, taskbar icon is set using:

```python
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SmartNetGuard.GUI")
```

---

## Requirements

- Python 3.10+
- PyQt5 / PySide2 (provided via `pyqtgraph`)
- PyQtGraph
- NumPy
- SmartNetGuard runner & models (L1/L2)
- Windows/Linux/macOS

---

## License

Distributed under the **MIT License** as part of the SmartNetGuard project.

---

## Author

**SmartNetGuard — Network Threat Detection Pipeline (L1 + L2)**  
Author: **Artiom Maliovanii**

