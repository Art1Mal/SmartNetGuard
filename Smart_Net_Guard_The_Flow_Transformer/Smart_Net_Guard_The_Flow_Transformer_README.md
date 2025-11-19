# SmartNetGuard — Flow Transformer  
**Packet-to-Feature Engine Feeding L1 AutoEncoder & L2 Classifier**

---

## Overview

`the_flow_transformer.py` implements **FeatureTransformer**, the packet-to-flow feature extractor used in the SmartNetGuard pipeline.

It converts a raw packet stream (live or offline PCAP decode) into **uniform flow-time windows (“bins”)** and computes the exact feature set required by:

- **L1 AutoEncoder** (BASE7)
- **L2 Classifier** (BASE7 + backward features + timing + TLS metadata)

This module is the **first stage** of the SmartNetGuard pipeline.

---

## Key Responsibilities

- Accept packets in real time via:

```python
transformer.ingest_packet(ts, src, sport, dst, dport, proto, size, raw_bytes=None)
```
* Group packets into fixed-size time bins (e.g., 100 ms)
* Track flows in both forward and backward directions
* Compute:
  * Forward statistics (BASE7)
  * Backward statistics
  * Timing metrics (IAT)
  * Rate metrics (packets/sec)
  * Additive counters (totals)
* Optionally extract TLS metadata:
  * SNI
  * ALPN
  * JA3 raw fingerprint
* Periodically flush bins into a DataFrame:
```
df = transformer.flush()
# or:
df = transformer.flush_older_than(now_ts, max_age_sec)
```
Why It Exists in SmartNetGuard
SmartNetGuard uses two ML models:
L1 AutoEncoder
Detects anomalies using the 7 forward BASE features.
L2 Classifier
Classifies anomalous windows into:
* volumetric_flood
* http_flood
* bot
* portscan
🔥 The Flow Transformer guarantees both models receive deterministic, clean, window-aggregated features.

---

Output DataFrame Schema
Each row represents one flow × one time bin.
Meta Columns
| Column        | Meaning                            |
| ------------- | ---------------------------------- |
| src/dst       | IP addresses                       |
| sport/dport   | ports                              |
| proto         | L4 protocol number (6=TCP, 17=UDP) |
| time_index    | integer bin index                  |
| ts_edge       | bin start timestamp (sec)          |
| flow_duration | bin width in microseconds          |

---

Forward Direction (BASE7)
Core L1 features:
* tot_fwd_pkts
* totlen_fwd_pkts
* fwd_pkt_len_max
* fwd_pkt_len_mean
* flow_iat_mean (µs)
* flow_pkts_per_sec
* flow_duration (µs)
Forward direction is fixed by the first packet seen.

---

Backward Direction
Used heavily for L2 classifier:
* tot_bwd_pkts
* totlen_bwd_pkts
* bwd_pkt_len_max
* bwd_pkt_len_mean
* bwd_iat_mean (µs)
If no backward packets → zeros are emitted (safe for model).

---

Additive Counters
* flow_pkts_total
* flow_bytes_total

---

TLS Metadata (Optional)
If packet raw_bytes are given, transformer extracts:
| Field      | Meaning                                   |
| ---------- | ----------------------------------------- |
| `is_tls`   | 1 if TLS ClientHello/ServerHello detected |
| `tls_sni`  | Server Name Indication                    |
| `tls_alpn` | ALPN protocol                             |
| `tls_ja3`  | JA3 raw signature (without MD5)           |
Supported handshake types:
* ClientHello (1)
* ServerHello (2)
TLS is best-effort and optional.

---

Time Binning
Packet timestamp is converted into a time bin:
```
bidx = floor(ts / time_bin_sec)
```
Default:
```
time_bin_sec = 0.1   # 100 ms
```
Each flow holds:
```
bins: Dict[int, BinAgg]
```

---

Timestamp Unit Autodetection
Datasets mix seconds and microseconds.
Transformer auto-detects format:
* If ts > 1e12 → timestamp already in µs
* Else → seconds → converted to µs
Ensures consistent IAT in microseconds across all datasets.

---

Inter-Arrival Timing (IAT)
Forward/backward IAT computed separately:
```
iat_us = (ts - last_ts) * factor
```
At flush:
* Forward IAT mean
* Backward IAT mean
* Weighted combined flow_iat_mean
Fallback (if only one packet):
```
flow_iat_mean = bin_duration_us / max(total_pkts - 1, 1)
```
---

Internal Structures
BinAgg — per-bin aggregator
Stores:
* packet counts
* byte sums
 max lengths
* IAT sums
* ts_edge
---

FlowState — per-flow state
Stores:
* flow key
* forward direction definition
* TLS metadata
* last timestamps (for IAT)
* bins dictionary
---

FeatureTransformer — main engine
* groups packets into bins
* accumulates stats
* performs TLS extraction
* flushes bins into DataFrame
* manages memory
---

API Methods
ingest_packet(...)
Accepts one packet.
Arguments include:
* timestamp (sec or µs)
* IPs and ports
* protocol
* size
* raw_bytes (optional TLS)
---

flush()
Full memory drain.
* Converts all bins → rows
* Clears all internal state
* Performs IAT consistency check
* Returns DataFrame
---

flush_older_than(now_ts, max_age_sec)
Partial drain for long-running streaming.
* Removes bins older than given threshold
* Keeps active flow state intact
* Returns DataFrame (possibly empty)
---

Example Usage
Offline PCAP Processing
```
from the_flow_transformer import FeatureTransformer

transformer = FeatureTransformer(time_bin_sec=0.1)

for pkt in decode_pcap("file.pcap"):
    transformer.ingest_packet(
        ts=pkt.ts,
        src=pkt.src,
        sport=pkt.sport,
        dst=pkt.dst,
        dport=pkt.dport,
        proto=pkt.proto,
        size=pkt.length,
        raw_bytes=pkt.raw_bytes
    )

df = transformer.flush()
```

---

Streaming Mode
```
import time
from the_flow_transformer import FeatureTransformer

transformer = FeatureTransformer(time_bin_sec=0.1)

while True:
    for pkt in receive_stream():
        transformer.ingest_packet(...)

    df_old = transformer.flush_older_than(time.time(), max_age_sec=10)
    if not df_old.empty:
        process(df_old)
```

---
Performance & Safety
* No sensitive data stored
* No payload contents saved
* TLS metadata is minimal and safe
* Very low memory footprint
* Deterministic output via stable sorting
* Fully safe for public GitHub repository

---
Publishing Notes
This module:
* Contains no secrets
* Contains no private IPs
* Stores no payload data
* Is fully safe for open-source release

---
License
Distributed under the MIT License as part of SmartNetGuard.

---
Author
SmartNetGuard — Network Threat Detection Pipeline (L1 + L2)
Author: Artiom Maliovanii
