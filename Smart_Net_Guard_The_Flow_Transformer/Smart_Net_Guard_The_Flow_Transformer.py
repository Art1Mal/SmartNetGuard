# the_flow_transformer.py
# -*- coding: utf-8 -*-

"""
FeatureTransformer is a traffic packetizer into uniform time bins and a basic feature calculator.

Key points:
- IAT is calculated and returned in microseconds (ms) - this is our norm.
- Optional parsing of TLS ClientHello/ServerHello for SNI/ALPN/JA3 (if raw_bytes are passed).
- The output DataFrame contains EXACTLY those columns that the L1/L2 runners are expecting.

Purpose:
    runner calls:
        transformer.ingest_packet(ts, src, sport, dst, dport, proto, size, raw_bytes=None)
    and periodically:
        df = transformer.flush()
    (or alternatively: df = transformer.flush_older_than(now_ts, max_age_sec) - see method below)

Output DataFrame:
  meta:     src, sport, dst, dport, proto, time_index, ts_edge, flow_duration
  forward:  tot_fwd_pkts, totlen_fwd_pkts, fwd_pkt_len_max, fwd_pkt_len_mean
  backward: tot_bwd_pkts, totlen_bwd_pkts, bwd_pkt_len_max, bwd_pkt_len_mean
  units: flow_pkts_per_sec, flow_iat_mean (µs), bwd_iat_mean (µs)
  additives:  flow_pkts_total, flow_bytes_total
  TLS: is_tls (0/1), tls_sni, tls_alpn, tls_ja3 (JA3 raw string without MD5)

TLS Notes:
- We only parse ClientHello/ServerHello (record type 22, handshake types 1 and 2).
- We form JA3 as a “raw” CSV (version, ciphers, extensions, elliptic_curves, ec_point_formats), we do not calculate MD5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import math, struct
import pandas as pd


# =============================================================================
# TLS parsing helpers (minimalistic header parsing – fast and secure)
# =============================================================================

def _is_tls_record_clienthello(buf: bytes) -> bool:
    """
    Quick check: first TLS record = Handshake(ClientHello)?
    - Checking ContentType=22.
    - We shift the record header by 5 bytes.
    - We look at the first byte HandshakeType == 1 (ClientHello).
    """
    try:
        if len(buf) < 6:
            return False
        if buf[0] != 22:  # TLS Handshake record
            return False
        # The length of the recording is not critical to us - we only read the first fragment
        # rec_len = struct.unpack("!H", buf[3:5])[0]
        hs_off = 5
        if len(buf) < hs_off + 1:
            return False
        return buf[hs_off] == 1  # ClientHello
    except Exception:
        return False


def _is_tls_record_serverhello(buf: bytes) -> bool:
    """
    Similar, but HandshakeType == 2 (ServerHello).
    """
    try:
        if len(buf) < 6:
            return False
        if buf[0] != 22:
            return False
        hs_off = 5
        if len(buf) < hs_off + 1:
            return False
        return buf[hs_off] == 2  # ServerHello
    except Exception:
        return False


def _parse_varlen(data: bytes, off: int, size_bytes: int) -> Tuple[bytes, int]:
    """
    Variable length parser for format fields:
      [len (size_bytes)] [payload(len)]
    Returns (payload, new_offset). If there is not enough data, an empty payload and the previous off.
    """
    if off + size_bytes > len(data):
        return b"", off
    ln = int.from_bytes(data[off:off + size_bytes], "big")
    off += size_bytes
    end = min(len(data), off + ln)
    return data[off:end], end


def _parse_client_hello(buf: bytes) -> dict:
    """
    Simplified parsing of ClientHello → SNI / ALPN / JA3 (raw).
    Returns a dictionary: {'sni': str|None, 'alpn': str|None, 'ja3_raw': str|None}
    We don't validate all TLS fields; the goal is to cheaply extract the minimum amount of useful metainformation.
    """
    out = {"sni": None, "alpn": None, "ja3_raw": None}
    try:
        # record header(5) + handshake header(4)
        off = 5
        if len(buf) < off + 4:
            return out
        hs_type = buf[off]
        if hs_type != 1:
            return out
        off += 1
        _hs_len = int.from_bytes(buf[off:off + 3], "big")
        off += 3

        # client_version(2)
        if off + 2 > len(buf):
            return out
        client_version = struct.unpack("!H", buf[off:off + 2])[0]
        off += 2

        # random(32)
        off += 32

        # session_id, cipher_suites, compression_methods, extensions
        _sess, off = _parse_varlen(buf, off, 1)
        cs, off = _parse_varlen(buf, off, 2)
        _comp, off = _parse_varlen(buf, off, 1)
        exts_blob, off = _parse_varlen(buf, off, 2)

        # --- JA3 components ---
        # 1) version
        ja3_version = str(client_version)
        # 2) ciphers
        ciphers = []
        for i in range(0, len(cs), 2):
            ciphers.append(str(struct.unpack("!H", cs[i:i + 2])[0]))
        ja3_ciphers = "-".join(ciphers) if ciphers else ""

        # Parse extensions (SNI / ALPN / supported_groups / ec_point_formats)
        sni_host = None
        alpn_proto = None
        ext_ids = []
        elliptic_curves = []
        ec_point_formats = []

        ex_off = 0
        while ex_off + 4 <= len(exts_blob):
            ext_type = struct.unpack("!H", exts_blob[ex_off:ex_off + 2])[0]
            ext_len = struct.unpack("!H", exts_blob[ex_off + 2:ex_off + 4])[0]
            ex_off += 4
            ext_data = exts_blob[ex_off:ex_off + ext_len]
            ex_off += ext_len
            ext_ids.append(str(ext_type))

            if ext_type == 0x00:  # SNI
                ed_off = 2  # list_len(2)
                while ed_off + 3 <= len(ext_data):
                    name_type = ext_data[ed_off]
                    name_len = struct.unpack("!H", ext_data[ed_off + 1:ed_off + 3])[0]
                    ed_off += 3
                    name = ext_data[ed_off:ed_off + name_len]
                    ed_off += name_len
                    if name_type == 0 and name:
                        try:
                            sni_host = name.decode("utf-8", "ignore")
                        except Exception:
                            sni_host = None

            elif ext_type == 0x10:  # ALPN
                ed_off = 2  # list_len(2)
                if ed_off + 1 <= len(ext_data):
                    l = ext_data[ed_off]
                    if ed_off + 1 + l <= len(ext_data):
                        try:
                            alpn_proto = ext_data[ed_off + 1:ed_off + 1 + l].decode("utf-8", "ignore")
                        except Exception:
                            alpn_proto = None

            elif ext_type == 0x0a:  # supported_groups (elliptic_curves)
                if len(ext_data) >= 2:
                    n = struct.unpack("!H", ext_data[0:2])[0]
                    ed_off = 2
                    while ed_off + 2 <= min(2 + n, len(ext_data)):
                        ec = struct.unpack("!H", ext_data[ed_off:ed_off + 2])[0]
                        elliptic_curves.append(str(ec))
                        ed_off += 2

            elif ext_type == 0x0b:  # ec_point_formats
                if len(ext_data) >= 1:
                    n = ext_data[0]
                    ed_off = 1
                    while ed_off < 1 + n and ed_off < len(ext_data):
                        ec_point_formats.append(str(ext_data[ed_off]))
                        ed_off += 1

        ja3_exts = "-".join(ext_ids) if ext_ids else ""
        ja3_ec = "-".join(elliptic_curves) if elliptic_curves else ""
        ja3_ecpf = "-".join(ec_point_formats) if ec_point_formats else ""
        ja3_raw = ",".join([ja3_version, ja3_ciphers, ja3_exts, ja3_ec, ja3_ecpf])

        out["sni"] = sni_host
        out["alpn"] = alpn_proto
        out["ja3_raw"] = ja3_raw
    except Exception:
        # We simply ignore any parsing errors - this information is secondary and optional
        pass
    return out


# =============================================================================
# State Models: Bin Accumulator and Flow State
# =============================================================================

@dataclass
class BinAgg:
    """
    Aggregates for one time bin for one flow (forward/backward separately).
    We store base counters and adders to quickly get averages/maximums.
    """
    # forward (in the direction of the primary packet in this flow)
    f_pkts: int = 0
    f_bytes: int = 0
    f_len_sum: int = 0
    f_len_max: int = 0
    f_iat_us_sum: float = 0.0
    f_iat_cnt: int = 0

    # backward
    b_pkts: int = 0
    b_bytes: int = 0
    b_len_sum: int = 0
    b_len_max: int = 0
    b_iat_us_sum: float = 0.0
    b_iat_cnt: int = 0

    # meta
    ts_edge: float = 0.0  # start bin (unix ts, sec)


@dataclass
class FlowState:
    """
    State of one TCP/UDP flow (defined by 5 fields).
    first_dir_srcdst fixes the "forward" direction to ensure that everything is calculated consistently.
    """
    key: Tuple[str, int, str, int, int]
    first_dir_srcdst: Tuple[str, int, str, int]  # defines "forward"
    last_ts_fwd: Optional[float] = None
    last_ts_bwd: Optional[float] = None
    tls_sni: Optional[str] = None
    tls_alpn: Optional[str] = None
    tls_ja3: Optional[str] = None
    is_tls: int = 0

    # per-bin aggregates
    bins: Dict[int, BinAgg] = field(default_factory=dict)


# =============================================================================
# Basic transformer class
# =============================================================================

class FeatureTransformer:
    """
    Packetizer: accepts packets one by one, distributes them into uniform time windows (bin indices),
    accumulates fast aggregates (counters/maximums/averages), optionally enriches with TLS metadata.
    """

    def __init__(self, time_bin_sec: float = 0.1):
        """
        time_bin_sec: time window width in seconds (e.g. 0.1 = 100 ms).
        Note: IAT is returned in microseconds (µs), which is what the runner expects.
       """
        self.bin = float(time_bin_sec)
        self.flows: Dict[Tuple[str, int, str, int, int], FlowState] = {}
        # IAT is always in µs (see multiplication in flush/flush_older_than)
        self.iat_unit = "us"
        self.us = 1_000_000.0
        self.signature = "FT-REV:2025-10-25-bwd-fix"
        print("[FT] Using FeatureTransformer", self.signature)

    # ---- internal helpers ----
    def _bin_index(self, ts: float) -> int:
        """
        Integer index of the bin into which the packet with Unix time ts falls.
        Example: when bin=0.1, the interval [0.2, 0.3) corresponds to index 2.
        """
        return int(math.floor(ts / self.bin))

    def _is_forward(self, st: FlowState, src: str, sport: int, dst: str, dport: int) -> bool:
        """
        We determine the direction of the packet relative to the “forward” flow (the first packet seen sets forward).
        """
        return (src, sport, dst, dport) == st.first_dir_srcdst

    def _touch_bin(self, st: FlowState, bidx: int, ts: float) -> BinAgg:
        """
        We get (or create) an aggregator for a specific bin of a given flow.
        """
        bagg = st.bins.get(bidx)
        if bagg is None:
            bagg = BinAgg(ts_edge=(bidx * self.bin))
            st.bins[bidx] = bagg
        return bagg

    # ---- TLS capture (optional) ----
    def _maybe_parse_tls(self, st: FlowState, raw_bytes: Optional[bytes]):
        """
        A simple attempt to detect TLS and extract the minimum metadata (SNI/ALPN/JA3) from ClientHello.
        If you have already marked it as TLS, we will not parse it again.
        """
        if raw_bytes is None or st.is_tls:
            return
        try:
            if _is_tls_record_clienthello(raw_bytes):
                info = _parse_client_hello(raw_bytes)
                st.is_tls = 1
                st.tls_sni = info.get("sni") or st.tls_sni
                st.tls_alpn = info.get("alpn") or st.tls_alpn
                st.tls_ja3 = info.get("ja3_raw") or st.tls_ja3
            elif _is_tls_record_serverhello(raw_bytes):
                # ServerHello: SNI/ALPN is not generally available; we only note the TLS fact
                st.is_tls = 1
        except Exception:
            # don't panic - TLS is meta secondary
            pass

    # ---- public API ----
    def ingest_packet(
            self,
            ts: float,
            src: str, sport: int,
            dst: str, dport: int,
            proto: int,
            size: int,
            raw_bytes: Optional[bytes] = None
    ):
        """
        Adds one package to the aggregates.

       Parameters:
          ts — Unix time in seconds (float)
          src/dst — IP strings
          sport/dport — ports (int)
          proto - 6 (TCP) / 17 (UDP) / etc.
          size — the size of the L3/L4 packet (like the runner)
          raw_bytes — optional raw bytes of the IP packet for TLS analysis

        Behavior:
          - We put the package into the bin with index floor(ts/bin).
          - The forward direction is determined by the first flow packet and does not change.
          - We calculate IAT separately for fwd/bwd and accumulate the sum/quantity; we take the average at flush.
          - Dimensions and counters are adjusted only within the corresponding bin.
        """
        #  Symmetrical search for the existing flow state (both directions)
        key1 = (src, sport, dst, dport, int(proto))
        key2 = (dst, dport, src, sport, int(proto))  # reverse direction
        st = self.flows.get(key1) or self.flows.get(key2)

        # If the flow is not yet known, we create it with a “direct” key
        if st is None:
            st = FlowState(key=key1, first_dir_srcdst=(src, sport, dst, dport))
            self.flows[key1] = st

        # TLS (optional, silent)
        self._maybe_parse_tls(st, raw_bytes)

        # Bin Index and Aggregator
        bidx = self._bin_index(ts)
        bagg = self._touch_bin(st, bidx, ts)

        # Normalize the size
        sz = max(0, int(size))

        # Package direction
        is_fwd = self._is_forward(st, src, sport, dst, dport)

        # --- Autodetect time units ---
        # If timestamp > 1e12 → already in microseconds (e.g. from some PCAP CICIDS)
        time_factor = 1.0 if ts > 1e12 else 1_000_000.0

        # IAT is calculated in microseconds
        if is_fwd:
            if st.last_ts_fwd is not None:
                iat_us = (ts - st.last_ts_fwd) * time_factor
                if iat_us >= 0:
                    bagg.f_iat_us_sum += iat_us
                    bagg.f_iat_cnt += 1
            st.last_ts_fwd = ts
            # forward counters
            bagg.f_pkts += 1
            bagg.f_bytes += sz
            bagg.f_len_sum += sz
            bagg.f_len_max = max(bagg.f_len_max, sz)
        else:
            if st.last_ts_bwd is not None:
                iat_us = (ts - st.last_ts_bwd) * time_factor
                if iat_us >= 0:
                    bagg.b_iat_us_sum += iat_us
                    bagg.b_iat_cnt += 1
            st.last_ts_bwd = ts
            # backward counters
            bagg.b_pkts += 1
            bagg.b_bytes += sz
            bagg.b_len_sum += sz
            bagg.b_len_max = max(bagg.b_len_max, sz)

    def flush(self) -> pd.DataFrame:
        """
        Complete "drain" of all accumulated bins across all flows in the DataFrame and CLEARING of memory.

        Returns:
          DataFrame with columns that are waiting for L1/L2 (see file header).
          If nothing has been accumulated, the DataFrame is empty.
        """
        rows: List[dict] = []

        for key, st in self.flows.items():
            src, sport, dst, dport, proto = key
            for bidx, bagg in st.bins.items():
                total_pkts = bagg.f_pkts + bagg.b_pkts
                pkts_per_sec = (total_pkts / self.bin) if self.bin > 0 else 0.0
                f_len_mean = (bagg.f_len_sum / bagg.f_pkts) if bagg.f_pkts > 0 else 0.0
                b_len_mean = (bagg.b_len_sum / bagg.b_pkts) if bagg.b_pkts > 0 else 0.0
                # assessments by direction
                f_iat_mean = (bagg.f_iat_us_sum / bagg.f_iat_cnt) if bagg.f_iat_cnt > 0 else 0.0
                b_iat_mean = (bagg.b_iat_us_sum / bagg.b_iat_cnt) if bagg.b_iat_cnt > 0 else 0.0

                # weighing: if there are both, we weigh by the number of intervals
                w_f = float(bagg.f_iat_cnt)
                w_b = float(bagg.b_iat_cnt)
                if w_f + w_b > 0:
                    flow_iat_mean = (f_iat_mean * w_f + b_iat_mean * w_b) / (w_f + w_b)
                else:
                    # fallback: IAT score based on bin duration (us) and packet count
                    # if there is 1 packet in the bin → (total_pkts-1)=0, we add *1
                    total_pkts = bagg.f_pkts + bagg.b_pkts
                    flow_iat_mean = (self.bin * self.us) / max(total_pkts - 1, 1)

                row = {
                    # meta
                    "src": src, "sport": int(sport), "dst": dst, "dport": int(dport), "proto": int(proto),
                    "time_index": int(bidx),  # integer bin index
                    "ts_edge": float(bagg.ts_edge),  # bin start (sec)
                    "flow_duration": float(self.bin * self.us),  # bin duration (microseconds)

                    # forward
                    "tot_fwd_pkts": int(bagg.f_pkts),
                    "totlen_fwd_pkts": int(bagg.f_bytes),
                    "fwd_pkt_len_max": int(bagg.f_len_max),
                    "fwd_pkt_len_mean": float(f_len_mean),

                    # backward (if not, there will be zeros)
                    "tot_bwd_pkts": int(bagg.b_pkts),
                    "totlen_bwd_pkts": int(bagg.b_bytes),
                    "bwd_pkt_len_max": int(bagg.b_len_max),
                    "bwd_pkt_len_mean": float(b_len_mean),

                    # units
                    "flow_pkts_per_sec": float(pkts_per_sec),
                    "flow_iat_mean": float(flow_iat_mean),  # microseconds (weighted + fallback)
                    "bwd_iat_mean": float(b_iat_mean),  # microseconds (leaving for compatibility)

                    # "just in case" for rules/analytics
                    "flow_pkts_total": int(total_pkts),
                    "flow_bytes_total": int(bagg.f_bytes + bagg.b_bytes),

                    # TLS context
                    "is_tls": int(st.is_tls),
                    "tls_sni": st.tls_sni or "",
                    "tls_alpn": st.tls_alpn or "",
                    "tls_ja3": st.tls_ja3 or "",
                }
                rows.append(row)

        # After flush, we completely clear the state.
        self.flows.clear()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Stable row ordering for pipeline determinism

        # === Self-check of units ===
        try:
            df_tmp = df[(df["flow_pkts_per_sec"] > 0) & (df["flow_iat_mean"] > 0)]
            if len(df_tmp):
                cons = (df_tmp["flow_pkts_per_sec"] * df_tmp["flow_iat_mean"]).median()
                if 1e3 <= cons <= 1e7:
                    print(f"[OK] IAT units consistent (consistency median = {cons:.2f})")
                else:
                    print(f"[WARN] IAT units look off (consistency median = {cons:.2f})")
            else:
                print("[WARN] IAT units check skipped (no valid rows)")
        except Exception as _:
            pass

        df.sort_values(["src", "sport", "dst", "dport", "proto", "time_index"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        # (optional) if you need strictly float32, you can uncomment:
        # float_cols = ["flow_pkts_per_sec", "flow_iat_mean", "bwd_iat_mean", "flow_duration",
        #               "fwd_pkt_len_mean", "bwd_pkt_len_mean", "ts_edge"]
        # df[float_cols] = df[float_cols].astype("float32")

        return df

    def flush_older_than(self, now_ts: float, max_age_sec: float) -> pd.DataFrame:
        """
        Partial "drain" of bins older than a specified time threshold.
        Use to save memory during long runs:
          - Called from the runner periodically (for example, once every 1–5 seconds).
          - Only bins with time_index <= cutoff_idx will be included in the DataFrame.

        Parameters:
          now_ts — current Unix timestamp (sec)
          max_age_sec — keep in memory no older than N seconds

        Returns:
          DataFrame with the "old" rows, the rest is left in memory.
        """
        cutoff_idx = int(math.floor((now_ts - max_age_sec) / self.bin))
        if not self.flows or max_age_sec <= 0:
            return pd.DataFrame()

        rows: List[dict] = []

        # Bypassing all flows and selecting old bins
        for key, st in list(self.flows.items()):
            src, sport, dst, dport, proto = key
            # find the indexes of the bins that need to be merged
            old_bidx = [bidx for bidx in st.bins.keys() if bidx <= cutoff_idx]
            if not old_bidx:
                continue
            old_bidx.sort()
            for bidx in old_bidx:
                bagg = st.bins.pop(bidx)
                total_pkts = bagg.f_pkts + bagg.b_pkts
                pkts_per_sec = (total_pkts / self.bin) if self.bin > 0 else 0.0
                f_len_mean = (bagg.f_len_sum / bagg.f_pkts) if bagg.f_pkts > 0 else 0.0
                b_len_mean = (bagg.b_len_sum / bagg.b_pkts) if bagg.b_pkts > 0 else 0.0
                # assessments by direction
                f_iat_mean = (bagg.f_iat_us_sum / bagg.f_iat_cnt) if bagg.f_iat_cnt > 0 else 0.0
                b_iat_mean = (bagg.b_iat_us_sum / bagg.b_iat_cnt) if bagg.b_iat_cnt > 0 else 0.0

                # weighing: if there are both, we weigh by the number of intervals
                w_f = float(bagg.f_iat_cnt)
                w_b = float(bagg.b_iat_cnt)
                if w_f + w_b > 0:
                    flow_iat_mean = (f_iat_mean * w_f + b_iat_mean * w_b) / (w_f + w_b)
                else:
                    # fallback: IAT score based on bin duration (us) and packet count
                    # if there is 1 packet in the bin → (total_pkts-1)=0, we add *1
                    total_pkts = bagg.f_pkts + bagg.b_pkts
                    flow_iat_mean = (self.bin * self.us) / max(total_pkts - 1, 1)

                row = {
                    # meta
                    "src": src, "sport": int(sport), "dst": dst, "dport": int(dport), "proto": int(proto),
                    "time_index": int(bidx),
                    "ts_edge": float(bagg.ts_edge),
                    "flow_duration": float(self.bin * self.us),

                    # forward
                    "tot_fwd_pkts": int(bagg.f_pkts),
                    "totlen_fwd_pkts": int(bagg.f_bytes),
                    "fwd_pkt_len_max": int(bagg.f_len_max),
                    "fwd_pkt_len_mean": float(f_len_mean),

                    # backward
                    "tot_bwd_pkts": int(bagg.b_pkts),
                    "totlen_bwd_pkts": int(bagg.b_bytes),
                    "bwd_pkt_len_max": int(bagg.b_len_max),
                    "bwd_pkt_len_mean": float(b_len_mean),

                    # units
                    "flow_pkts_per_sec": float(pkts_per_sec),
                    "flow_iat_mean": float(flow_iat_mean),  # microseconds (weighted + fallback)
                    "bwd_iat_mean": float(b_iat_mean),  # microseconds (left for compatibility)

                    # additional
                    "flow_pkts_total": int(total_pkts),
                    "flow_bytes_total": int(bagg.f_bytes + bagg.b_bytes),

                    # TLS
                    "is_tls": int(st.is_tls),
                    "tls_sni": st.tls_sni or "",
                    "tls_alpn": st.tls_alpn or "",
                    "tls_ja3": st.tls_ja3 or "",
                }
                rows.append(row)

            # If the flow has no beans left, you can throw away the entire FlowState
            if not st.bins:
                self.flows.pop(key, None)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # === Quick self-check of units (does not interfere with work) ===
        try:
            df_tmp = df[(df["flow_pkts_per_sec"] > 0) & (df["flow_iat_mean"] > 0)]
            if len(df_tmp):
                cons = (df_tmp["flow_pkts_per_sec"] * df_tmp["flow_iat_mean"]).median()
                if 1e3 <= cons <= 1e7:
                    print(f"[OK] IAT units consistent (consistency median = {cons:.2f})")
                else:
                    print(f"[WARN] IAT units look off (consistency median = {cons:.2f})")
            else:
                print("[WARN] IAT units check skipped (no valid rows)")
        except Exception as _:
            pass

        df.sort_values(["src", "sport", "dst", "dport", "proto", "time_index"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        return df
