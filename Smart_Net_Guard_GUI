#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNetGuard — Live Run GUI (Qt + PyQtGraph)

This module provides an interactive real-time GUI for SmartNetGuard inference:

- Launches SmartNetGuard runner (`Smart_Net_Guard_Run_2.py`) either on:
  * offline PCAP files
  * live network interface (sniffer mode)

- Visualizes reconstruction error (MSE) as a live bar chart using PyQtGraph:
  * Exponential Moving Average (EMA) smoothing
  * Dynamic normalization based on recent percentile
  * Color-coded bars (green/yellow/red) for severity levels

- Handles classification events on a per-window basis:
  * Maintains per-class confidence streaks and refractory periods
  * Converts “streaks” of confident windows into higher-level events
  * Creates per-event artifacts (PNG snapshot, TXT, JSON) for analysis / retraining
  * Generates Zero-Day / unknown anomaly flags based on predicted labels
  * Shows visual popups and optionally sends e-mail notifications

The GUI is designed as a thin orchestration layer around the SmartNetGuard
runner, focusing on observability, alerting and demo-friendly visualizations.
"""

import os
import sys
import subprocess
import threading
import queue
import datetime
import shlex
import csv
import smtplib
import ssl
import re
import json
import ctypes
from email.mime.text import MIMEText

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Enable antialiasing for smoother bar edges / lines
pg.setConfigOptions(antialias=True)


# ===================== CONFIGURE THESE PATHS =====================
# Absolute path to the Python interpreter inside the project virtual environment.
PYTHON = r"C:\path\to\your\venv\Scripts\python.exe"  # ← CHANGE ME locally

# Absolute path to the SmartNetGuard main runner script.
RUNNER = r"C:\path\to\Smart_Net_Guard_Run_2.py"      # ← CHANGE ME locally

# Paths to trained L1 (AutoEncoder) and L2 (classifier) runs.
L1 = r"C:\path\to\L1_ready_for_working\run_xxx"   # ← CHANGE ME locally
L2 = r"C:\path\to\L2_ready_for_working\run_yyy"   # ← CHANGE ME locally

# Path to L1 clipping configuration (used by the runner to clip input features).
CLIP = r"C:\path\to\experiments\calib\l1_clip.json" # ← CHANGE ME locally

# Default output base directory for demonstration runs.
DEFAULT_OUT = r"C:\path\to\presentation\Demo_Run.pcap"    # ← CHANGE ME locally

# Default list of PCAP files to appear in the GUI at startup.
DEFAULT_PCAPS = [
    r"C:\path\to\presentation\Demo_Run.pcap"  # ← CHANGE ME locally,
]

# Path to application icon (.ico) to be used in the Windows taskbar and window.
ICON_PATH = r"C:\path\to\icons\app.ico"


# ===================== RESPONSE HINTS (for incident response) =====================

# Short human-readable response recommendations for each canonical class.
# These hints are written into TXT/JSON artifacts and used in popups/e-mails.
RESPONSE_HINTS = {
    "volumetric_flood": [
        "Temporarily limit the speed or blackhole malicious prefixes on the edge router.",
        "Check your provider's schedules and enable upstream DDoS protection if necessary."
    ],
    "http_flood": [
        "Enable rate-limit by IP/session on HTTP endpoints (WAF / reverse-proxy).",
        "Add a CAPTCHA/JS challenge for suspicious User-Agent and IP."
    ],
    "bot": [
        "Block suspicious IPs/ASNs and tighten authentication for external services.",
        "Check for leaks/compromises of passwords and API keys, especially in unusual geos."
    ],
    "portscan": [
        "Block the scanning source IP on your firewall during the investigation.",
        "Check the list of open ports and close or limit any unnecessary services."
    ],
    "unknown_anomaly": [
        "Treat as a potential Zero-Day: isolate the host/segment if possible.",
        "Save PCAP and logs for the appropriate interval for further forensic and training purposes."
    ],
    "default": [
        "Check if the traffic is legitimate based on logs/PCAP and context.",
        "If the traffic is malicious, block the sources and save the example for training."
    ],
}


def build_response_hint(canonical_cls: str) -> str:
    """
    Build a multi-line textual block with response recommendations for a class.

    Parameters
    ----------
    canonical_cls : str
        Canonical class name (e.g. 'http_flood', 'bot', 'unknown_anomaly').

    Returns
    -------
    str
        Multi-line string with bullet-point recommendations.
    """
    hints = RESPONSE_HINTS.get(canonical_cls)
    if hints is None:
        hints = RESPONSE_HINTS["default"]
    lines = ["Recommended response:"]
    for h in hints:
        lines.append(f"  • {h}")
    return "\n".join(lines)


def normalize_class_name(pred: str) -> tuple[str, bool]:
    """
    Normalize a predicted class name and detect Zero-Day / unknown anomalies.

    Any label that looks like "unknown", "zeroday", "NoAnimals" etc. is mapped
    into a single canonical 'unknown_anomaly' bucket and treated as Zero-Day.

    Parameters
    ----------
    pred : str
        Raw class name as reported by the model or policy pipeline.

    Returns
    -------
    tuple[str, bool]
        (canonical_name, is_zero_day)

        canonical_name : str
            Either one of known classes (e.g. 'http_flood') or 'unknown_anomaly'.
        is_zero_day : bool
            True if this prediction should be treated as Zero-Day / unknown.
    """
    if not pred:
        return "unknown_anomaly", True

    p = pred.lower()
    # Anything that looks like unknown / zeroday / noanimals is treated as unknown anomaly.
    if p.startswith("unknown") or "zeroday" in p or "noanimals" in p:
        return "unknown_anomaly", True

    # For known classes we just pass through the original name.
    return pred, False


# ===================== HELPERS =====================

def bias_content() -> str:
    """
    Return static calibration bias JSON content used for this demo.

    Notes
    -----
    In production you would typically:
    - either learn these biases automatically,
    - or load them from a configuration file.
    Here we keep the content hard-coded for reproducible demo runs.
    """
    return """{
  "logit_volumetric_flood": -3.6,
  "logit_http_flood": 7.8,
  "logit_bot": -2.7,
  "logit_portscan": -2.3
}""".strip()


def ensure_dirs_and_bias(base_out: str) -> str:
    """
    Ensure that the output directory exists and create a bias JSON file in it.

    Parameters
    ----------
    base_out : str
        Base output directory for the current run. Will be created if missing.

    Returns
    -------
    str
        Full path to the created (or overwritten) calibration bias JSON file.
    """
    os.makedirs(base_out, exist_ok=True)
    bias_json = os.path.join(base_out, "calib_bias_windows_run.json")
    with open(bias_json, "w", encoding="utf-8") as f:
        f.write(bias_content())
    return bias_json


def build_args_for_pcap(
    pcap_path: str,
    out_dir: str,
    bias_json: str,
    pre_l2_on: bool,
    stitch_on: bool,
    stitch_min_len: int,
    stitch_max_gap: int,
    policy_preset: str,
    http_whitelist: str | None,
    policy_debug: bool,
    stream_json: bool
) -> list[str]:
    """
    Build command-line argument list for SmartNetGuard runner on a single PCAP.

    Parameters
    ----------
    pcap_path : str
        Path to the PCAP file that will be processed.
    out_dir : str
        Directory where all runner outputs (CSV, logs, coords, events, ...) are stored.
    bias_json : str
        Path to JSON file with calibration biases for L2 logits.
    pre_l2_on : bool
        Whether to enable pre-L2 filter (reconstruction-based filter before classification).
    stitch_on : bool
        Whether to enable event stitching in the runner.
    stitch_min_len : int
        Minimum length of stitched event (in windows).
    stitch_max_gap : int
        Maximum allowed gap between windows inside the same stitched event.
    policy_preset : str
        Name of policy preset to use (e.g. "default", "strict").
    http_whitelist : str | None
        Optional path to HTTP whitelist file (CIDR/IP per line) to be used in policy.
    policy_debug : bool
        If True, runner will emit verbose policy debug logs.
    stream_json : bool
        If True, runner will stream JSON window events to stdout.

    Returns
    -------
    list[str]
        List of CLI arguments to be passed to the Python runner.
    """
    args = [
        "--source", "pcap",
        "--pcap", pcap_path,
        "--pcap_flush_pkts", "800000",
        "--pcap_flush_sec", "15",
        "--bin", "0.05",
        "--iat_unit", "us",
        "--all_windows",

        "--l1_run", L1,
        "--l2_run", L2,

        "--pre_l2_filter", ("on" if pre_l2_on else "off"),
        "--recon_budget", "0.30",

        "--zd_gate", "on",
        "--zd_pmax", "0.60",
        "--zd_entropy", "1.20",
        "--zd_margin", "0.09",

        "--ood", "on",
        "--ood_thresh", "1.60",
        "--ood_alpha", "0.35",
        "--ood_beta", "0.25",
        "--ood_gamma", "0.0",

        "--dump", "csv",
        "--dump_logits", "on",
        "--dump_logits_prefix", "logit_",
        "--l1_clip_json", CLIP,
        "--out", out_dir,

        "--confident_dump", "on",
        "--confident_pmax", "0.60",
        "--confident_subdir", "coords",

        "--calib_T", "0.9",
        f"--calib_bias_json={bias_json}",
        "--l1_batch", "128",
        "--l2_batch", "128",
        "--win_batch", "4096",

        "--policy_preset", policy_preset,
    ]

    if stitch_on:
        args += ["--stitch_min_len", str(stitch_min_len), "--stitch_max_gap", str(stitch_max_gap)]
    else:
        # This combination effectively disables stitching at runner level.
        args += ["--stitch_min_len", "999999", "--stitch_max_gap", "0"]

    if http_whitelist and os.path.exists(http_whitelist):
        args += ["--http_whitelist_file", http_whitelist]
    if policy_debug:
        args += ["--policy_debug"]
    if stream_json:
        args += ["--stream_json"]
    return args


def build_args_for_live(
    iface: str,
    out_dir: str,
    bias_json: str,
    stitch_on: bool,
    stitch_min_len: int,
    stitch_max_gap: int,
    policy_preset: str,
    http_whitelist: str | None,
    policy_debug: bool,
    stream_json: bool
) -> list[str]:
    """
    Build command-line argument list for SmartNetGuard runner on LIVE traffic.

    This assumes that the runner supports:
    - '--source sniffer'
    - '--iface <name>'

    Parameters
    ----------
    iface : str
        Name of network interface to sniff (e.g. "Ethernet", "eth0").
    out_dir : str
        Output directory for all artifacts of the live run.
    bias_json : str
        Path to JSON file with calibration biases for L2 logits.
    stitch_on : bool
        Whether to enable event stitching at runner level.
    stitch_min_len : int
        Minimum length of stitched event (in windows).
    stitch_max_gap : int
        Maximum allowed gap between windows inside the same stitched event.
    policy_preset : str
        Name of policy preset to use.
    http_whitelist : str | None
        Optional HTTP whitelist file path, if any.
    policy_debug : bool
        Verbose policy debug mode.
    stream_json : bool
        Whether to stream JSON window records to stdout.

    Returns
    -------
    list[str]
        List of CLI arguments to pass to the runner.
    """
    args = [
        "--source", "sniffer",
        "--iface", iface,
        "--bin", "0.05",
        "--iat_unit", "us",
        "--all_windows",

        "--l1_run", L1,
        "--l2_run", L2,

        # In live mode we always enable pre-L2 filter for production-like behavior.
        "--pre_l2_filter", "on",
        "--recon_budget", "0.30",

        "--zd_gate", "on",
        "--zd_pmax", "0.60",
        "--zd_entropy", "1.20",
        "--zd_margin", "0.09",

        "--ood", "on",
        "--ood_thresh", "1.60",
        "--ood_alpha", "0.35",
        "--ood_beta", "0.25",
        "--ood_gamma", "0.0",

        "--dump", "csv",
        "--dump_logits", "on",
        "--dump_logits_prefix", "logit_",
        "--l1_clip_json", CLIP,
        "--out", out_dir,

        "--confident_dump", "on",
        "--confident_pmax", "0.60",
        "--confident_subdir", "coords",

        "--calib_T", "0.9",
        f"--calib_bias_json={bias_json}",
        "--l1_batch", "128",
        "--l2_batch", "128",
        "--win_batch", "4096",

        "--policy_preset", policy_preset,
    ]

    if stitch_on:
        args += ["--stitch_min_len", str(stitch_min_len), "--stitch_max_gap", str(stitch_max_gap)]
    else:
        args += ["--stitch_min_len", "999999", "--stitch_max_gap", "0"]

    if http_whitelist and os.path.exists(http_whitelist):
        args += ["--http_whitelist_file", http_whitelist]
    if policy_debug:
        args += ["--policy_debug"]
    if stream_json:
        args += ["--stream_json"]
    return args


# ===================== ALERT DIALOG =====================

class AlertDialog(QtWidgets.QDialog):
    """
    Simple blinking dialog for event notifications.

    Responsibilities
    ----------------
    - Display a short incident summary to the operator.
    - Blink background color to attract attention.
    - Auto-close after a fixed timeout (6 seconds).
    """
    def __init__(self, parent: QtWidgets.QWidget, text: str):
        """
        Parameters
        ----------
        parent : QtWidgets.QWidget
            Parent widget (usually the main window).
        text : str
            Text to show inside the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle("⚠ SmartNetGuard: Event detected")
        self.setModal(False)

        # Multi-line label with white text.
        self.label = QtWidgets.QLabel(text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("QLabel { color: white; font: 11pt 'Segoe UI'; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        # Internal blink state flag.
        self._state = False

        # Timer for blinking background color.
        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(250)

        self._blink()

        # Auto-close after 6 seconds.
        QtCore.QTimer.singleShot(6000, self.accept)
        self.resize(520, 260)

    def _blink(self) -> None:
        """
        Toggle dialog background color between two red shades.
        Called periodically by QTimer.
        """
        self._state = not self._state
        color = "#8b0000" if self._state else "#b22222"
        self.setStyleSheet(f"QDialog {{ background-color: {color}; }}")


# ===================== MAIN GUI (Qt + PyQtGraph) =====================

class SmartNetGuardWindow(QtWidgets.QMainWindow):
    """
    Main Qt window for SmartNetGuard live/demo runs.

    High-level responsibilities
    ---------------------------
    - Manage SmartNetGuard runner subprocess (PCAP or LIVE).
    - Parse runner stdout/stderr (JSON or text) to extract:
        * per-window class predictions
        * maximum class probability (pmax)
        * reconstruction MSE
    - Maintain a real-time reconstruction error visualization (bar chart).
    - Convert window-level predictions into human-readable events:
        * per-class streak detection
        * refractory logic
        * per-event artifact creation (PNG/TXT/JSON)
        * alerts (popups + optional e-mail)
    - Provide GUI controls for:
        * PCAP list, live interface, output directory
        * policy/stitching parameters
        * alert thresholds and EMA coefficient
        * e-mail credentials and SMTP settings
    """

    # Signals emitted from worker thread → main thread (Qt event loop)
    new_mse = QtCore.pyqtSignal(float)
    new_window_event = QtCore.pyqtSignal(str, object, object, str, str)

    # Pre-compiled regexes for textual log parsing (fallback mode).
    _re_pred = re.compile(r"pred=([a-zA-Z0-9_]+)")
    _re_pmax = re.compile(r"pmax=([0-9]*\.?[0-9]+)")
    _re_rmse = re.compile(r"recon_mse=([0-9]*\.?[0-9]+)")

    def __init__(self):
        """
        Initialize the main window, all GUI widgets and internal state.

        Notes
        -----
        The window is not shown by the constructor. It is shown in the
        `if __name__ == "__main__"` block, after QApplication is created.
        """
        super().__init__()
        self.setWindowTitle("SmartNetGuard — Live Run (Qt)")
        self.resize(1300, 900)

        # ---------------------- Process & threading state ----------------------
        #: `subprocess.Popen` instance for the current runner, or None if idle.
        self.proc: subprocess.Popen | None = None

        #: Flag indicating that we requested runner termination.
        self.stop_flag = threading.Event()

        #: Queue of log lines arriving from the worker thread.
        self.q: "queue.Queue[str]" = queue.Queue()

        # ---------------------- Visualization state ---------------------------
        #: Raw EMA-smoothed MSE values for the tail windows.
        self._ys_raw: list[float] = []

        #: Normalized MSE values (0..1) aligned with `_ys_raw`.
        self._ys_norm: list[float] = []

        #: Exponential Moving Average current value.
        self.ema_val: float | None = None

        #: Smoothing factor for EMA (0..1).
        self.ema_alpha: float = 0.20

        #: Threshold for "warning" level in normalized MSE.
        self.warn_thr: float = 0.45

        #: Threshold for "alarm" level in normalized MSE.
        self.alarm_thr: float = 0.60

        #: Maximum number of windows to keep in GUI history.
        self.max_points: int = 600

        #: Percentile used for dynamic normalization of EMA stream.
        self.norm_pctl: int = 95

        #: Window size (in points) for percentile-based normalization.
        self.norm_win: int = 600

        #: Last normalized MSE value (backed by `_ys_norm`).
        self._last_norm_y: float | None = None

        #: Path to the on-disk CSV file with per-window MSE values (if used).
        self._mse_file: str | None = None

        #: Byte offset inside `_mse_file` for incremental tail reading.
        self._mse_pos: int = 0

        # ---------------------- Alerting logic --------------------------------
        #: Threshold on pmax for a window to be considered "confident".
        self.confident_thr: float = 0.75

        #: Refractory period (seconds) between two events of the same class.
        self.refractory_sec: int = 15

        #: Maximum number of e-mail alerts per class (to avoid spamming).
        self.max_alerts_per_class: int = 3

        #: Per-class state dictionary:
        #:   key   = raw class string (e.g. "http_flood"),
        #:   value = {"streak": int, "last_ts": datetime, "email_sent": int}.
        self._class_state: dict[str, dict] = {}

        #: Global monotonically increasing index of the last window.
        self._win_counter: int = -1

        #: Boolean flag indicating whether the UI is paused (no visualization updates).
        self.paused: bool = False

        # ---------------------- Animation state -------------------------------
        #: Target normalized MSE values used in animation step.
        self._ys_norm_target: list[float] = []

        #: Currently displayed normalized MSE values (animated towards target).
        self._ys_norm_disp: list[float] = []

        #: Animation coefficient (0..1) controlling smoothness of bar updates.
        self.anim_alpha: float = 0.35

        # ---------------------- PyQtGraph objects -----------------------------
        #: Plot widget for MSE visualization.
        self.mse_plot: pg.PlotWidget | None = None

        #: Bar layers for low/mid/high segments of MSE values.
        self.bars_low: pg.BarGraphItem | None = None
        self.bars_mid: pg.BarGraphItem | None = None
        self.bars_high: pg.BarGraphItem | None = None

        #: Flag set to True when bar data has changed and a redraw is needed.
        self._mse_dirty: bool = False

        # Build and connect all UI widgets.
        self._build_ui()

        # Connect signals to slots.
        self.new_mse.connect(self._push_mse)
        self.new_window_event.connect(self._handle_window_event)

        # Timer driving UI updates (log draining + animation).
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.ui_tick)
        self.ui_timer.start(40)  # ~25 FPS

    # -------------------------------------------------------------------------
    # UI layout and helpers
    # -------------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Construct the full Qt widget hierarchy for the main window.

        Layout overview
        ----------------
        - Top horizontal split:
            * Left: Input/Output group (OUT dir, PCAP list, Live iface)
            * Right: flags/policy context/e-mail configuration
        - Middle: control buttons (Start PCAP, Start LIVE, Pause, Stop)
        - Lower middle: log panel (PlainTextEdit)
        - Bottom: PyQtGraph bar chart for normalized reconstruction error (MSE)
        """
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_v = QtWidgets.QVBoxLayout(central)
        main_v.setContentsMargins(6, 6, 6, 6)
        main_v.setSpacing(6)

        top_h = QtWidgets.QHBoxLayout()
        main_v.addLayout(top_h)

        # --- Input / Output (left) ---
        io_group = QtWidgets.QGroupBox("Input / Output", self)
        io_layout = QtWidgets.QGridLayout(io_group)

        lbl_out = QtWidgets.QLabel("Output base:")
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setText(DEFAULT_OUT)
        btn_out = QtWidgets.QPushButton("📁")
        btn_out.clicked.connect(self.pick_outdir)

        io_layout.addWidget(lbl_out, 0, 0)
        io_layout.addWidget(self.out_edit, 0, 1)
        io_layout.addWidget(btn_out, 0, 2)

        lbl_pcap = QtWidgets.QLabel("PCAP list:")
        self.pcap_list = QtWidgets.QListWidget()
        self.pcap_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.pcap_list.setFixedHeight(90)
        for p in DEFAULT_PCAPS:
            if os.path.exists(p):
                self.pcap_list.addItem(p)

        btn_add_pcap = QtWidgets.QPushButton("Add PCAP…")
        btn_add_pcap.clicked.connect(self.add_pcap)

        # Button to remove selected PCAP entries.
        btn_del_pcap = QtWidgets.QPushButton("Remove selected")
        btn_del_pcap.clicked.connect(self.remove_selected_pcaps)

        io_layout.addWidget(lbl_pcap, 1, 0, QtCore.Qt.AlignTop)
        io_layout.addWidget(self.pcap_list, 1, 1, 2, 1)
        io_layout.addWidget(btn_add_pcap, 1, 2, QtCore.Qt.AlignTop)
        io_layout.addWidget(btn_del_pcap, 2, 2, QtCore.Qt.AlignTop)

        # Live interface name input.
        lbl_iface = QtWidgets.QLabel("Live iface:")
        self.live_iface = QtWidgets.QLineEdit()
        self.live_iface.setPlaceholderText("e.g. Ethernet, eth0 ...")
        io_layout.addWidget(lbl_iface, 3, 0)
        io_layout.addWidget(self.live_iface, 3, 1)

        io_layout.setColumnStretch(1, 1)
        io_group.setMinimumWidth(520)
        top_h.addWidget(io_group, stretch=3)

        # --- Right vertical block ---
        right_v = QtWidgets.QVBoxLayout()
        right_v.setSpacing(4)
        top_h.addLayout(right_v, stretch=4)

        # Processing flags.
        flags = QtWidgets.QGroupBox("Processing flags", self)
        fl = QtWidgets.QGridLayout(flags)

        self.chk_pre_l2 = QtWidgets.QCheckBox("Pre-L2 filter (prod)")
        self.chk_pre_l2.setChecked(True)
        self.chk_stitch = QtWidgets.QCheckBox("Event stitching")
        self.chk_stitch.setChecked(True)
        self.chk_policy_debug = QtWidgets.QCheckBox("Policy debug")
        self.chk_stream_json = QtWidgets.QCheckBox("Stream JSON")
        self.chk_stream_json.setChecked(True)

        fl.addWidget(self.chk_pre_l2, 0, 0)
        fl.addWidget(self.chk_stitch, 0, 1)
        fl.addWidget(self.chk_policy_debug, 0, 2)
        fl.addWidget(self.chk_stream_json, 0, 3)

        fl.addWidget(QtWidgets.QLabel("stitch_min_len"), 1, 0, QtCore.Qt.AlignRight)
        self.ent_st_min = QtWidgets.QLineEdit("8")
        self.ent_st_min.setFixedWidth(60)
        fl.addWidget(self.ent_st_min, 1, 1)

        fl.addWidget(QtWidgets.QLabel("stitch_max_gap"), 1, 2, QtCore.Qt.AlignRight)
        self.ent_st_gap = QtWidgets.QLineEdit("2")
        self.ent_st_gap.setFixedWidth(60)
        fl.addWidget(self.ent_st_gap, 1, 3)

        fl.addWidget(QtWidgets.QLabel("confident_pmax"), 2, 0, QtCore.Qt.AlignRight)
        self.ent_conf_pmax = QtWidgets.QLineEdit("0.75")
        self.ent_conf_pmax.setFixedWidth(60)
        fl.addWidget(self.ent_conf_pmax, 2, 1)

        fl.addWidget(QtWidgets.QLabel("refractory_sec"), 2, 2, QtCore.Qt.AlignRight)
        self.ent_refr = QtWidgets.QLineEdit("15")
        self.ent_refr.setFixedWidth(60)
        fl.addWidget(self.ent_refr, 2, 3)

        fl.addWidget(QtWidgets.QLabel("max_alerts_per_class"), 3, 0, QtCore.Qt.AlignRight)
        self.ent_max_alerts = QtWidgets.QLineEdit("3")
        self.ent_max_alerts.setFixedWidth(60)
        fl.addWidget(self.ent_max_alerts, 3, 1)

        flags.setMaximumHeight(120)
        right_v.addWidget(flags)

        # Policy / Context + thresholds.
        policy = QtWidgets.QGroupBox("Policy / Context", self)
        pl = QtWidgets.QGridLayout(policy)

        pl.addWidget(QtWidgets.QLabel("Policy preset:"), 0, 0, QtCore.Qt.AlignRight)
        self.cmb_policy = QtWidgets.QComboBox()
        self.cmb_policy.addItems(["default", "strict"])
        self.cmb_policy.setCurrentText("default")
        self.cmb_policy.setFixedWidth(120)
        pl.addWidget(self.cmb_policy, 0, 1)

        pl.addWidget(QtWidgets.QLabel("HTTP whitelist file:"), 0, 2, QtCore.Qt.AlignRight)
        self.ent_whitelist = QtWidgets.QLineEdit()
        pl.addWidget(self.ent_whitelist, 0, 3)
        btn_wh = QtWidgets.QPushButton("Browse…")
        btn_wh.clicked.connect(self.pick_whitelist)
        pl.addWidget(btn_wh, 0, 4)

        pl.setColumnStretch(3, 1)

        pl.addWidget(QtWidgets.QLabel("Warn (yellow):"), 1, 0, QtCore.Qt.AlignRight)
        self.ent_warn = QtWidgets.QLineEdit("0.45")
        self.ent_warn.setFixedWidth(70)
        pl.addWidget(self.ent_warn, 1, 1)

        pl.addWidget(QtWidgets.QLabel("Alarm (red):"), 1, 2, QtCore.Qt.AlignRight)
        self.ent_alarm = QtWidgets.QLineEdit("0.60")
        self.ent_alarm.setFixedWidth(70)
        pl.addWidget(self.ent_alarm, 1, 3)

        pl.addWidget(QtWidgets.QLabel("EMA α:"), 1, 4, QtCore.Qt.AlignRight)
        self.ent_alpha = QtWidgets.QLineEdit("0.20")
        self.ent_alpha.setFixedWidth(60)
        pl.addWidget(self.ent_alpha, 1, 5)

        policy.setMaximumHeight(90)
        right_v.addWidget(policy)

        # E-mail Alerts configuration.
        mail = QtWidgets.QGroupBox("E-mail Alerts", self)
        ml = QtWidgets.QGridLayout(mail)

        ml.addWidget(QtWidgets.QLabel("To:"), 0, 0, QtCore.Qt.AlignRight)
        self.email_to = QtWidgets.QLineEdit()
        self.email_to.setFixedWidth(220)
        ml.addWidget(self.email_to, 0, 1)

        ml.addWidget(QtWidgets.QLabel("From:"), 0, 2, QtCore.Qt.AlignRight)
        self.email_from = QtWidgets.QLineEdit()
        self.email_from.setFixedWidth(220)
        ml.addWidget(self.email_from, 0, 3)

        ml.addWidget(QtWidgets.QLabel("App password:"), 0, 4, QtCore.Qt.AlignRight)
        self.email_pass = QtWidgets.QLineEdit()
        self.email_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.email_pass.setFixedWidth(200)
        ml.addWidget(self.email_pass, 0, 5)

        ml.addWidget(QtWidgets.QLabel("SMTP host:"), 1, 0, QtCore.Qt.AlignRight)
        self.email_host = QtWidgets.QLineEdit("smtp.gmail.com")
        self.email_host.setFixedWidth(200)
        ml.addWidget(self.email_host, 1, 1)

        ml.addWidget(QtWidgets.QLabel("Port:"), 1, 2, QtCore.Qt.AlignRight)
        self.email_port = QtWidgets.QLineEdit("587")
        self.email_port.setFixedWidth(80)
        ml.addWidget(self.email_port, 1, 3)

        mail.setMaximumHeight(90)
        right_v.addWidget(mail)

        # --- Buttons + log ---
        btn_row = QtWidgets.QHBoxLayout()
        main_v.addLayout(btn_row)

        self.btn_start = QtWidgets.QPushButton("▶ Start (PCAP)")
        self.btn_start_live = QtWidgets.QPushButton("▶ Start (LIVE)")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_open = QtWidgets.QPushButton("Open OUT folder")

        # Slightly larger and bold font for primary control buttons.
        btn_font = self.btn_start.font()
        btn_font.setPointSize(btn_font.pointSize() + 1)
        btn_font.setBold(True)
        for b in (self.btn_start, self.btn_start_live, self.btn_pause, self.btn_stop):
            b.setFont(btn_font)
            b.setMinimumHeight(32)
            b.setMinimumWidth(110)

        # Connect button signals.
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start_live.clicked.connect(self.on_start_live)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_open.clicked.connect(self.open_out)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_start_live)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_open)

        # Logs.
        log_group = QtWidgets.QGroupBox("Log", self)
        log_layout = QtWidgets.QVBoxLayout(log_group)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)

        note = QtWidgets.QLabel(
            "Note: CSV and logs appear on the fly in "
            "OUT/run_<timestamp>/<pcap_basename> or OUT/run_<timestamp>_live."
        )
        note.setStyleSheet("color: gray;")
        log_layout.addWidget(note)

        main_v.addWidget(log_group, stretch=2)

        # --- MSE plot ---
        mse_group = QtWidgets.QGroupBox("Reconstruction Error (MSE)", self)
        mse_layout = QtWidgets.QVBoxLayout(mse_group)

        self.mse_plot = pg.PlotWidget()
        self.mse_plot.setLabel("bottom", "Windows")
        self.mse_plot.setLabel("left", "MSE (normalized)")
        self.mse_plot.setYRange(0, 1.0)
        self.mse_plot.showGrid(x=True, y=True, alpha=0.25)

        # BarGraphItems for different severity regions.
        self.bars_low = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush("#00aa00"))
        self.bars_mid = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush("#ffd700"))
        self.bars_high = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush("#ff0000"))
        self.mse_plot.addItem(self.bars_low)
        self.mse_plot.addItem(self.bars_mid)
        self.mse_plot.addItem(self.bars_high)

        mse_layout.addWidget(self.mse_plot)
        main_v.addWidget(mse_group, stretch=5)

    # ----------------- UI helpers -----------------

    def log_print(self, s: str) -> None:
        """
        Append a timestamped log line to the GUI log panel.

        Parameters
        ----------
        s : str
            Line of text to append. Timestamp will be added automatically.
        """
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {s}")
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def pick_outdir(self) -> None:
        """
        Open a directory selection dialog and update the output base path.
        """
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.out_edit.text() or os.getcwd()
        )
        if d:
            self.out_edit.setText(d)

    def pick_whitelist(self) -> None:
        """
        Open a file selection dialog for HTTP whitelist and update the path field.
        """
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose whitelist (CIDR/IP per line)",
            "",
            "Text files (*.txt *.list *.cfg *.conf);;All files (*.*)",
        )
        if f:
            self.ent_whitelist.setText(f)

    def add_pcap(self) -> None:
        """
        Open file selection dialog and append chosen PCAPs to the list widget.
        """
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Choose PCAP files",
            "",
            "pcap/pcapng (*.pcap *.pcapng);;All files (*.*)",
        )
        for f in files:
            self.pcap_list.addItem(f)

    def remove_selected_pcaps(self) -> None:
        """
        Remove selected PCAP files from the list widget.
        """
        for item in self.pcap_list.selectedItems():
            row = self.pcap_list.row(item)
            self.pcap_list.takeItem(row)

    def open_out(self) -> None:
        """
        Open the configured output base directory using OS file manager.
        """
        d = self.out_edit.text().strip() or DEFAULT_OUT
        if os.path.isdir(d):
            try:
                os.startfile(d)
            except Exception:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))
        else:
            QtWidgets.QMessageBox.warning(self, "No folder", d)

    # -------------------------------------------------------------------------
    # start/stop/pause helpers
    # -------------------------------------------------------------------------

    def _read_thresholds_and_flags(self) -> tuple[float, float, float, int, int, float, int, int]:
        """
        Read threshold/flag values from UI fields.

        Returns
        -------
        tuple
            (warn_thr, alarm_thr, ema_alpha,
             stitch_min_len, stitch_max_gap,
             confident_thr, refractory_sec,
             max_alerts_per_class)

        Notes
        -----
        In case of invalid user input, sensible defaults are applied.
        """
        try:
            warn_thr = float(self.ent_warn.text().strip())
            alarm_thr = float(self.ent_alarm.text().strip())
            ema_alpha = max(0.01, min(float(self.ent_alpha.text().strip()), 0.9))
            st_min = max(1, int(self.ent_st_min.text().strip()))
            st_gap = max(0, int(self.ent_st_gap.text().strip()))
            confident_thr = float(self.ent_conf_pmax.text().strip())
            refractory_sec = max(0, int(self.ent_refr.text().strip()))
            max_alerts_per_class = max(1, int(self.ent_max_alerts.text().strip()))
        except Exception:
            warn_thr, alarm_thr, ema_alpha = 0.45, 0.60, 0.20
            st_min, st_gap = 8, 2
            confident_thr = 0.75
            refractory_sec = 15
            max_alerts_per_class = 3
        return (
            warn_thr,
            alarm_thr,
            ema_alpha,
            st_min,
            st_gap,
            confident_thr,
            refractory_sec,
            max_alerts_per_class,
        )

    def _reset_visual_state(self) -> None:
        """
        Reset all visualization and per-class state for a fresh run.

        This is called when:
        - a new PCAP is started, or
        - a new LIVE run is started.
        """
        self._ys_raw, self._ys_norm = [], []
        self._ys_norm_target, self._ys_norm_disp = [], []
        self.ema_val = None
        self._mse_file, self._mse_pos = None, 0
        self._last_norm_y = None
        self._class_state.clear()
        self._mse_dirty = True
        self._win_counter = -1

    def on_start(self) -> None:
        """
        Handler for the "Start (PCAP)" button.

        Responsibilities
        ----------------
        - Validate UI inputs (OUT dir, selected PCAPs).
        - Create run subdirectory under OUT base.
        - Build runner arguments for each selected PCAP.
        - Start a worker thread that:
            * launches runner process,
            * reads stdout line-by-line,
            * parses prediction/MSE values,
            * emits signals into main thread for visualization & alerts.
        """
        if self.proc:
            QtWidgets.QMessageBox.warning(self, "Already launched", "The process is already running.")
            return

        self.paused = False
        self.btn_pause.setText("Pause")

        base_root = self.out_edit.text().strip()
        if not base_root:
            QtWidgets.QMessageBox.critical(self, "Error", "Specify Output base.")
            return

        run_stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        base_out = os.path.join(base_root, run_stamp)
        os.makedirs(base_out, exist_ok=True)

        (
            self.warn_thr,
            self.alarm_thr,
            self.ema_alpha,
            st_min,
            st_gap,
            self.confident_thr,
            self.refractory_sec,
            self.max_alerts_per_class,
        ) = self._read_thresholds_and_flags()

        pre_l2_on = bool(self.chk_pre_l2.isChecked())
        stitch_on = bool(self.chk_stitch.isChecked())
        policy_debug = bool(self.chk_policy_debug.isChecked())
        stream_json = bool(self.chk_stream_json.isChecked())
        policy_preset = self.cmb_policy.currentText().strip()
        whitelist_path = self.ent_whitelist.text().strip() or None

        bias_json = ensure_dirs_and_bias(base_out)

        selected = self.pcap_list.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.critical(
                self,
                "No PCAP selected",
                "Please select at least one PCAP file in the list."
            )
            return

        pcaps = [it.text() for it in selected]
        pcaps = [p for p in pcaps if os.path.exists(p)]
        if not pcaps:
            QtWidgets.QMessageBox.critical(self, "Error", "Selected PCAP paths do not exist.")
            return

        self.stop_flag.clear()
        self.btn_start.setEnabled(False)
        self.btn_start_live.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_visual_state()

        def worker() -> None:
            """
            Worker thread body for offline (PCAP) mode.

            For each PCAP:
            - Build the full command.
            - Launch the runner process.
            - Stream stdout and parse lines into GUI signals.
            """
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["LC_ALL"] = "C.UTF-8"
                env["LANG"] = "C.UTF-8"

                for p in pcaps:
                    leaf = os.path.splitext(os.path.basename(p))[0]
                    out_dir = os.path.join(base_out, leaf)
                    os.makedirs(out_dir, exist_ok=True)

                    args = build_args_for_pcap(
                        p,
                        out_dir,
                        bias_json,
                        pre_l2_on=pre_l2_on,
                        stitch_on=stitch_on,
                        stitch_min_len=st_min,
                        stitch_max_gap=st_gap,
                        policy_preset=policy_preset,
                        http_whitelist=whitelist_path,
                        policy_debug=policy_debug,
                        stream_json=stream_json,
                    )
                    full_cmd = [PYTHON, RUNNER] + args
                    self.q.put(f"RUN: {' '.join(shlex.quote(x) for x in full_cmd)}")

                    self._mse_file = os.path.join(out_dir, "windows_mse.csv")
                    self._mse_pos = 0
                    self.ema_val = None
                    self._ys_raw, self._ys_norm = [], []
                    self._ys_norm_target, self._ys_norm_disp = [], []
                    self._last_norm_y = None
                    self._win_counter = -1

                    self.proc = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        universal_newlines=True,
                        env=env,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )

                    for line in self.proc.stdout:
                        if self.stop_flag.is_set():
                            break
                        line = line.lstrip("\ufeff").rstrip()
                        self.q.put(line)
                        pred, pmax, rmse = self._parse_line(line)
                        if rmse is not None:
                            self.new_mse.emit(rmse)
                        if pred is not None:
                            self.new_window_event.emit(pred, pmax, rmse, line, out_dir)

                    rc = self.proc.wait()
                    self.q.put(f"=== Completed: {leaf} (rc={rc}) ===")
                    if self.stop_flag.is_set():
                        break
            except Exception as e:  # noqa: BLE001
                self.q.put(f"[ERROR] {e}")
            finally:
                self.proc = None
                self.stop_flag.clear()
                self.q.put("__PROC_DONE__")

        threading.Thread(target=worker, daemon=True).start()
        self.log_print("I'm starting (PCAP)...")

    def on_start_live(self) -> None:
        """
        Handler for the "Start (LIVE)" button.

        Responsibilities
        ----------------
        - Validate UI inputs (OUT dir, interface name).
        - Create run subdirectory under OUT base with `_live` suffix.
        - Build runner arguments for live sniffer.
        - Start a worker thread that reads and parses stdout in real time.
        """
        if self.proc:
            QtWidgets.QMessageBox.warning(self, "Already launched", "The process is already running.")
            return

        self.paused = False
        self.btn_pause.setText("Pause")

        base_root = self.out_edit.text().strip()
        if not base_root:
            QtWidgets.QMessageBox.critical(self, "Error", "Specify Output base.")
            return

        iface = self.live_iface.text().strip()
        if not iface:
            QtWidgets.QMessageBox.critical(self, "Error", "Specify Live iface (e.g. Ethernet, eth0).")
            return

        run_stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_live")
        out_dir = os.path.join(base_root, run_stamp)
        os.makedirs(out_dir, exist_ok=True)

        (
            self.warn_thr,
            self.alarm_thr,
            self.ema_alpha,
            st_min,
            st_gap,
            self.confident_thr,
            self.refractory_sec,
            self.max_alerts_per_class,
        ) = self._read_thresholds_and_flags()

        # Pre-L2 is always ON in live mode. Other flags are driven by UI.
        stitch_on = bool(self.chk_stitch.isChecked())
        policy_debug = bool(self.chk_policy_debug.isChecked())
        stream_json = bool(self.chk_stream_json.isChecked())
        policy_preset = self.cmb_policy.currentText().strip()
        whitelist_path = self.ent_whitelist.text().strip() or None

        bias_json = ensure_dirs_and_bias(out_dir)

        self.stop_flag.clear()
        self.btn_start.setEnabled(False)
        self.btn_start_live.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_visual_state()

        def worker_live() -> None:
            """
            Worker thread body for LIVE sniffer mode.

            - Launch runner in sniffer mode on a given interface.
            - Stream stdout and parse JSON/text lines into GUI events.
            """
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["LC_ALL"] = "C.UTF-8"
                env["LANG"] = "C.UTF-8"

                args = build_args_for_live(
                    iface,
                    out_dir,
                    bias_json,
                    stitch_on=stitch_on,
                    stitch_min_len=st_min,
                    stitch_max_gap=st_gap,
                    policy_preset=policy_preset,
                    http_whitelist=whitelist_path,
                    policy_debug=policy_debug,
                    stream_json=stream_json,
                )
                full_cmd = [PYTHON, RUNNER] + args
                self.q.put(f"RUN LIVE: {' '.join(shlex.quote(x) for x in full_cmd)}")

                self._mse_file = os.path.join(out_dir, "windows_mse.csv")
                self._mse_pos = 0
                self.ema_val = None
                self._ys_raw, self._ys_norm = [], []
                self._ys_norm_target, self._ys_norm_disp = [], []
                self._last_norm_y = None
                self._win_counter = -1

                self.proc = subprocess.Popen(
                    full_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in self.proc.stdout:
                    if self.stop_flag.is_set():
                        break
                    line = line.lstrip("\ufeff").rstrip()
                    self.q.put(line)
                    pred, pmax, rmse = self._parse_line(line)
                    if rmse is not None:
                        self.new_mse.emit(rmse)
                    if pred is not None:
                        self.new_window_event.emit(pred, pmax, rmse, line, out_dir)

                rc = self.proc.wait()
                self.q.put(f"=== LIVE capture finished (rc={rc}) ===")
            except Exception as e:  # noqa: BLE001
                self.q.put(f"[ERROR] {e}")
            finally:
                self.proc = None
                self.stop_flag.clear()
                self.q.put("__PROC_DONE__")

        threading.Thread(target=worker_live, daemon=True).start()
        self.log_print(f"I'm starting LIVE on iface={iface}...")

    def on_stop(self) -> None:
        """
        Handler for the "Stop" button.

        Sets the `stop_flag` for worker threads and tries to terminate the
        underlying subprocess.
        """
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.log_print("I'm stopping...")

    def on_pause(self) -> None:
        """
        Handler for the "Pause" button.

        Pauses/resumes visualization (MSE updates). Does not stop the runner.
        """
        if not self.proc:
            return
        self.paused = not self.paused
        self.btn_pause.setText("Resume" if self.paused else "Pause")

    # -------------------------------------------------------------------------
    # Main UI timer tick / log draining / file-based MSE reading
    # -------------------------------------------------------------------------

    def ui_tick(self) -> None:
        """
        Periodic UI tick (driven by `ui_timer`).

        Responsibilities
        ----------------
        - Drain log queue and append allowed lines to the log panel.
        - If JSON streaming is disabled:
            * read tail of CSV file with MSE values from disk.
        - If bar data is marked as dirty:
            * perform one animation step and refresh plot.
        """
        self._drain_log_queue()

        if not self.chk_stream_json.isChecked():
            self._update_mse_plot_from_file_tail()

        if self._mse_dirty:
            self._animate_mse()

    def _drain_log_queue(self) -> None:
        """
        Drain all available log lines from the worker queue and print them.

        Only lines containing certain keywords (ALLOW list) are shown in the log
        to keep the GUI readable.
        """
        ALLOW = ("pred=", "EVENT", "[ERROR]", "[WARN]", "Completed", "RUN", "ZD", "OOD", "LIVE capture")
        try:
            while True:
                s = self.q.get_nowait()
                if s == "__PROC_DONE__":
                    self.btn_start.setEnabled(True)
                    self.btn_start_live.setEnabled(True)
                    self.btn_stop.setEnabled(False)
                else:
                    if any(tok in s for tok in ALLOW):
                        self.log_print(s)
        except queue.Empty:
            pass

    # -------------------------------------------------------------------------
    # Stdout parser (JSON or text)
    # -------------------------------------------------------------------------

    def _parse_line(self, line: str) -> tuple[str | None, float | None, float | None]:
        """
        Parse one stdout line from the runner and extract prediction & MSE info.

        The runner may emit:
        - JSON objects with 'kind' == 'window', or
        - plain text log lines containing 'pred=...', 'pmax=...', 'recon_mse=...'.

        Parameters
        ----------
        line : str
            Single stdout line from the runner.

        Returns
        -------
        tuple
            (pred, pmax, rmse)

            pred : str | None
                Raw class string (if present) or None.
            pmax : float | None
                Maximum predicted probability for that window (if available).
            rmse : float | None
                Reconstruction MSE (if available).
        """
        pred: str | None = None
        pmax: float | None = None
        rmse: float | None = None

        # JSON streaming mode (preferred).
        if line.startswith("{") and ("\"kind\"" in line):
            try:
                obj = json.loads(line)
                if obj.get("kind") == "window":
                    pred = str(obj.get("pred", "unknown"))
                    if obj.get("pmax") is not None:
                        pmax = float(obj.get("pmax"))
                    if obj.get("recon_mse") is not None:
                        rmse = float(obj.get("recon_mse"))
            except Exception:
                pass

        # Text log fallback ("pred=... pmax=... recon_mse=...").
        if pred is None and ("pred=" in line and "pmax=" in line):
            try:
                m_pred = self._re_pred.search(line)
                m_pmax = self._re_pmax.search(line)
                m_rmse = self._re_rmse.search(line)
                pred = m_pred.group(1) if m_pred else "unknown"
                pmax = float(m_pmax.group(1)) if m_pmax else None
                rmse = float(m_rmse.group(1)) if m_rmse else None
            except Exception:
                pass

        return pred, pmax, rmse

    # -------------------------------------------------------------------------
    # Visualization core (MSE pipeline)
    # -------------------------------------------------------------------------

    def _push_mse(self, val: float) -> None:
        """
        Push a new MSE value into the visualization pipeline.

        The pipeline consists of:
        1) Optional pause check.
        2) EMA smoothing.
        3) Dynamic normalization based on recent percentile.
        4) Animated update of bar chart.

        Parameters
        ----------
        val : float
            Raw reconstruction MSE value for the current window.
        """
        if self.paused:
            return

        try:
            v = float(val)
        except Exception:
            return

        self._win_counter += 1

        # Update EMA.
        self.ema_val = v if self.ema_val is None else (
            self.ema_alpha * v + (1 - self.ema_alpha) * self.ema_val
        )

        # Append EMA to raw list and truncate history.
        self._ys_raw.append(self.ema_val)
        self._ys_raw = self._ys_raw[-self.max_points:]

        # Compute normalization reference (percentile) on a tail.
        tail = self._ys_raw[-self.norm_win:] if len(self._ys_raw) >= 5 else self._ys_raw
        ref = np.percentile(tail, self.norm_pctl) if tail else None
        if (ref is not None) and (ref > 0):
            self._ys_norm = [min(y / ref, 1.0) for y in self._ys_raw]
        else:
            mx = max(self._ys_raw) if self._ys_raw else 1.0
            self._ys_norm = [y / max(mx, 1e-9) for y in self._ys_raw]

        if self._ys_norm:
            self._last_norm_y = self._ys_norm[-1]

        # Setup animation targets.
        self._ys_norm_target = list(self._ys_norm)
        if (not self._ys_norm_disp) or (len(self._ys_norm_disp) != len(self._ys_norm_target)):
            self._ys_norm_disp = list(self._ys_norm_target)

        self._mse_dirty = True

    def _animate_mse(self) -> None:
        """
        Perform one animation step for bar chart values.

        The goal is to avoid sudden jumps in bar heights by linearly interpolating
        between current display values and new target values.
        """
        if not self._ys_norm_target:
            self._mse_dirty = False
            return

        if not self._ys_norm_disp:
            self._ys_norm_disp = list(self._ys_norm_target)
            self._update_mse_bars()
            self._mse_dirty = False
            return

        if len(self._ys_norm_disp) != len(self._ys_norm_target):
            self._ys_norm_disp = list(self._ys_norm_target)
            self._update_mse_bars()
            self._mse_dirty = False
            return

        disp = np.array(self._ys_norm_disp, dtype=float)
        target = np.array(self._ys_norm_target, dtype=float)

        alpha = self.anim_alpha
        new_disp = disp + alpha * (target - disp)

        if np.max(np.abs(new_disp - target)) < 1e-3:
            new_disp = target
            self._mse_dirty = False

        self._ys_norm_disp = new_disp.tolist()
        self._update_mse_bars()

    def _update_mse_bars(self) -> None:
        """
        Update the BarGraphItems (green/yellow/red) with current normalized data.

        This method:
        - splits normalized values into three regions:
            * < warn_thr   → green bars
            * [warn_thr, alarm_thr) → yellow bars
            * >= alarm_thr → red bars
        - sets X-range of the plot to the latest window indices.
        """
        ys = self._ys_norm_disp or self._ys_norm
        n = len(ys)
        if n == 0 or self._win_counter < 0:
            return

        y_arr = np.array(ys, dtype=float)

        xs_start = self._win_counter - n + 1
        xs = np.arange(xs_start, self._win_counter + 1, dtype=float)

        warn = self.warn_thr
        alarm = self.alarm_thr

        low_mask = y_arr < warn
        mid_mask = (y_arr >= warn) & (y_arr < alarm)
        high_mask = y_arr >= alarm

        self.bars_low.setOpts(x=xs[low_mask], height=y_arr[low_mask])
        self.bars_mid.setOpts(x=xs[mid_mask], height=y_arr[mid_mask])
        self.bars_high.setOpts(x=xs[high_mask], height=y_arr[high_mask])

        left = max(0, self._win_counter - self.max_points + 1)
        right = self._win_counter + 1
        self.mse_plot.setXRange(left, right, padding=0.05)

    def _update_mse_plot_from_file_tail(self) -> None:
        """
        If streaming JSON is disabled, incrementally read MSE values from CSV.

        This method:
        - seeks to previous file position (`_mse_pos`),
        - reads new rows from `windows_mse.csv`,
        - pushes each MSE value into `_push_mse`,
        - stores the new file position for the next tick.
        """
        try:
            if not self._mse_file or not os.path.exists(self._mse_file):
                return
            with open(self._mse_file, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self._mse_pos)
                rdr = csv.reader(f)
                for row in rdr:
                    if len(row) >= 3:
                        try:
                            self._push_mse(float(row[2]))
                        except Exception:
                            pass
                self._mse_pos = f.tell()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # E-mail sending
    # -------------------------------------------------------------------------

    def send_email(self, subject: str, body: str) -> None:
        """
        Send an e-mail using the SMTP configuration from the GUI.

        Parameters
        ----------
        subject : str
            E-mail subject line.
        body : str
            Plain-text body.

        Notes
        -----
        - Uses STARTTLS on the configured SMTP host/port.
        - Requires application password (e.g. for Gmail).
        """
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.email_from.text().strip()
            msg["To"] = self.email_to.text().strip()
            with smtplib.SMTP(self.email_host.text().strip(), int(self.email_port.text().strip())) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(msg["From"], self.email_pass.text().strip())
                s.send_message(msg)
            self.log_print(f"[EMAIL] Sent → {msg['To']}")
        except Exception as e:  # noqa: BLE001
            self.log_print(f"[EMAIL ERROR] {e}")

    # -------------------------------------------------------------------------
    # Window-level events → Alerts / Snapshots / Zero-Day handling
    # -------------------------------------------------------------------------

    def _handle_window_event(
        self,
        pred: str,
        pmax,
        rmse,
        raw: str,
        out_dir: str
    ) -> None:
        """
        Handle a per-window classification event and decide if it becomes a full event.

        Logic
        -----
        - For each raw class (`pred`), maintain a streak counter of confident windows.
        - A window is considered confident if `pmax >= confident_thr`.
        - When streak reaches `stitch_min_len` and refractory time has passed:
            * reset streak for that class,
            * create a logical event,
            * show a popup,
            * store PNG/TXT/JSON snapshot,
            * optionally send an e-mail with response hints.

        Parameters
        ----------
        pred : str
            Raw class prediction (e.g. 'http_flood', 'unknown_anomaly').
        pmax : float or None
            Maximum predicted probability for this window (can be None).
        rmse : float or None
            Reconstruction MSE for this window (can be None).
        raw : str
            Raw log line as emitted by the runner.
        out_dir : str
            Output directory for this PCAP/LIVE run (per PCAP subdir in offline mode).
        """
        if pred is None:
            return

        # Map raw class into canonical bucket and mark Zero-Day if needed.
        canonical_cls, is_zd = normalize_class_name(pred)

        now = datetime.datetime.now()
        try:
            st_min = max(1, int(self.ent_st_min.text().strip()))
        except Exception:
            st_min = 8

        st = self._class_state.get(pred, {
            "streak": 0,
            "last_ts": now - datetime.timedelta(hours=1),
            "email_sent": 0,
        })

        confident = (pmax is not None and pmax >= self.confident_thr)

        # Update streak depending on confidence.
        if confident:
            st["streak"] += 1
        else:
            # Small decay to avoid infinite streaks from occasional mis-matches.
            st["streak"] = max(0, st["streak"] - 1)

        should_event = (
            st["streak"] >= st_min
            and (now - st["last_ts"]).total_seconds() >= self.refractory_sec
        )

        self._class_state[pred] = st
        if not should_event:
            return

        # Turn streak into an event and reset streak.
        st["streak"] = 0
        st["last_ts"] = now
        self._class_state[pred] = st

        # Show popup to operator.
        self._popup_alert(pred, canonical_cls, pmax, rmse, raw)

        # Build per-class event directory: OUT/.../events/<canonical_cls>.
        cls_dir = os.path.join(out_dir, "events", canonical_cls)
        os.makedirs(cls_dir, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(cls_dir, f"event_{ts}.png")
        txt_path = os.path.join(cls_dir, f"event_{ts}.txt")
        json_path = os.path.join(cls_dir, f"event_{ts}.json")

        # Determine severity from normalized MSE.
        severity = "low"
        if self._last_norm_y is not None:
            if self._last_norm_y >= self.alarm_thr:
                severity = "high"
            elif self._last_norm_y >= self.warn_thr:
                severity = "medium"

        response_hint = build_response_hint(canonical_cls)

        try:
            # Capture a snapshot of the MSE plot.
            pix = self.mse_plot.grab()
            pix.save(png_path, "PNG")

            # Write human-readable TXT summary (includes Zero-Day flag and hints).
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(
                    f"RAW_CLASS: {pred}\n"
                    f"CANONICAL_CLASS: {canonical_cls}\n"
                    f"IS_ZERO_DAY: {bool(is_zd)}\n"
                    f"Pmax: {('NA' if pmax is None else f'{pmax:.4f}')}\n"
                    f"Recon MSE (raw): {('NA' if rmse is None else f'{rmse:.4f}')}\n"
                    f"Recon MSE (norm UI): {('NA' if self._last_norm_y is None else f'{self._last_norm_y:.4f}')}\n"
                    f"Severity: {severity}\n"
                    f"Time: {now}\n\n"
                    f"{response_hint}\n\n"
                    f"RAW LINE:\n{raw}\n"
                )

            # Write machine-readable JSON record for future retraining / analysis.
            event_record = {
                "timestamp_iso": now.isoformat(),
                "pcap_dir": out_dir,
                "raw_class": pred,
                "canonical_class": canonical_cls,
                "is_zero_day": bool(is_zd),
                "pmax": None if pmax is None else float(pmax),
                "recon_mse_raw": None if rmse is None else float(rmse),
                "recon_mse_norm_ui": None if self._last_norm_y is None else float(self._last_norm_y),
                "severity": severity,
                "response_hint": response_hint,
                "log_line": raw,
                "snapshot_png": os.path.basename(png_path),
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(event_record, jf, indent=2)
        except Exception:
            # We do not interrupt runner because of snapshot errors.
            pass

        self.log_print(f"[EVENT] {pred} → {canonical_cls} pmax={pmax} streak→event; snapshot={png_path}")

        # E-mail alert if all required fields are provided and rate limit not exceeded.
        if (
            self.email_to.text().strip()
            and self.email_from.text().strip()
            and self.email_pass.text().strip()
            and st["email_sent"] < self.max_alerts_per_class
        ):
            pmax_str = "NA" if pmax is None else f"{pmax:.2f}"
            mse_norm_str = "NA" if self._last_norm_y is None else f"{self._last_norm_y:.3f}"
            subj_prefix = "[SmartNetGuard][ZeroDay]" if is_zd else "[SmartNetGuard]"
            subj = f"{subj_prefix} {canonical_cls} event (pmax={pmax_str})"
            body = (
                f"RAW_CLASS: {pred}\n"
                f"CANONICAL_CLASS: {canonical_cls}\n"
                f"IS_ZERO_DAY: {bool(is_zd)}\n"
                f"Pmax: {pmax_str}\n"
                f"Recon MSE (norm UI): {mse_norm_str}\n"
                f"Severity: {severity}\n"
                f"Time: {now}\n\n"
                f"{response_hint}\n\n"
                f"RAW LINE:\n{raw}\n"
                f"(Snapshot & JSON saved in: {cls_dir})"
            )
            self.send_email(subj, body)
            st["email_sent"] += 1
            self._class_state[pred] = st

    def _popup_alert(
        self,
        raw_cls: str,
        canonical_cls: str,
        pmax,
        rmse,
        raw: str
    ) -> None:
        """
        Show a blinking popup dialog for a newly created event.

        Parameters
        ----------
        raw_cls : str
            Raw class name as returned by runner/policy.
        canonical_cls : str
            Canonical class after normalization (e.g. 'unknown_anomaly').
        pmax : float or None
            Maximum predicted probability for the window triggering the event.
        rmse : float or None
            Reconstruction MSE for that window.
        raw : str
            Raw log line, embedded into popup for debugging.
        """
        response_hint = build_response_hint(canonical_cls)
        msg = (
            f"RAW CLASS: {raw_cls}\n"
            f"CANONICAL CLASS: {canonical_cls}\n"
            f"Pmax: {('NA' if pmax is None else f'{pmax:.3f}')}\n"
            f"Recon MSE (raw): {('NA' if rmse is None else f'{rmse:.4f}')}\n"
            f"Recon MSE (norm UI): {('NA' if self._last_norm_y is None else f'{self._last_norm_y:.3f}')}\n"
            f"Time: {datetime.datetime.now()}\n\n"
            f"{response_hint}\n\n"
            f"RAW LINE:\n{raw}"
        )
        dlg = AlertDialog(self, msg)
        dlg.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Make sure icon appears in Windows taskbar (AppUserModelID hack).
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SmartNetGuard.GUI")
    except Exception:
        # On non-Windows or older Windows versions this may fail — it's safe to ignore.
        pass

    app = QtWidgets.QApplication(sys.argv)

    # Set global application icon if available.
    if os.path.exists(ICON_PATH):
        icon = QtGui.QIcon(ICON_PATH)
        app.setWindowIcon(icon)
    else:
        icon = None

    # Create and show main window.
    win = SmartNetGuardWindow()
    if icon is not None:
        win.setWindowIcon(icon)

    win.show()
    sys.exit(app.exec())
