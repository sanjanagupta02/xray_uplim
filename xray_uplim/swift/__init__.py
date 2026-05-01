"""
xray_uplim.swift
----------------
Swift XRT upper-limit pipeline.

Supports both PC (Photon Counting) and WT (Window Timing) readout modes,
detected automatically from the event file DATAMODE header keyword.
Results are reported as a single XRT instrument entry.
"""

from .pipeline import run_uplim, process_observation
from .config   import SwiftConfig
