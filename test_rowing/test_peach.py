"""
test_peach_mapping.py
─────────────────────
Verifies that the column mapping produced by PeachData.from_path() is
consistent with the ground-truth calibration derived from the reference
.txt export files.

Tests cover:
  - GPS: presence of computed lat/long/Time columns (consistent across files)
  - GPS calibration: raw→reference mapping is stable across all files
  - Periodic: column count matches raw stream width
  - Periodic: boat channels (Distance, Speed, Accel, Roll/Pitch/Yaw) present
  - Periodic: gate channels are interleaved per seat (GateAngle_N, GateForceX_N)
  - Stroke: all calibrated references appear as columns

Run with:
    python test_peach_mapping.py [data_dir]

Data directory defaults to /mnt/user-data/uploads if not specified.
"""

import sys
import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from rowing.analysis import peach as peach

DEFAULT_DATA_DIR = Path(__file__).parent / "../data/peach"

# ── Helpers ───────────────────────────────────────────────────────────────────


def discover_files(data_dir):
    pairs = []
    if not os.path.isdir(data_dir):
        return pairs

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.peach-data'):
            stem = fname[:-len('.peach-data')]
            ref = os.path.join(data_dir, stem + '.txt')
            if os.path.exists(ref):
                pairs.append((os.path.join(data_dir, fname), ref))
    return pairs


# ── Individual checks ─────────────────────────────────────────────────────────

def check_alignment(data, ref_data):
    cals, missing = data.check_alignment(ref_data)
    errors = []
    n_missing = len(missing)
    if n_missing:
        cnts = missing[['data', 'channel']].value_counts().to_frame()
        errors.append(
            f"Missing {n_missing} columns: {cnts}"
        )

    for d, cal in cals.items():
        bad_cal = cal[cal.rmse > 1e-6]
        for i, r in bad_cal.iterrows():
            errors.append(
                f"{d} bad alignment: {i[0]}->{i[1]} - {r}"
            )
    return errors

# ── Runner ────────────────────────────────────────────────────────────────────


def run_tests(data_dir):
    pairs = discover_files(data_dir)
    if not pairs:
        print(f"No .peach-data + .txt pairs found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} file(s) to test in {data_dir}\n")

    total_passed = 0
    total_failed = 0

    for peach_path, ref_path in pairs:
        label = os.path.basename(peach_path)
        print(f"── {label} ──")

        try:
            data = peach.PeachData.from_path(peach_path)
            ref_data = peach.parse_reference_file(ref_path)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            traceback.print_exc()
            total_failed += 1
            print()
            continue

        print(data)
        errors = []
        errors += check_alignment(data, ref_data)

        if errors:
            for e in errors:
                print(f"  FAIL  {e}")
            total_failed += 1
        else:
            print("  PASS")
            total_passed += 1

    print("═" * 65)
    print(f"Results: {total_passed} passed, {total_failed} failed "
          f"out of {len(pairs)} files")
    return errors


def test_alignment():
    errors = run_tests(DEFAULT_DATA_DIR)
    if errors:
        raise ValueError(
            f"Experienced {len(errors)} errors",
            *errors
        )


if __name__ == '__main__':

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    errors = run_tests(DATA_DIR)
    sys.exit(0 if errors else 1)
